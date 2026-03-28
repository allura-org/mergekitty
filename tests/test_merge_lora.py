import os

import torch
from click.testing import CliRunner
from common import make_picollama, make_tokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from mergekitty.common import AdapterReference, ModelPath, ModelReference
from mergekitty.io import LazyTensorLoader
from mergekitty.io.tasks import LoadTensor
from mergekitty.options import MergeOptions
from mergekitty.scripts.merge_lora import _reconstruct_invocation, main, plan_merge
from mergekitty.task import Task


def _walk_tasks(tasks: list[Task]):
    seen = set()
    stack = list(tasks)
    while stack:
        task = stack.pop()
        if task in seen:
            continue
        seen.add(task)
        yield task
        stack.extend(task.arguments().values())


def _build_adapter(
    base_model_path: str,
    adapter_path: str,
    *,
    target_modules: list[str],
    modules_to_save: list[str] | None = None,
    resized_vocab_size: int | None = None,
) -> None:
    model = LlamaForCausalLM.from_pretrained(base_model_path)
    if resized_vocab_size is not None:
        model.resize_token_embeddings(resized_vocab_size)

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=target_modules,
            modules_to_save=modules_to_save or [],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                values = torch.linspace(
                    -0.25,
                    0.25,
                    param.numel(),
                    dtype=param.dtype,
                    device=param.device,
                ).reshape(param.shape)
                param.copy_(values)
            elif "modules_to_save" in name:
                param.add_(0.125)

    peft_model.save_pretrained(adapter_path)


def _expected_merged_model(
    base_model_path: str,
    adapter_path: str,
    *,
    resized_vocab_size: int | None = None,
) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(base_model_path)
    if resized_vocab_size is not None:
        model.resize_token_embeddings(resized_vocab_size)
    return PeftModel.from_pretrained(model, adapter_path).merge_and_unload()


def _assert_saved_model_matches(
    output_path: str, expected_model: LlamaForCausalLM
) -> None:
    loader = LazyTensorLoader.from_disk(output_path, lazy_unpickle=False)
    expected_state = expected_model.state_dict()

    for tensor_name in sorted(loader.index.tensor_paths):
        actual = loader.get_tensor(tensor_name)
        expected = expected_state[tensor_name].cpu()
        if torch.is_floating_point(actual):
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4), tensor_name
        else:
            assert torch.equal(actual, expected), tensor_name


class TestMergeLora:
    def test_merges_standard_lora_adapter(self, tmp_path):
        base_model_path = make_picollama(tmp_path / "base-model")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(base_model_path)

        adapter_path = tmp_path / "adapter"
        _build_adapter(
            base_model_path,
            str(adapter_path),
            target_modules=["q_proj"],
        )

        output_path = tmp_path / "merged"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                base_model_path,
                str(adapter_path),
                str(output_path),
                "--compute-device",
                "cpu",
                "--storage-device",
                "cpu",
            ],
        )

        assert result.exit_code == 0, result.output
        assert os.path.exists(output_path / "model.safetensors.index.json")
        assert os.path.exists(output_path / "config.json")
        assert os.path.exists(output_path / "README.md")
        assert os.path.exists(output_path / "tokenizer.json")

        expected_model = _expected_merged_model(base_model_path, str(adapter_path))
        _assert_saved_model_matches(str(output_path), expected_model)

        config = AutoConfig.from_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        assert config.vocab_size == 64
        assert len(tokenizer) == 64

    def test_merges_extended_vocab_and_modules_to_save(self, tmp_path):
        base_model_path = make_picollama(tmp_path / "base-model", vocab_size=32)
        make_tokenizer(vocab_size=32, added_tokens=[]).save_pretrained(base_model_path)

        adapter_path = tmp_path / "adapter"
        _build_adapter(
            base_model_path,
            str(adapter_path),
            target_modules=["q_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            resized_vocab_size=40,
        )
        make_tokenizer(vocab_size=40, added_tokens=[]).save_pretrained(adapter_path)

        output_path = tmp_path / "merged"
        runner = CliRunner()
        result = runner.invoke(
            main, [base_model_path, str(adapter_path), str(output_path)]
        )

        assert result.exit_code == 0, result.output
        assert os.path.exists(output_path / "README.md")
        assert os.path.exists(output_path / "tokenizer.json")

        expected_model = _expected_merged_model(
            base_model_path,
            str(adapter_path),
            resized_vocab_size=40,
        )
        _assert_saved_model_matches(str(output_path), expected_model)

        config = AutoConfig.from_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        assert config.vocab_size == 40
        assert len(tokenizer) == 40

        readme = (output_path / "README.md").read_text(encoding="utf-8")
        assert "extended vocabulary" in readme

    def test_plan_merge_loads_lora_tensors_onto_storage_device(self, tmp_path):
        base_model_path = make_picollama(tmp_path / "base-model")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(base_model_path)

        adapter_path = tmp_path / "adapter"
        _build_adapter(
            base_model_path,
            str(adapter_path),
            target_modules=["q_proj"],
        )

        plan = plan_merge(
            base_model_ref=ModelReference.model_validate(base_model_path),
            adapter_ref=AdapterReference(
                adapter=ModelPath.model_validate(str(adapter_path))
            ),
            output_path=str(tmp_path / "merged"),
            options=MergeOptions(storage_device="cuda"),
        )

        load_tasks = [
            task for task in _walk_tasks(plan.tasks) if isinstance(task, LoadTensor)
        ]
        assert load_tasks
        assert {task.device for task in load_tasks} == {"cuda"}

    def test_plan_merge_can_load_lora_tensors_onto_compute_device(self, tmp_path):
        base_model_path = make_picollama(tmp_path / "base-model")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(base_model_path)

        adapter_path = tmp_path / "adapter"
        _build_adapter(
            base_model_path,
            str(adapter_path),
            target_modules=["q_proj"],
        )

        plan = plan_merge(
            base_model_ref=ModelReference.model_validate(base_model_path),
            adapter_ref=AdapterReference(
                adapter=ModelPath.model_validate(str(adapter_path))
            ),
            output_path=str(tmp_path / "merged"),
            options=MergeOptions(
                compute_device="cuda",
                storage_device="cpu",
                load_to_compute=True,
            ),
        )

        load_tasks = [
            task for task in _walk_tasks(plan.tasks) if isinstance(task, LoadTensor)
        ]
        assert load_tasks
        assert {task.device for task in load_tasks} == {"cuda"}

    def test_reconstruct_invocation_uses_new_device_flags(self):
        invocation = _reconstruct_invocation(
            base_model="base-model",
            adapter_model="adapter-model",
            options=MergeOptions(
                compute_device="cuda",
                storage_device="cpu",
                load_to_compute=True,
            ),
        )

        assert "--compute-device cuda" in invocation
        assert "--load-to-compute" in invocation
        assert "--storage-device" not in invocation
        assert "--cuda" not in invocation
        assert "--read-to-gpu" not in invocation
        assert "--low-cpu-memory" not in invocation
