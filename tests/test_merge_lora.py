import os

import torch
from click.testing import CliRunner
from common import make_picollama, make_tokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from mergekitty.io import LazyTensorLoader
from mergekitty.scripts.merge_lora import main


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
            main, [base_model_path, str(adapter_path), str(output_path)]
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
