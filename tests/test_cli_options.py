import json

import click
import pytest
import torch
from click.testing import CliRunner
from common import make_picollama, make_tokenizer
from transformers import AutoConfig

from mergekitty.architecture import get_architecture_info
from mergekitty.common import ModelReference
from mergekitty.config import InputModelDefinition, MergeConfiguration
from mergekitty.io.tasks import GatherTensors, LoadTensor
from mergekitty.merge import _model_out_config
from mergekitty.options import MergeOptions, add_merge_options
from mergekitty.plan import MergePlanner
from mergekitty.task import Task


@click.command("dump-merge-options")
@add_merge_options
def dump_merge_options(merge_options: MergeOptions) -> None:
    click.echo(merge_options.model_dump_json())


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


class TestCliOptions:
    def test_help_lists_new_device_flags_only(self):
        runner = CliRunner()
        result = runner.invoke(dump_merge_options, ["--help"])

        assert result.exit_code == 0, result.output
        assert "--compute-device" in result.output
        assert "--storage-device" in result.output
        assert "--load-to-compute" in result.output
        assert "--cuda" not in result.output
        assert "--read-to-gpu" not in result.output
        assert "--low-cpu-memory" not in result.output

    def test_defaults_use_cpu_for_compute_and_storage(self):
        runner = CliRunner()
        result = runner.invoke(dump_merge_options, [])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["compute_device"] == "cpu"
        assert payload["storage_device"] == "cpu"
        assert payload["load_to_compute"] is False
        assert "cuda" not in payload
        assert "read_to_gpu" not in payload
        assert "low_cpu_memory" not in payload

    def test_accepts_explicit_device_choices_case_insensitively(self):
        runner = CliRunner()
        result = runner.invoke(
            dump_merge_options,
            [
                "--compute-device",
                "CUDA",
                "--storage-device",
                "Cpu",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["compute_device"] == "cuda"
        assert payload["storage_device"] == "cpu"

    def test_accepts_load_to_compute_flag(self):
        runner = CliRunner()
        result = runner.invoke(
            dump_merge_options,
            [
                "--compute-device",
                "cuda",
                "--storage-device",
                "cpu",
                "--load-to-compute",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["compute_device"] == "cuda"
        assert payload["storage_device"] == "cpu"
        assert payload["load_to_compute"] is True

    def test_rejects_invalid_device_choice(self):
        runner = CliRunner()
        result = runner.invoke(dump_merge_options, ["--compute-device", "gpu"])

        assert result.exit_code != 0
        assert "Invalid value for '--compute-device'" in result.output
        assert "cpu" in result.output
        assert "cuda" in result.output

    @pytest.mark.parametrize(
        "legacy_flag",
        [
            "--cuda",
            "--no-cuda",
            "--read-to-gpu",
            "--no-read-to-gpu",
            "--low-cpu-memory",
            "--no-low-cpu-memory",
        ],
    )
    def test_rejects_legacy_device_flags(self, legacy_flag: str):
        runner = CliRunner()
        result = runner.invoke(dump_merge_options, [legacy_flag])

        assert result.exit_code != 0
        assert "No such option" in result.output


class TestPlannerDeviceSelection:
    def test_yaml_merge_loads_tensors_onto_storage_device(self, tmp_path):
        model_path = make_picollama(tmp_path / "model")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        merge_config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=model_path)],
            dtype="bfloat16",
        )
        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        planner = MergePlanner(
            config=merge_config,
            arch_info=arch_info,
            options=MergeOptions(storage_device="cuda"),
            out_model_config=_model_out_config(merge_config, arch_info),
        )

        tasks = planner.plan_to_disk(str(tmp_path / "out"))
        gather_tasks = [
            task for task in _walk_tasks(tasks) if isinstance(task, GatherTensors)
        ]

        assert gather_tasks
        assert {task.device for task in gather_tasks} == {"cuda"}

    def test_yaml_merge_can_load_tensors_onto_compute_device(self, tmp_path):
        model_path = make_picollama(tmp_path / "model")
        make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)

        merge_config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=model_path)],
            dtype="bfloat16",
        )
        cfg = AutoConfig.from_pretrained(model_path)
        arch_info = get_architecture_info(cfg)
        planner = MergePlanner(
            config=merge_config,
            arch_info=arch_info,
            options=MergeOptions(
                compute_device="cuda",
                storage_device="cpu",
                load_to_compute=True,
            ),
            out_model_config=_model_out_config(merge_config, arch_info),
        )

        tasks = planner.plan_to_disk(str(tmp_path / "out"))
        gather_tasks = [
            task for task in _walk_tasks(tasks) if isinstance(task, GatherTensors)
        ]

        assert gather_tasks
        assert {task.device for task in gather_tasks} == {"cuda"}

    def test_load_tensor_keeps_requested_load_device_between_tasks(self):
        task = LoadTensor(
            model=ModelReference.model_validate("model"),
            tensor="weight",
            device="cuda",
        )

        assert task.result_storage_device(
            execution_device=torch.device("cpu"),
            default_storage_device=torch.device("cpu"),
        ) == torch.device("cuda")
