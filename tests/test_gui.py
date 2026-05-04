from pathlib import Path

import pytest

from mergekitty.scripts import gui


LINEAR_YAML = """
merge_method: linear
models:
  - model: model-a
    parameters:
      weight: 0.25
  - model: model-b
    parameters:
      weight: 0.75
"""


def test_parameter_templates_include_expected_fields():
    method = gui.REGISTERED_MERGE_METHODS["slerp"]

    block_template = gui.block_parameter_template(method.parameters())
    inline_template = gui.inline_parameter_template(method.tensor_parameters())

    assert "t:" in block_template
    assert "nuslerp_flatten: true" in block_template
    assert inline_template == "{weight: null}"


def test_build_builder_configuration_supports_lora_and_custom_tokenizer():
    config = gui.build_builder_configuration(
        method_name="linear",
        base_model="",
        dtype="bfloat16",
        out_dtype="float16",
        tokenizer_source="tokenizer-org/tokenizer-model",
        chat_template="chatml",
        global_parameters="normalize: false",
        model_rows=[
            ["model-a", "main", "", "", "", "{weight: 0.4}"],
            [
                "model-b",
                "",
                "adapter-b",
                "dev",
                "LlamaForCausalLM",
                "{weight: 0.6}",
            ],
        ],
    )

    assert config.merge_method == "linear"
    assert config.dtype == "bfloat16"
    assert config.out_dtype == "float16"
    assert config.chat_template == "chatml"
    assert config.parameters == {"normalize": False}
    assert str(config.tokenizer_source) == "tokenizer-org/tokenizer-model"
    assert config.models[0].model.model.path == "model-a"
    assert config.models[0].model.model.revision == "main"
    assert config.models[1].model.lora.path == "adapter-b"
    assert config.models[1].model.lora.revision == "dev"
    assert config.models[1].model.override_architecture == "LlamaForCausalLM"


def test_build_builder_configuration_requires_base_model_for_model_stock():
    with pytest.raises(ValueError, match="requires a base model"):
        gui.build_builder_configuration(
            method_name="model_stock",
            base_model="",
            dtype="",
            out_dtype="",
            tokenizer_source="base",
            chat_template="auto",
            global_parameters="filter_wise: false",
            model_rows=[
                ["model-a", "", "", "", "", ""],
                ["model-b", "", "", "", "", ""],
                ["model-c", "", "", "", "", ""],
            ],
        )


def test_build_builder_configuration_rejects_missing_required_model_parameter():
    with pytest.raises(ValueError, match="missing a value for `weight`"):
        gui.build_builder_configuration(
            method_name="linear",
            base_model="",
            dtype="",
            out_dtype="",
            tokenizer_source="base",
            chat_template="auto",
            global_parameters="normalize: true",
            model_rows=[["model-a", "", "", "", "", "{weight: null}"]],
        )


def test_build_builder_configuration_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unsupported parameters: nope"):
        gui.build_builder_configuration(
            method_name="linear",
            base_model="",
            dtype="",
            out_dtype="",
            tokenizer_source="base",
            chat_template="auto",
            global_parameters="normalize: true",
            model_rows=[["model-a", "", "", "", "", "{weight: 1.0, nope: 2}"]],
        )


def test_run_merge_from_yaml_writes_local_output_and_uploads(tmp_path, monkeypatch):
    calls = {}
    log_file = tmp_path / "merge.log"
    output_directory = tmp_path / "merged-model"

    def fake_run_merge(config, out_path, options, config_source=None):
        calls["config"] = config
        calls["out_path"] = out_path
        calls["options"] = options
        calls["config_source"] = config_source
        Path(out_path).mkdir(parents=True, exist_ok=True)
        (Path(out_path) / "README.md").write_text("ok", encoding="utf-8")

    def fake_create_repo(**kwargs):
        calls["create_repo"] = kwargs

    def fake_upload_folder(**kwargs):
        calls["upload_folder"] = kwargs

    monkeypatch.setattr(gui, "run_merge", fake_run_merge)
    monkeypatch.setattr(gui.hf, "create_repo", fake_create_repo)
    monkeypatch.setattr(gui.hf, "upload_folder", fake_upload_folder)

    status = gui.run_merge_from_yaml(
        yaml_config=LINEAR_YAML,
        log_file=str(log_file),
        output_directory=str(output_directory),
        upload_repo_id="user/test-merge",
        upload_private=True,
        upload_token="secret-token",
        compute_device="cpu",
        storage_device="cpu",
        allow_crimes=False,
        trust_remote_code=False,
        lazy_unpickle=True,
        load_to_compute=False,
        write_model_card=True,
        quiet=False,
        executor="single",
        random_seed=7,
    )

    resolved_output = str(output_directory.resolve())
    assert "uploaded to `https://huggingface.co/user/test-merge`" in status
    assert calls["out_path"] == resolved_output
    assert calls["config"].merge_method == "linear"
    assert calls["options"].random_seed == 7
    assert calls["create_repo"]["repo_id"] == "user/test-merge"
    assert calls["create_repo"]["private"] is True
    assert calls["upload_folder"]["folder_path"] == resolved_output
    assert "Starting merge with method: linear" in log_file.read_text(encoding="utf-8")


def test_run_merge_from_yaml_logs_failures(tmp_path, monkeypatch):
    log_file = tmp_path / "merge.log"

    def fake_run_merge(*_args, **_kwargs):
        raise RuntimeError("merge exploded")

    monkeypatch.setattr(gui, "run_merge", fake_run_merge)

    status = gui.run_merge_from_yaml(
        yaml_config=LINEAR_YAML,
        log_file=str(log_file),
        output_directory=str(tmp_path / "merged-model"),
        upload_repo_id="",
        upload_private=True,
        upload_token="",
        compute_device="cpu",
        storage_device="cpu",
        allow_crimes=False,
        trust_remote_code=False,
        lazy_unpickle=True,
        load_to_compute=False,
        write_model_card=True,
        quiet=False,
        executor="single",
        random_seed=None,
    )

    log_contents = log_file.read_text(encoding="utf-8")
    assert "merge exploded" in status
    assert "Traceback" in log_contents
    assert "merge exploded" in log_contents
