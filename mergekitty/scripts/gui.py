# Copyright (C) 2025-2026 Allura-org

from __future__ import annotations

import os
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable, Optional

import click
import huggingface_hub as hf
import torch
import yaml

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "gradio is not installed. Please install mergekitty with "
        "`pip install mergekitty[gradio]`."
    ) from exc

try:
    from gradio_log import Log
except ImportError:
    Log = None

try:
    from gradio_huggingfacehub_search import HuggingfaceHubSearch
except ImportError:
    HuggingfaceHubSearch = None

from mergekitty.common import ModelReference
from mergekitty.config import MergeConfiguration
from mergekitty.merge import run_merge
from mergekitty.merge_methods import REGISTERED_MERGE_METHODS
from mergekitty.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekitty.options import MergeOptions

MODEL_TABLE_HEADERS = [
    "Model",
    "Revision",
    "LoRA",
    "LoRA revision",
    "Override architecture",
    "Parameters",
]
MODEL_TABLE_DATATYPES = ["str"] * len(MODEL_TABLE_HEADERS)
CHAT_TEMPLATE_CHOICES = ["auto", "alpaca", "chatml", "llama3", "mistral", "exaone"]
DTYPE_CHOICES = ["", "bfloat16", "float16", "float32"]
BASE_MODEL_REQUIRED_METHODS = {
    "breadcrumbs",
    "breadcrumbs_ties",
    "consensus_ta",
    "consensus_ties",
    "dare_linear",
    "dare_ties",
    "della",
    "della_linear",
    "model_stock",
    "nearswap",
    "sce",
    "task_arithmetic",
    "ties",
}
DEFAULT_CONFIG_TEMPLATE = """merge_method: linear
models:
  - model: your-org/model-a
    parameters:
      weight: 0.5
  - model: your-org/model-b
    parameters:
      weight: 0.5
dtype: bfloat16
"""
CUSTOM_CSS = """
.mk-shell {
    gap: 1rem;
}

.mk-sidebar {
    border-right: 1px solid var(--border-color-primary);
    padding-right: 1rem;
}

.mk-main {
    padding-left: 0.25rem;
}
"""


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_method_name() -> str:
    return next(iter(REGISTERED_MERGE_METHODS))


def _method_choices() -> list[tuple[str, str]]:
    return [
        (f"{method.pretty_name() or name} ({name})", name)
        for name, method in REGISTERED_MERGE_METHODS.items()
    ]


def _get_method(method_name: str) -> MergeMethod:
    try:
        return REGISTERED_MERGE_METHODS[method_name]
    except KeyError as exc:
        raise ValueError(f"Unknown merge method: {method_name}") from exc


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _yaml_scalar(value: Any) -> str:
    dumped = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    return dumped.removesuffix("\n...")


def block_parameter_template(parameter_defs: Iterable[ConfigParameterDef]) -> str:
    lines = []
    for definition in parameter_defs:
        if definition.default_value is None:
            lines.append(f"{definition.name}:")
        else:
            lines.append(f"{definition.name}: {_yaml_scalar(definition.default_value)}")
    return "\n".join(lines)


def inline_parameter_template(parameter_defs: Iterable[ConfigParameterDef]) -> str:
    payload = {
        definition.name: definition.default_value for definition in parameter_defs
    }
    if not payload:
        return ""
    return yaml.safe_dump(payload, default_flow_style=True, sort_keys=False).strip()


def method_summary_markdown(method_name: str) -> str:
    method = _get_method(method_name)
    pretty_name = method.pretty_name() or method_name
    base_requirement = "Optional"
    if method_name in BASE_MODEL_REQUIRED_METHODS:
        base_requirement = "Required"
    elif method_name == "slerp":
        base_requirement = (
            "Required for classic SLERP (`t`), optional for NuSLERP (`weight`)"
        )

    lines = [
        f"### {pretty_name}",
        f"`merge_method: {method_name}`",
        f"Base model: {base_requirement}",
    ]
    if method.reference_url():
        lines.append(f"Reference: {method.reference_url()}")

    global_params = list(method.parameters())
    model_params = list(method.tensor_parameters())
    lines.append("")
    lines.append(
        "Global parameters: "
        + (
            ", ".join(param.name for param in global_params)
            if global_params
            else "none"
        )
    )
    lines.append(
        "Per-model parameters: "
        + (", ".join(param.name for param in model_params) if model_params else "none")
    )
    lines.append("")
    lines.append(
        "This builder emits `models:` configs. Use the Raw YAML tab for `slices:` "
        "configs and other advanced layouts."
    )
    return "\n".join(lines)


def blank_model_row(method_name: str, model: str = "") -> list[str]:
    return [
        model,
        "",
        "",
        "",
        "",
        inline_parameter_template(_get_method(method_name).tensor_parameters()),
    ]


def normalize_model_rows(rows: Any) -> list[list[str]]:
    if rows is None:
        return []
    if hasattr(rows, "values") and callable(getattr(rows.values, "tolist", None)):
        rows = rows.values.tolist()

    normalized_rows = []
    for row in rows:
        row_values = list(row) if isinstance(row, (list, tuple)) else [row]
        cells = [
            _normalize_string(value) for value in row_values[: len(MODEL_TABLE_HEADERS)]
        ]
        while len(cells) < len(MODEL_TABLE_HEADERS):
            cells.append("")
        normalized_rows.append(cells)
    return normalized_rows


def add_model_row(
    model: str, rows: Any, method_name: str
) -> tuple[str, list[list[str]]]:
    updated_rows = normalize_model_rows(rows)
    updated_rows.append(blank_model_row(method_name, _normalize_string(model)))
    return "", updated_rows


def add_blank_model_row(rows: Any, method_name: str) -> list[list[str]]:
    updated_rows = normalize_model_rows(rows)
    updated_rows.append(blank_model_row(method_name))
    return updated_rows


def reset_parameter_templates(
    method_name: str, rows: Any
) -> tuple[str, list[list[str]]]:
    global_template = block_parameter_template(_get_method(method_name).parameters())
    model_template = inline_parameter_template(
        _get_method(method_name).tensor_parameters()
    )

    updated_rows = []
    for row in normalize_model_rows(rows):
        new_row = row.copy()
        new_row[5] = model_template
        updated_rows.append(new_row)

    if not updated_rows:
        updated_rows.append(blank_model_row(method_name))
    return global_template, updated_rows


def update_method_summary(
    method_name: str, existing_global_parameters: str
) -> tuple[str, str]:
    global_parameters = _normalize_string(existing_global_parameters)
    if not global_parameters:
        global_parameters = block_parameter_template(
            _get_method(method_name).parameters()
        )
    return method_summary_markdown(method_name), global_parameters


def parse_parameter_mapping(raw_text: str, context: str) -> dict[str, Any]:
    text = _normalize_string(raw_text)
    if not text:
        return {}

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{context} is not valid YAML: {exc}") from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must be a YAML mapping")
    return parsed


def validate_parameter_mapping(
    raw_mapping: dict[str, Any],
    parameter_defs: Iterable[ConfigParameterDef],
    context: str,
) -> dict[str, Any]:
    definitions = {definition.name: definition for definition in parameter_defs}
    unknown = sorted(set(raw_mapping) - set(definitions))
    if unknown:
        raise ValueError(
            f"{context} contains unsupported parameters: {', '.join(unknown)}"
        )

    validated: dict[str, Any] = {}
    for name, definition in definitions.items():
        if name not in raw_mapping:
            if definition.required:
                raise ValueError(f"{context} is missing required parameter `{name}`")
            continue

        value = raw_mapping[name]
        if value is None:
            if definition.required:
                raise ValueError(f"{context} is missing a value for `{name}`")
            continue

        validated[name] = value

    return validated


def build_model_reference(
    model: str,
    revision: str = "",
    lora: str = "",
    lora_revision: str = "",
    override_architecture: str = "",
) -> ModelReference:
    model_path = _normalize_string(model)
    if not model_path:
        raise ValueError("Model path cannot be blank")

    payload: dict[str, Any] = {"model": {"path": model_path}}
    revision_value = _normalize_string(revision)
    if revision_value:
        payload["model"]["revision"] = revision_value

    lora_value = _normalize_string(lora)
    if lora_value:
        payload["lora"] = {"path": lora_value}
        lora_revision_value = _normalize_string(lora_revision)
        if lora_revision_value:
            payload["lora"]["revision"] = lora_revision_value

    override_value = _normalize_string(override_architecture)
    if override_value:
        payload["override_architecture"] = override_value

    return ModelReference.model_validate(payload)


def parse_tokenizer_source(value: str) -> Any:
    text = _normalize_string(value)
    if not text:
        return None

    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered in {"base", "union"}:
        return lowered
    return ModelReference.parse(text)


def build_builder_configuration(
    method_name: str,
    base_model: str,
    dtype: str,
    out_dtype: str,
    tokenizer_source: str,
    chat_template: str,
    global_parameters: str,
    model_rows: Any,
) -> MergeConfiguration:
    method = _get_method(method_name)

    builder_config: dict[str, Any] = {
        "merge_method": method_name,
        "models": [],
    }

    global_mapping = validate_parameter_mapping(
        parse_parameter_mapping(global_parameters, "Global parameters"),
        method.parameters(),
        "Global parameters",
    )
    if global_mapping:
        builder_config["parameters"] = global_mapping

    base_model_text = _normalize_string(base_model)
    if base_model_text:
        builder_config["base_model"] = ModelReference.parse(base_model_text)
    elif method_name in BASE_MODEL_REQUIRED_METHODS:
        raise ValueError(f"{method.pretty_name() or method_name} requires a base model")
    elif (
        method_name == "slerp"
        and builder_config.get("parameters", {}).get("t") is not None
    ):
        raise ValueError("SLERP with a global `t` requires a base model")

    for index, row in enumerate(normalize_model_rows(model_rows), start=1):
        if not any(_normalize_string(value) for value in row):
            continue

        model_path = row[0]
        if not model_path:
            raise ValueError(f"Model row {index} is missing a model path")

        model_entry: dict[str, Any] = {
            "model": build_model_reference(
                model=row[0],
                revision=row[1],
                lora=row[2],
                lora_revision=row[3],
                override_architecture=row[4],
            )
        }
        model_mapping = validate_parameter_mapping(
            parse_parameter_mapping(row[5], f"Model row {index} parameters"),
            method.tensor_parameters(),
            f"Model row {index} parameters",
        )
        if model_mapping:
            model_entry["parameters"] = model_mapping
        builder_config["models"].append(model_entry)

    if not builder_config["models"]:
        raise ValueError("Add at least one model before building a configuration")

    dtype_value = _normalize_string(dtype)
    if dtype_value:
        builder_config["dtype"] = dtype_value

    out_dtype_value = _normalize_string(out_dtype)
    if out_dtype_value:
        builder_config["out_dtype"] = out_dtype_value

    tokenizer_source_value = parse_tokenizer_source(tokenizer_source)
    if _normalize_string(tokenizer_source) or tokenizer_source_value is None:
        builder_config["tokenizer_source"] = tokenizer_source_value

    chat_template_value = _normalize_string(chat_template)
    if chat_template_value:
        builder_config["chat_template"] = chat_template_value

    return MergeConfiguration.model_validate(builder_config)


def parse_yaml_configuration(yaml_config: str) -> MergeConfiguration:
    try:
        parsed = yaml.safe_load(yaml_config)
    except yaml.YAMLError as exc:
        raise ValueError(f"Configuration YAML is invalid: {exc}") from exc

    if parsed is None:
        raise ValueError("Configuration YAML is empty")
    return MergeConfiguration.model_validate(parsed)


def build_options(
    compute_device: str,
    storage_device: str,
    allow_crimes: bool,
    trust_remote_code: bool,
    lazy_unpickle: bool,
    load_to_compute: bool,
    write_model_card: bool,
    quiet: bool,
    executor: str,
    random_seed: Optional[float],
) -> MergeOptions:
    seed = None if random_seed is None else int(random_seed)
    return MergeOptions(
        allow_crimes=bool(allow_crimes),
        trust_remote_code=bool(trust_remote_code),
        lazy_unpickle=bool(lazy_unpickle),
        load_to_compute=bool(load_to_compute),
        write_model_card=bool(write_model_card),
        quiet=bool(quiet),
        compute_device=_normalize_string(compute_device) or _default_device(),
        storage_device=_normalize_string(storage_device) or _default_device(),
        executor=_normalize_string(executor) or "single",
        random_seed=seed,
    )


def reset_log(log_file: str) -> None:
    with open(log_file, "w", encoding="utf-8"):
        pass


def status_message(message: str, *, error: bool = False) -> str:
    prefix = "Error" if error else "Status"
    return f"**{prefix}:** {message}"


def preview_builder_yaml(
    method_name: str,
    base_model: str,
    dtype: str,
    out_dtype: str,
    tokenizer_source: str,
    chat_template: str,
    global_parameters: str,
    model_rows: Any,
) -> tuple[Any, Any, str]:
    try:
        config = build_builder_configuration(
            method_name=method_name,
            base_model=base_model,
            dtype=dtype,
            out_dtype=out_dtype,
            tokenizer_source=tokenizer_source,
            chat_template=chat_template,
            global_parameters=global_parameters,
            model_rows=model_rows,
        )
    except Exception as exc:
        return gr.update(), gr.update(), status_message(str(exc), error=True)

    yaml_text = config.to_yaml()
    return yaml_text, yaml_text, status_message("Builder configuration validated.")


def validate_yaml_editor(yaml_config: str) -> tuple[Any, str]:
    try:
        config = parse_yaml_configuration(yaml_config)
    except Exception as exc:
        return gr.update(), status_message(str(exc), error=True)

    return config.to_yaml(), status_message("YAML configuration validated.")


def resolve_output_directory(output_directory: str) -> str:
    output_path = _normalize_string(output_directory)
    if not output_path:
        raise ValueError("Local output directory is required")
    return str(Path(output_path).expanduser().resolve())


def run_merge_from_yaml(
    yaml_config: str,
    log_file: str,
    output_directory: str,
    upload_repo_id: str,
    upload_private: bool,
    upload_token: str,
    compute_device: str,
    storage_device: str,
    allow_crimes: bool,
    trust_remote_code: bool,
    lazy_unpickle: bool,
    load_to_compute: bool,
    write_model_card: bool,
    quiet: bool,
    executor: str,
    random_seed: Optional[float],
) -> str:
    normalized_output_directory = ""
    try:
        config = parse_yaml_configuration(yaml_config)
        normalized_yaml = config.to_yaml()
        normalized_output_directory = resolve_output_directory(output_directory)
        options = build_options(
            compute_device=compute_device,
            storage_device=storage_device,
            allow_crimes=allow_crimes,
            trust_remote_code=trust_remote_code,
            lazy_unpickle=lazy_unpickle,
            load_to_compute=load_to_compute,
            write_model_card=write_model_card,
            quiet=quiet,
            executor=executor,
            random_seed=random_seed,
        )

        reset_log(log_file)
        with open(log_file, "a", encoding="utf-8") as log_handle:
            print(f"Starting merge with method: {config.merge_method}", file=log_handle)
            print(f"Output directory: {normalized_output_directory}", file=log_handle)
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                run_merge(
                    config,
                    normalized_output_directory,
                    options=options,
                    config_source=normalized_yaml,
                )

                repo_id = _normalize_string(upload_repo_id)
                if repo_id:
                    token = _normalize_string(upload_token) or hf.get_token()
                    print(f"Uploading output directory to {repo_id}", file=log_handle)
                    hf.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=bool(upload_private),
                        exist_ok=True,
                        token=token,
                    )
                    hf.upload_folder(
                        repo_id=repo_id,
                        folder_path=normalized_output_directory,
                        repo_type="model",
                        token=token,
                    )
                    print(
                        f"Upload finished: https://huggingface.co/{repo_id}",
                        file=log_handle,
                    )

        repo_id = _normalize_string(upload_repo_id)
        if repo_id:
            return status_message(
                "Merge complete. Output written to "
                f"`{normalized_output_directory}` and uploaded to "
                f"`https://huggingface.co/{repo_id}`."
            )
        return status_message(
            f"Merge complete. Output written to `{normalized_output_directory}`."
        )
    except Exception as exc:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as log_handle:
                traceback.print_exc(file=log_handle)

        output_hint = ""
        if normalized_output_directory:
            output_hint = f" Local output directory: `{normalized_output_directory}`."
        return status_message(
            f"{type(exc).__name__}: {exc}.{output_hint}",
            error=True,
        )


def build_and_run_from_builder(
    method_name: str,
    base_model: str,
    dtype: str,
    out_dtype: str,
    tokenizer_source: str,
    chat_template: str,
    global_parameters: str,
    model_rows: Any,
    log_file: str,
    output_directory: str,
    upload_repo_id: str,
    upload_private: bool,
    upload_token: str,
    compute_device: str,
    storage_device: str,
    allow_crimes: bool,
    trust_remote_code: bool,
    lazy_unpickle: bool,
    load_to_compute: bool,
    write_model_card: bool,
    quiet: bool,
    executor: str,
    random_seed: Optional[float],
) -> tuple[Any, Any, str]:
    try:
        config = build_builder_configuration(
            method_name=method_name,
            base_model=base_model,
            dtype=dtype,
            out_dtype=out_dtype,
            tokenizer_source=tokenizer_source,
            chat_template=chat_template,
            global_parameters=global_parameters,
            model_rows=model_rows,
        )
    except Exception as exc:
        return gr.update(), gr.update(), status_message(str(exc), error=True)

    yaml_text = config.to_yaml()
    status = run_merge_from_yaml(
        yaml_config=yaml_text,
        log_file=log_file,
        output_directory=output_directory,
        upload_repo_id=upload_repo_id,
        upload_private=upload_private,
        upload_token=upload_token,
        compute_device=compute_device,
        storage_device=storage_device,
        allow_crimes=allow_crimes,
        trust_remote_code=trust_remote_code,
        lazy_unpickle=lazy_unpickle,
        load_to_compute=load_to_compute,
        write_model_card=write_model_card,
        quiet=quiet,
        executor=executor,
        random_seed=random_seed,
    )
    return yaml_text, yaml_text, status


def build_model_input(label: str, *, value: str = ""):
    if HuggingfaceHubSearch is not None:
        return HuggingfaceHubSearch(label=label, value=value, interactive=True)
    return gr.Textbox(
        label=label,
        value=value,
        placeholder="org/model, local/path, or model+lora",
        interactive=True,
    )


def build_log_component(log_file: str):
    if Log is not None:
        return Log(log_file=log_file, dark=True, label="Logs", container=True)
    return gr.Textbox(
        label="Logs",
        value=f"Streaming log viewer unavailable. Log file: {log_file}",
        lines=12,
        interactive=False,
    )


def build_gui() -> gr.Blocks:
    default_method = _default_method_name()
    default_device = _default_device()
    log_file = tempfile.NamedTemporaryFile(
        prefix="mergekitty-gui-", suffix=".log", delete=False
    )
    log_file.close()

    with gr.Blocks(
        title="Mergekitty GUI",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as gui:
        log_file_state = gr.State(value=log_file.name)

        gr.Markdown(
            "# Mergekitty GUI\n"
            "Build common `models:` merges in the builder tab, or drop to raw YAML "
            "for advanced configs. Merges write locally first, with optional upload "
            "to Hugging Face after the merge succeeds."
        )

        with gr.Row(elem_classes=["mk-shell"]):
            with gr.Column(scale=4, min_width=320, elem_classes=["mk-sidebar"]):
                gr.Markdown("### Destination")
                output_directory = gr.Textbox(
                    label="Local output directory",
                    placeholder="~/models/my-merge",
                )
                upload_repo_id = gr.Textbox(
                    label="Optional Hugging Face repo ID",
                    placeholder="username/my-merge",
                )
                upload_private = gr.Checkbox(
                    label="Create uploaded repo as private",
                    value=True,
                )
                upload_token = gr.Textbox(
                    label="Optional Hugging Face token",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    placeholder="Uses your cached login if left blank",
                )

                with gr.Accordion("Execution options", open=False):
                    compute_device = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value=default_device,
                        label="Compute device",
                    )
                    storage_device = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value=default_device,
                        label="Storage device",
                    )
                    executor = gr.Dropdown(
                        choices=["single", "parallel"],
                        value="single",
                        label="Executor",
                    )
                    random_seed = gr.Number(
                        label="Random seed",
                        value=None,
                        precision=0,
                    )
                    allow_crimes = gr.Checkbox(
                        label="Allow architecture mixing",
                        value=False,
                    )
                    trust_remote_code = gr.Checkbox(
                        label="Trust remote code",
                        value=False,
                    )
                    lazy_unpickle = gr.Checkbox(
                        label="Use lazy unpickle",
                        value=True,
                    )
                    load_to_compute = gr.Checkbox(
                        label="Load tensors directly to compute device",
                        value=False,
                    )
                    write_model_card = gr.Checkbox(
                        label="Write README.md and config",
                        value=True,
                    )
                    quiet = gr.Checkbox(
                        label="Quiet merge execution",
                        value=False,
                    )

            with gr.Column(scale=8, min_width=640, elem_classes=["mk-main"]):
                with gr.Tabs():
                    with gr.Tab("Builder"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                merge_method = gr.Dropdown(
                                    label="Merge method",
                                    choices=_method_choices(),
                                    value=default_method,
                                )
                                method_summary = gr.Markdown(
                                    method_summary_markdown(default_method)
                                )
                                base_model = build_model_input("Optional base model")
                                dtype = gr.Dropdown(
                                    choices=DTYPE_CHOICES,
                                    value="",
                                    allow_custom_value=True,
                                    label="dtype",
                                )
                                out_dtype = gr.Dropdown(
                                    choices=DTYPE_CHOICES,
                                    value="",
                                    allow_custom_value=True,
                                    label="out_dtype",
                                )
                                tokenizer_source = gr.Dropdown(
                                    choices=["base", "union", "none"],
                                    value="base",
                                    allow_custom_value=True,
                                    label="Tokenizer source",
                                    info="Use `base`, `union`, `none`, or a model path.",
                                )
                                chat_template = gr.Dropdown(
                                    choices=CHAT_TEMPLATE_CHOICES,
                                    value="auto",
                                    allow_custom_value=True,
                                    label="Chat template",
                                )
                            with gr.Column(scale=7):
                                global_parameters = gr.Code(
                                    label="Global parameters YAML",
                                    language="yaml",
                                    value=block_parameter_template(
                                        _get_method(default_method).parameters()
                                    ),
                                )
                                reset_templates_button = gr.Button(
                                    "Reset parameter templates"
                                )
                                gr.Markdown(
                                    "Per-model parameters live in the table below. "
                                    "Use YAML or JSON-style mappings there, for "
                                    "example `{weight: 0.5}`."
                                )

                        with gr.Row():
                            model_search = build_model_input("Add model")
                            add_model_button = gr.Button(
                                "Add model", variant="secondary"
                            )
                            add_blank_row_button = gr.Button(
                                "Add blank row", variant="secondary"
                            )

                        models_table = gr.Dataframe(
                            headers=MODEL_TABLE_HEADERS,
                            datatype=MODEL_TABLE_DATATYPES,
                            row_count=(1, "dynamic"),
                            col_count=(len(MODEL_TABLE_HEADERS), "fixed"),
                            value=[blank_model_row(default_method)],
                            label="Input models",
                            interactive=True,
                            wrap=True,
                        )

                        with gr.Row():
                            preview_builder_button = gr.Button(
                                "Preview YAML", variant="secondary"
                            )
                            run_builder_button = gr.Button(
                                "Run merge", variant="primary"
                            )

                        builder_preview = gr.Code(
                            label="Generated YAML",
                            language="yaml",
                        )

                    with gr.Tab("Raw YAML"):
                        yaml_editor = gr.Code(
                            label="Merge configuration",
                            language="yaml",
                            value=DEFAULT_CONFIG_TEMPLATE,
                        )
                        with gr.Row():
                            validate_yaml_button = gr.Button(
                                "Validate YAML", variant="secondary"
                            )
                            run_yaml_button = gr.Button("Run merge", variant="primary")

                status = gr.Markdown(status_message("Ready."))
                build_log_component(log_file.name)

        merge_method.change(
            update_method_summary,
            inputs=[merge_method, global_parameters],
            outputs=[method_summary, global_parameters],
        )
        reset_templates_button.click(
            reset_parameter_templates,
            inputs=[merge_method, models_table],
            outputs=[global_parameters, models_table],
        )
        add_model_button.click(
            add_model_row,
            inputs=[model_search, models_table, merge_method],
            outputs=[model_search, models_table],
        )
        add_blank_row_button.click(
            add_blank_model_row,
            inputs=[models_table, merge_method],
            outputs=[models_table],
        )

        builder_inputs = [
            merge_method,
            base_model,
            dtype,
            out_dtype,
            tokenizer_source,
            chat_template,
            global_parameters,
            models_table,
        ]

        preview_builder_button.click(
            preview_builder_yaml,
            inputs=builder_inputs,
            outputs=[builder_preview, yaml_editor, status],
        )
        run_builder_button.click(
            build_and_run_from_builder,
            inputs=builder_inputs
            + [
                log_file_state,
                output_directory,
                upload_repo_id,
                upload_private,
                upload_token,
                compute_device,
                storage_device,
                allow_crimes,
                trust_remote_code,
                lazy_unpickle,
                load_to_compute,
                write_model_card,
                quiet,
                executor,
                random_seed,
            ],
            outputs=[builder_preview, yaml_editor, status],
        )
        validate_yaml_button.click(
            validate_yaml_editor,
            inputs=[yaml_editor],
            outputs=[yaml_editor, status],
        )
        run_yaml_button.click(
            run_merge_from_yaml,
            inputs=[
                yaml_editor,
                log_file_state,
                output_directory,
                upload_repo_id,
                upload_private,
                upload_token,
                compute_device,
                storage_device,
                allow_crimes,
                trust_remote_code,
                lazy_unpickle,
                load_to_compute,
                write_model_card,
                quiet,
                executor,
                random_seed,
            ],
            outputs=[status],
        )

    return gui


def main(share: bool):
    build_gui().queue(default_concurrency_limit=1).launch(share=share)


@click.command("mergekitty-gui")
@click.option(
    "--share", is_flag=True, required=False, default=False, help="Share the GUI"
)
def cli_main(share: bool):
    main(share)


if __name__ == "__main__":
    cli_main()
