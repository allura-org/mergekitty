# Copyright (C) 2025-2026 Allura-org
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import click
import transformers

from mergekitty.architecture import WeightInfo, get_architecture_info
from mergekitty.card import generate_card_merged_lora
from mergekitty.common import AdapterReference, ModelPath, ModelReference
from mergekitty.custom_task import MergeLoraTask
from mergekitty.executor import SingleThreadedExecutor
from mergekitty.io.tasks import (
    FinalizeModel,
    LoadTensor,
    LoaderCache,
    SaveTensor,
    TensorWriterTask,
)
from mergekitty.options import MergeOptions, add_merge_options

_ADAPTER_PREFIX = "base_model.model."
_FULL_WEIGHT_SUFFIXES = (
    ".modules_to_save.weight",
    ".original_module.weight",
    ".base_layer.weight",
    ".weight",
)
_FULL_WEIGHT_PRIORITY = {
    ".weight": 0,
    ".modules_to_save.weight": 1,
    ".original_module.weight": 2,
    ".base_layer.weight": 3,
}


@dataclass
class LoraModuleSpec:
    module_name: str
    tensor_a: Optional[str] = None
    tensor_b: Optional[str] = None
    base_tensor: Optional[str] = None
    is_embedding: bool = False

    def is_complete(self) -> bool:
        return self.tensor_a is not None and self.tensor_b is not None


@dataclass
class AdapterLayout:
    lora_modules: dict[str, LoraModuleSpec] = field(default_factory=dict)
    full_modules: dict[str, str] = field(default_factory=dict)


@dataclass
class PlanResults:
    tasks: list
    base_vocab_size: int
    merged_vocab_size: int


def _module_name_from_tensor(tensor_name: str, suffix: str) -> str:
    stripped = tensor_name[len(_ADAPTER_PREFIX) :]
    return stripped[: -len(suffix)]


def _parse_adapter_layout(adapter_ref: AdapterReference) -> AdapterLayout:
    loader = LoaderCache().get(adapter_ref)
    layout = AdapterLayout()
    full_weight_candidates: dict[str, tuple[int, str]] = {}

    for tensor_name in loader.index.tensor_paths:
        if not tensor_name.startswith(_ADAPTER_PREFIX):
            continue

        if tensor_name.endswith(".lora_A.weight"):
            module_name = _module_name_from_tensor(tensor_name, ".lora_A.weight")
            layout.lora_modules.setdefault(
                module_name, LoraModuleSpec(module_name=module_name)
            ).tensor_a = tensor_name
            continue

        if tensor_name.endswith(".lora_B.weight"):
            module_name = _module_name_from_tensor(tensor_name, ".lora_B.weight")
            layout.lora_modules.setdefault(
                module_name, LoraModuleSpec(module_name=module_name)
            ).tensor_b = tensor_name
            continue

        if tensor_name.endswith(".lora_embedding_A"):
            module_name = _module_name_from_tensor(tensor_name, ".lora_embedding_A")
            spec = layout.lora_modules.setdefault(
                module_name, LoraModuleSpec(module_name=module_name)
            )
            spec.tensor_a = tensor_name
            spec.is_embedding = True
            continue

        if tensor_name.endswith(".lora_embedding_B"):
            module_name = _module_name_from_tensor(tensor_name, ".lora_embedding_B")
            spec = layout.lora_modules.setdefault(
                module_name, LoraModuleSpec(module_name=module_name)
            )
            spec.tensor_b = tensor_name
            spec.is_embedding = True
            continue

        for suffix in _FULL_WEIGHT_SUFFIXES:
            if not tensor_name.endswith(suffix):
                continue

            module_name = _module_name_from_tensor(tensor_name, suffix)
            if suffix == ".base_layer.weight":
                layout.lora_modules.setdefault(
                    module_name, LoraModuleSpec(module_name=module_name)
                ).base_tensor = tensor_name

            current = full_weight_candidates.get(module_name)
            priority = _FULL_WEIGHT_PRIORITY[suffix]
            if current is None or priority < current[0]:
                full_weight_candidates[module_name] = (priority, tensor_name)
            break

    incomplete = [
        module_name
        for module_name, spec in layout.lora_modules.items()
        if (spec.tensor_a is None) != (spec.tensor_b is None)
    ]
    if incomplete:
        raise RuntimeError(
            "Adapter contains incomplete LoRA tensor pairs for modules: "
            + ", ".join(sorted(incomplete))
        )

    for module_name, (_priority, tensor_name) in full_weight_candidates.items():
        spec = layout.lora_modules.get(module_name)
        if spec and spec.is_complete() and spec.base_tensor == tensor_name:
            continue
        layout.full_modules[module_name] = tensor_name

    return layout


def _resolve_weight_tensor_name(loader, weight: WeightInfo) -> Optional[str]:
    names = [weight.name] + list(weight.aliases or ()) + list(weight.tied_names or ())
    for name in names:
        if name in loader.index.tensor_paths:
            return name
    return None


def _weight_module_candidates(weight: WeightInfo) -> list[str]:
    names = [weight.name] + list(weight.aliases or ()) + list(weight.tied_names or ())
    res = []
    for name in names:
        if name.endswith(".weight"):
            res.append(name[: -len(".weight")])
    return res


def _matching_module_name(weight: WeightInfo, modules: set[str]) -> Optional[str]:
    for module_name in _weight_module_candidates(weight):
        if module_name in modules:
            return module_name
    return None


def _module_alpha(adapter_config, module_name: str) -> float:
    alpha_pattern = getattr(adapter_config, "alpha_pattern", {}) or {}
    return float(alpha_pattern.get(module_name, adapter_config.lora_alpha))


def _validate_supported_adapter(adapter_config) -> None:
    if getattr(adapter_config, "peft_type", None) != "LORA":
        raise RuntimeError(
            "mergekitty-merge-lora currently supports only LoRA adapters"
        )
    if getattr(adapter_config, "bias", "none") != "none":
        raise RuntimeError("LoRA adapters with bias terms are not yet supported")
    if getattr(adapter_config, "use_dora", False):
        raise RuntimeError("DoRA adapters are not yet supported")


def _load_tokenizer(
    base_model_ref: ModelReference,
    adapter_ref: AdapterReference,
    trust_remote_code: bool,
):
    for source_name, path, revision in [
        ("adapter", adapter_ref.adapter.path, adapter_ref.adapter.revision),
        ("base", base_model_ref.model.path, base_model_ref.model.revision),
    ]:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            return tokenizer, source_name
        except Exception as exc:
            logging.info(
                "Unable to load tokenizer from %s source %s",
                source_name,
                path,
                exc_info=exc,
            )

    return None, None


def _reconstruct_invocation(
    base_model: str,
    adapter_model: str,
    options: MergeOptions,
) -> str:
    defaults = MergeOptions()
    parts = [
        "mergekitty-merge-lora",
        base_model,
        adapter_model,
        "OUTPUT_PATH",
    ]

    for field_name, info in MergeOptions.model_fields.items():
        value = getattr(options, field_name)
        default = getattr(defaults, field_name)
        if value == default:
            continue

        option_name = field_name.replace("_", "-")
        if info.annotation is bool:
            parts.append(f"--{option_name}" if value else f"--no-{option_name}")
        else:
            parts.extend([f"--{option_name}", str(value)])

    return " ".join(parts)


def plan_merge(
    base_model_ref: ModelReference,
    adapter_ref: AdapterReference,
    output_path: str,
    options: MergeOptions,
) -> PlanResults:
    base_config = base_model_ref.config(trust_remote_code=options.trust_remote_code)
    arch_info = get_architecture_info(base_config)
    adapter_config = adapter_ref.config(trust_remote_code=options.trust_remote_code)
    _validate_supported_adapter(adapter_config)
    layout = _parse_adapter_layout(adapter_ref)

    base_loader = LoaderCache().get(base_model_ref)
    adapter_loader = LoaderCache().get(adapter_ref)

    writer_task = TensorWriterTask(
        out_path=output_path,
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
    )

    save_tasks = []
    used_modules = set()

    base_vocab_size = getattr(base_config, "vocab_size", 0)
    merged_vocab_size = base_vocab_size

    for weight in arch_info.all_weights(base_config):
        device = "cuda" if options.read_to_gpu else None
        lora_module_name = _matching_module_name(
            weight,
            {
                module_name
                for module_name, spec in layout.lora_modules.items()
                if spec.is_complete()
            },
        )
        full_module_name = _matching_module_name(weight, set(layout.full_modules))

        if lora_module_name:
            spec = layout.lora_modules[lora_module_name]
            base_tensor_task: LoadTensor
            if spec.base_tensor:
                base_tensor_task = LoadTensor(
                    model=adapter_ref,
                    tensor=spec.base_tensor,
                    device=device,
                    optional=weight.optional,
                )
            else:
                base_tensor_task = LoadTensor(
                    model=base_model_ref,
                    tensor=weight.name,
                    device=device,
                    optional=weight.optional,
                    aliases=weight.aliases,
                    tied_names=weight.tied_names,
                )

            tensor_task = MergeLoraTask(
                base_tensor=base_tensor_task,
                adapter_tensor_a=LoadTensor(
                    model=adapter_ref,
                    tensor=spec.tensor_a,
                    device=device,
                ),
                adapter_tensor_b=LoadTensor(
                    model=adapter_ref,
                    tensor=spec.tensor_b,
                    device=device,
                ),
                lora_alpha=_module_alpha(adapter_config, lora_module_name),
                use_rslora=getattr(adapter_config, "use_rslora", False),
                fan_in_fan_out=getattr(adapter_config, "fan_in_fan_out", False),
                is_embedding=spec.is_embedding,
            )
            used_modules.add(lora_module_name)
        elif full_module_name:
            tensor_task = LoadTensor(
                model=adapter_ref,
                tensor=layout.full_modules[full_module_name],
                device=device,
                optional=weight.optional,
            )
            used_modules.add(full_module_name)
        else:
            tensor_task = LoadTensor(
                model=base_model_ref,
                tensor=weight.name,
                device=device,
                optional=weight.optional,
                aliases=weight.aliases,
                tied_names=weight.tied_names,
            )

        save_tasks.append(
            SaveTensor(
                tensor_name=weight.name,
                tensor_task=tensor_task,
                writer_task=writer_task,
                clone=options.clone_tensors,
                optional=weight.optional,
                dtype=weight.force_dtype,
            )
        )

        if weight.is_embed:
            base_tensor_name = _resolve_weight_tensor_name(base_loader, weight)
            if base_tensor_name:
                base_vocab_size = max(
                    base_vocab_size,
                    base_loader.get_tensor(base_tensor_name).shape[0],
                )

            if lora_module_name and layout.lora_modules[lora_module_name].base_tensor:
                merged_vocab_size = max(
                    merged_vocab_size,
                    adapter_loader.get_tensor(
                        layout.lora_modules[lora_module_name].base_tensor
                    ).shape[0],
                )
            elif full_module_name:
                merged_vocab_size = max(
                    merged_vocab_size,
                    adapter_loader.get_tensor(
                        layout.full_modules[full_module_name]
                    ).shape[0],
                )
            elif base_tensor_name:
                merged_vocab_size = max(
                    merged_vocab_size,
                    base_loader.get_tensor(base_tensor_name).shape[0],
                )

    expected_modules = {
        module_name
        for module_name, spec in layout.lora_modules.items()
        if spec.is_complete()
    } | set(layout.full_modules)
    unused_modules = expected_modules - used_modules
    if unused_modules:
        raise RuntimeError(
            "Adapter contains modules that do not match output weights: "
            + ", ".join(sorted(unused_modules))
        )

    finalize_task = FinalizeModel(
        tensor_save_tasks=tuple(save_tasks),
        writer_task=writer_task,
    )

    return PlanResults(
        tasks=save_tasks + [finalize_task],
        base_vocab_size=base_vocab_size,
        merged_vocab_size=merged_vocab_size,
    )


def run_merge_lora(
    base_model: str,
    adapter_model: str,
    output_path: str,
    merge_options: MergeOptions,
) -> None:
    logging.basicConfig(
        level=logging.WARNING if merge_options.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    base_model_ref = ModelReference.model_validate(base_model)
    if base_model_ref.lora is not None:
        raise RuntimeError("BASE_MODEL must not include an inline LoRA reference")

    adapter_ref = AdapterReference(adapter=ModelPath.model_validate(adapter_model))

    loader_cache = LoaderCache()
    loader_cache.setup(merge_options)

    adapter_config = adapter_ref.config(
        trust_remote_code=merge_options.trust_remote_code
    )
    _validate_supported_adapter(adapter_config)

    if (
        getattr(adapter_config, "base_model_name_or_path", None)
        and adapter_config.base_model_name_or_path != base_model_ref.model.path
    ):
        logging.warning(
            "Adapter was created for %s but merge is using %s",
            adapter_config.base_model_name_or_path,
            base_model_ref.model.path,
        )

    os.makedirs(output_path, exist_ok=True)

    plan = plan_merge(
        base_model_ref=base_model_ref,
        adapter_ref=adapter_ref,
        output_path=output_path,
        options=merge_options,
    )

    exec = SingleThreadedExecutor(
        tasks=plan.tasks,
        math_device="cuda" if merge_options.cuda else "cpu",
        storage_device="cuda" if merge_options.low_cpu_memory else "cpu",
    )
    exec.execute()

    cfg_out = base_model_ref.config(trust_remote_code=merge_options.trust_remote_code)
    cfg_out.vocab_size = plan.merged_vocab_size
    cfg_out.save_pretrained(output_path)

    tokenizer = None
    if merge_options.copy_tokenizer:
        tokenizer, tokenizer_source = _load_tokenizer(
            base_model_ref=base_model_ref,
            adapter_ref=adapter_ref,
            trust_remote_code=merge_options.trust_remote_code,
        )
        if tokenizer is None:
            logging.warning(
                "No tokenizer could be loaded from the adapter or base model"
            )
        else:
            logging.info("Saving %s tokenizer", tokenizer_source)
            tokenizer.save_pretrained(output_path, safe_serialization=True)
            if len(tokenizer) != plan.merged_vocab_size:
                logging.warning(
                    "Tokenizer size %s does not match merged vocab size %s",
                    len(tokenizer),
                    plan.merged_vocab_size,
                )

    if merge_options.write_model_card:
        invocation = _reconstruct_invocation(
            base_model=base_model,
            adapter_model=adapter_model,
            options=merge_options,
        )
        card_md = generate_card_merged_lora(
            base_model_ref=base_model_ref,
            adapter_ref=adapter_ref,
            invocation=invocation,
            extended=plan.merged_vocab_size > plan.base_vocab_size,
            vocab_size=plan.merged_vocab_size,
            name=os.path.basename(output_path),
        )
        with open(os.path.join(output_path, "README.md"), "w", encoding="utf-8") as fp:
            fp.write(card_md)


@click.command("mergekitty-merge-lora")
@click.argument("base_model", type=str)
@click.argument("adapter_model", type=str)
@click.argument("output_path", type=str)
@add_merge_options
def main(
    base_model: str,
    adapter_model: str,
    output_path: str,
    merge_options: MergeOptions,
) -> None:
    """Merge a LoRA adapter with a base model to create a new model."""

    run_merge_lora(
        base_model=base_model,
        adapter_model=adapter_model,
        output_path=output_path,
        merge_options=merge_options,
    )


if __name__ == "__main__":
    main()
