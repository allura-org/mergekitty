# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org
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

import importlib.resources
import string
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator
from transformers import PretrainedConfig
from typing_extensions import Literal

import mergekitty._data.architectures


class WeightInfo(BaseModel, frozen=True):
    """Information about an individual weight tensor in a model.

    Attributes:
        name (str):
            The name of the tensor representing the weight.
        is_embed (bool):
            Indicates whether the weight is for an embedding or language model head.
        input_space (Optional[str]):
            The name of the input space associated with the weight, if applicable.
        output_space (Optional[str]):
            The name of the output space associated with the weight, if applicable.
        optional (bool):
            Indicates whether the weight can be omitted from a model.
        aliases (Optional[List[str]]):
            List of alternative names for the weight, if applicable.
        tied_names (Optional[List[str]]):
            List of names for weights that are tied to this weight, if applicable.
        force_dtype (Optional[str]):
            Mandatory dtype for the weight, if applicable.
    """

    name: str
    is_embed: bool = False
    input_space: Optional[str] = None
    output_space: Optional[str] = None
    optional: bool = False
    tied: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    tied_names: Optional[Tuple[str, ...]] = None
    force_dtype: Optional[str] = None
    head_split: Literal[None, "input", "output"] = None
    is_kq: Optional[bool] = False
    is_sparse: Optional[bool] = False


class ProceduralSpaceInfo(BaseModel, frozen=True):
    """Defines a procedural space computed from one or more other spaces.

    Currently only supports residual connections.

    Attributes:
        name (str): The name of the space defined.
        type (str): The type of procedural space.
        inputs (List[str]): List of names of spaces used to define this space."""

    name: str
    type: Literal["residual"]
    inputs: List[str]


class ArchitectureInfo(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the name of the architecture."""
        ...

    @abstractmethod
    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights preceding the first layer."""
        ...

    @abstractmethod
    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights following the final layer."""
        ...

    @abstractmethod
    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        """Return a list of all weights associated with a given layer."""
        ...

    @abstractmethod
    def sliceable(self) -> bool:
        """
        Return True if the layers of this architecture can be meaningfully sliced.
        """
        ...

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"

    def num_experts_config_key(self) -> str:
        """Key in config that represents number of experts"""
        return "num_experts"

    def num_layers(self, config: PretrainedConfig) -> int:
        """Return the number of layers in a model."""
        return get_config_value(config, self.num_layers_config_key())

    def num_experts(self, config: PretrainedConfig) -> int:
        """Return the number of experts in a model, or None if not applicable."""
        try:
            return get_config_value(config, self.num_experts_config_key())
        except (AttributeError, KeyError):
            return None

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return all weights associated with a model."""
        num_layers = self.num_layers(config)
        res = list(self.pre_weights(config))
        for layer_idx in range(num_layers):
            res.extend(self.layer_weights(layer_idx, config))
        res.extend(self.post_weights(config))
        return res

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        """Return a list of all procedurally defined spaces in a model."""
        return []

    def has_defined_spaces(self) -> bool:
        """
        Return True if this architecture defines space information needed for
        matching-based merge methods.
        """
        return False


class ConfiguredArchitectureInfo(BaseModel, frozen=True, arbitrary_types_allowed=True):
    info: ArchitectureInfo
    config: PretrainedConfig

    def name(self) -> str:
        return self.info.name()

    def num_layers(self) -> int:
        return self.info.num_layers(self.config)

    def pre_weights(self) -> List[WeightInfo]:
        return self.info.pre_weights(self.config)

    def post_weights(self) -> List[WeightInfo]:
        return self.info.post_weights(self.config)

    def layer_weights(self, index: int) -> List[WeightInfo]:
        return self.info.layer_weights(index, self.config)

    def procedural_spaces(self) -> List[ProceduralSpaceInfo]:
        return self.info.procedural_spaces(self.config)

    def all_weights(self) -> List[WeightInfo]:
        return self.info.all_weights(self.config)


class JSONLayerTemplates(BaseModel, frozen=True):
    weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None


class JSONWeightTemplateGroup(BaseModel, frozen=True):
    weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None
    count_config_key: Optional[str] = None
    index_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_repeat_fields(self):
        if (self.count_config_key is None) != (self.index_name is None):
            raise ValueError(
                "count_config_key and index_name must be provided together"
            )
        return self


class JSONArchitectureDefinition(BaseModel, frozen=True):
    expected_model_type: str = Field(alias="model_type")
    architectures: List[str]
    match_model_type: Optional[str] = None
    match_model_type_config_key: Optional[str] = None
    pre_weights: List[WeightInfo]
    pre_weight_templates: Optional[List[JSONWeightTemplateGroup]] = None
    layer_templates: JSONLayerTemplates
    post_weights: List[WeightInfo]
    post_weight_templates: Optional[List[JSONWeightTemplateGroup]] = None
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None
    num_layers_config_key: Optional[str] = None
    num_experts_config_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_match_model_type_fields(self):
        if (self.match_model_type is None) != (
            self.match_model_type_config_key is None
        ):
            raise ValueError(
                "match_model_type and match_model_type_config_key must be provided together"
            )
        return self


class TemplateWithArithmetic(string.Template):
    idpattern = r"(?a:[_a-z][_a-z0-9]*([+-]1)?)"


def _with_index_substitutions(
    substitutions: Dict[str, int], key: str, value: int
) -> None:
    substitutions[key] = value
    substitutions[f"{key}+1"] = value + 1
    substitutions[f"{key}-1"] = value - 1


def _template_substitution(
    template: str,
    num_layers: int,
    layer_idx: Optional[int] = None,
    num_experts: Optional[int] = None,
    expert_idx: Optional[int] = None,
    extra_substitutions: Optional[Dict[str, int]] = None,
) -> str:
    if "{" not in template:
        return template

    substitutions: Dict[str, int] = {}
    _with_index_substitutions(substitutions, "num_layers", num_layers)

    if layer_idx is not None:
        _with_index_substitutions(substitutions, "layer_index", layer_idx)

    if num_experts is not None:
        _with_index_substitutions(substitutions, "num_experts", num_experts)
        _with_index_substitutions(substitutions, "expert_index", num_experts)
        if expert_idx is not None:
            _with_index_substitutions(substitutions, "expert_index", expert_idx)

    for key, value in (extra_substitutions or {}).items():
        _with_index_substitutions(substitutions, key, value)

    return TemplateWithArithmetic(template).substitute(substitutions)


def get_config_value(config: Union[PretrainedConfig, Dict], key_path: str):
    value = config
    for key in key_path.split("."):
        if isinstance(value, dict):
            value = value[key]
        else:
            value = getattr(value, key)
    return value


def set_config_value(
    config: Union[PretrainedConfig, Dict], key_path: str, value
) -> None:
    keys = key_path.split(".")
    target = config
    for key in keys[:-1]:
        if isinstance(target, dict):
            target = target[key]
        else:
            target = getattr(target, key)

    leaf = keys[-1]
    if isinstance(target, dict):
        target[leaf] = value
    else:
        setattr(target, leaf, value)


class JsonArchitectureInfo(ArchitectureInfo, BaseModel, frozen=True):
    definition: JSONArchitectureDefinition

    def matches_config(self, config: PretrainedConfig) -> bool:
        if self.definition.expected_model_type != config.model_type:
            return False

        if self.definition.match_model_type_config_key is None:
            return True

        try:
            actual_match_model_type = get_config_value(
                config, self.definition.match_model_type_config_key
            )
        except (AttributeError, KeyError):
            return False

        return actual_match_model_type == self.definition.match_model_type

    def _substitute(
        self,
        item: Union[WeightInfo, ProceduralSpaceInfo],
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
        expert_idx: Optional[int] = None,
        extra_substitutions: Optional[Dict[str, int]] = None,
    ) -> Union[WeightInfo, ProceduralSpaceInfo]:
        num_layers = self.num_layers(config)
        num_experts = self.num_experts(config)

        obj_dict = item.model_dump(mode="json", exclude_unset=True)
        for key in obj_dict:
            if isinstance(obj_dict[key], str):
                obj_dict[key] = _template_substitution(
                    obj_dict[key],
                    num_layers,
                    layer_idx,
                    num_experts,
                    expert_idx,
                    extra_substitutions=extra_substitutions,
                )
            elif isinstance(obj_dict[key], list):
                obj_dict[key] = [
                    (
                        _template_substitution(
                            s,
                            num_layers,
                            layer_idx,
                            num_experts,
                            expert_idx,
                            extra_substitutions=extra_substitutions,
                        )
                        if isinstance(s, str)
                        else s
                    )
                    for s in obj_dict[key]
                ]
        return type(item).model_validate(obj_dict)

    def _substitute_group(
        self,
        group: JSONWeightTemplateGroup,
        config: PretrainedConfig,
        item: Union[WeightInfo, ProceduralSpaceInfo],
    ) -> List[Union[WeightInfo, ProceduralSpaceInfo]]:
        if not group.count_config_key:
            return [self._substitute(item, config=config)]

        repeat_count = get_config_value(config, group.count_config_key)
        return [
            self._substitute(
                item,
                config=config,
                extra_substitutions={group.index_name: index},
            )
            for index in range(repeat_count)
        ]

    def name(self) -> str:
        return self.definition.expected_model_type

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        res = [
            self._substitute(wi, config=config) for wi in self.definition.pre_weights
        ]
        for group in self.definition.pre_weight_templates or []:
            for wi in group.weights:
                res.extend(self._substitute_group(group, config, wi))
        return res

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        if self.definition.num_experts_config_key is None:
            return [
                self._substitute(wi, config=config, layer_idx=index)
                for wi in self.definition.layer_templates.weights
            ]
        else:
            expert_weights = [
                wi for wi in self.definition.layer_templates.weights if wi.is_sparse
            ]
            regular_weights = [
                wi for wi in self.definition.layer_templates.weights if not wi.is_sparse
            ]
            return [
                self._substitute(wi, config=config, layer_idx=index)
                for wi in regular_weights
            ] + [
                self._substitute(
                    wi, config=config, layer_idx=index, expert_idx=expert_idx
                )
                for wi in expert_weights
                for expert_idx in range(self.num_experts(config))
            ]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        res = [
            self._substitute(wi, config=config) for wi in self.definition.post_weights
        ]
        for group in self.definition.post_weight_templates or []:
            for wi in group.weights:
                res.extend(self._substitute_group(group, config, wi))
        return res

    def sliceable(self) -> bool:
        return True

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        res = []
        for s in self.definition.procedural_spaces or []:
            res.append(self._substitute(s, config=config))
        for group in self.definition.pre_weight_templates or []:
            for s in group.procedural_spaces or []:
                res.extend(self._substitute_group(group, config, s))
        for idx in range(self.num_layers(config)):
            for s in self.definition.layer_templates.procedural_spaces or []:
                res.append(self._substitute(s, config=config, layer_idx=idx))
        for group in self.definition.post_weight_templates or []:
            for s in group.procedural_spaces or []:
                res.extend(self._substitute_group(group, config, s))
        return res

    def has_defined_spaces(self) -> bool:
        if (
            self.definition.procedural_spaces
            or any(
                group.procedural_spaces
                for group in (self.definition.pre_weight_templates or [])
            )
            or self.definition.layer_templates.procedural_spaces
            or any(
                group.procedural_spaces
                for group in (self.definition.post_weight_templates or [])
            )
        ):
            return True
        for wi in (
            self.definition.pre_weights
            + [
                wi
                for group in (self.definition.pre_weight_templates or [])
                for wi in group.weights
            ]
            + self.definition.layer_templates.weights
            + self.definition.post_weights
            + [
                wi
                for group in (self.definition.post_weight_templates or [])
                for wi in group.weights
            ]
        ):
            if wi.input_space or wi.output_space:
                return True
        return False

    def num_layers_config_key(self) -> str:
        return self.definition.num_layers_config_key or super().num_layers_config_key()

    def num_experts_config_key(self) -> str:
        return (
            self.definition.num_experts_config_key or super().num_experts_config_key()
        )


def _load_json_arch(name: str) -> JsonArchitectureInfo:
    text = importlib.resources.read_text(mergekitty._data.architectures, name)
    return JsonArchitectureInfo(
        definition=JSONArchitectureDefinition.model_validate_json(text)
    )


def _load_all_architectures() -> Tuple[
    List[JsonArchitectureInfo], Dict[str, List[JsonArchitectureInfo]]
]:
    architectures: List[JsonArchitectureInfo] = []
    for f in importlib.resources.contents(mergekitty._data.architectures):
        if f.lower().endswith(".json"):
            architectures.append(_load_json_arch(f))

    name_to_arch: Dict[str, List[JsonArchitectureInfo]] = {}
    for arch_info in architectures:
        for name in arch_info.definition.architectures:
            name_to_arch[name] = name_to_arch.get(name, [])
            name_to_arch[name].append(arch_info)
    return architectures, name_to_arch


JSON_ARCHITECTURES, NAME_TO_ARCH = _load_all_architectures()
MISTRAL_INFO = _load_json_arch("mistral.json")
QWEN2_INFO = _load_json_arch("qwen2.json")


def get_architecture_info(config: PretrainedConfig) -> ArchitectureInfo:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")

    arch_name = config.architectures[0]

    if arch_name not in NAME_TO_ARCH:
        raise RuntimeError(f"Unsupported architecture {arch_name}")

    candidates = list(NAME_TO_ARCH[arch_name])
    matches = [candidate for candidate in candidates if candidate.matches_config(config)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous model_type {config.model_type} for architecture {arch_name}"
        )

    raise RuntimeError(
        f"Unsupported model_type {config.model_type} for architecture {arch_name}"
    )
