# Copyright (C) 2025 Allura-org
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

from typing import Dict, List

import click
import peft
from pydantic import BaseModel
import torch
from mergekitty.common import AdapterReference, ModelReference
from mergekitty.executor import SingleThreadedExecutor
from mergekitty.task import Task
from mergekitty.io.tasks import TensorWriterTask
from mergekitty.options import MergeOptions, add_merge_options


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
    model_ref = ModelReference.model_validate(base_model)
    adapter_ref = AdapterReference.model_validate(adapter_model)

    plan = plan_merge(model_ref, adapter_ref, output_path, merge_options)
    executor = SingleThreadedExecutor()
    for task, result in executor.run(plan.tasks):
        pass


class LoraScaleTask(Task[torch.Tensor]):
    adapter_config: peft.config.PeftConfig

    def run(self) -> float:
        return self.adapter_config["lora_alpha"] / self.adapter_config["r"]


class MergeLoraTask(Task[torch.Tensor]):
    base_tensor: torch.Tensor
    adapter_tensor_a: torch.Tensor
    adapter_tensor_b: torch.Tensor
    lora_scale: float

    def arguments(self) -> Dict[str, Task]:
        return {
            "base_tensor": self.base_tensor,
            "adapter_tensor_a": self.adapter_tensor_a,
            "adapter_tensor_b": self.adapter_tensor_b,
            "lora_scale": LoraScaleTask(self.adapter_config),
        }

    def run(self) -> torch.Tensor:
        old_dtype = self.base_tensor.dtype
        tensor = self.base_tensor.to(torch.float32)
        tensor += self.lora_scale * (self.adapter_tensor_b @ self.adapter_tensor_a)
        return tensor.to(old_dtype)


class PlanResults(BaseModel):
    tasks: List[Task]
    base_vocab_size: int
    final_vocab_size: int


def plan_merge(
    model_ref: ModelReference,
    adapter_ref: AdapterReference,
    output_path: str,
    options: MergeOptions,
) -> PlanResults:
    TensorWriterTask(
        out_path=output_path,
        max_shard_size=-1,
        safe_serialization=options.safe_serialization,
    )

    # TODO
