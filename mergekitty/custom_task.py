"""Custom task implementations for mergekitty."""

import torch
import peft
from typing import Dict
from mergekitty.task import Task


class CustomLoraScaleTask(Task[float]):
    """Task that calculates the LoRA scaling factor based on adapter config."""

    adapter_config: peft.PeftConfig

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **kwargs) -> float:
        return self.adapter_config.lora_alpha / self.adapter_config.r

    def uses_accelerator(self) -> bool:
        return False


class CustomMergeLoraTask(Task[torch.Tensor]):
    """Task that merges a LoRA adapter with a base weight."""

    base_tensor: Task[torch.Tensor]
    adapter_tensor_a: Task[torch.Tensor]
    adapter_tensor_b: Task[torch.Tensor]
    adapter_config: peft.PeftConfig

    def arguments(self) -> Dict[str, Task]:
        return {
            "base_tensor": self.base_tensor,
            "adapter_tensor_a": self.adapter_tensor_a,
            "adapter_tensor_b": self.adapter_tensor_b,
            "lora_scale": CustomLoraScaleTask(adapter_config=self.adapter_config),
        }

    def execute(self, **kwargs) -> torch.Tensor:
        base_tensor = kwargs["base_tensor"]
        adapter_tensor_a = kwargs["adapter_tensor_a"]
        adapter_tensor_b = kwargs["adapter_tensor_b"]
        lora_scale = kwargs["lora_scale"]
        
        old_dtype = base_tensor.dtype
        tensor = base_tensor.to(torch.float32)
        tensor += lora_scale * (adapter_tensor_b @ adapter_tensor_a)
        return tensor.to(old_dtype)

    def uses_accelerator(self) -> bool:
        return True