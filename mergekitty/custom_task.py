"""Custom task implementations for mergekitty."""

import math
from typing import Dict

import torch

from mergekitty.task import Task


class MergeLoraTask(Task[torch.Tensor]):
    """Merge LoRA tensors into a base weight."""

    base_tensor: Task
    adapter_tensor_a: Task
    adapter_tensor_b: Task
    lora_alpha: float
    use_rslora: bool = False
    fan_in_fan_out: bool = False
    is_embedding: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {
            "base_tensor": self.base_tensor,
            "adapter_tensor_a": self.adapter_tensor_a,
            "adapter_tensor_b": self.adapter_tensor_b,
        }

    def execute(self, **kwargs) -> torch.Tensor:
        base_tensor = kwargs["base_tensor"]
        adapter_tensor_a = kwargs["adapter_tensor_a"]
        adapter_tensor_b = kwargs["adapter_tensor_b"]

        rank = adapter_tensor_a.shape[0]
        if rank <= 0:
            raise RuntimeError("LoRA rank must be positive")

        scale = self.lora_alpha / (math.sqrt(rank) if self.use_rslora else rank)

        old_dtype = base_tensor.dtype
        base_fp32 = base_tensor.to(torch.float32)
        tensor_a_fp32 = adapter_tensor_a.to(torch.float32)
        tensor_b_fp32 = adapter_tensor_b.to(torch.float32)

        delta = tensor_b_fp32 @ tensor_a_fp32
        if self.is_embedding or self.fan_in_fan_out:
            delta = delta.transpose(0, 1)

        if delta.shape != base_fp32.shape:
            raise RuntimeError(
                f"LoRA delta shape {delta.shape} does not match base tensor shape "
                f"{base_fp32.shape}"
            )

        return (base_fp32 + (delta * scale)).to(old_dtype)

    def uses_accelerator(self) -> bool:
        return True
