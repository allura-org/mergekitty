from typing import List

from mergekitty.moe.arch import MoEOutputArchitecture
from mergekitty.moe.deepseek import DeepseekMoE
from mergekitty.moe.mixtral import MixtralMoE

ALL_OUTPUT_ARCHITECTURES: List[MoEOutputArchitecture] = [MixtralMoE(), DeepseekMoE()]

try:
    from mergekitty.moe.qwen import QwenMoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(QwenMoE())

__all__ = [
    "ALL_OUTPUT_ARCHITECTURES",
    "MoEOutputArchitecture",
]
