# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org

from typing import List

from mergekitty.task import Task

from .parallelism import ParallelismExecutor
from .single_threaded import SingleThreadedExecutor


def get_executor_class(name: str):
    normalized = str(name or "single").lower().replace("-", "_")
    if normalized in {"single", "single_threaded"}:
        return SingleThreadedExecutor
    if normalized in {"parallel", "parallelism"}:
        return ParallelismExecutor
    raise ValueError(f"Unknown executor '{name}'")


def build_executor(
    *,
    tasks: List[Task],
    math_device: str,
    storage_device: str,
    executor: str,
):
    executor_cls = get_executor_class(executor)
    return executor_cls(
        tasks=tasks,
        math_device=math_device,
        storage_device=storage_device,
    )


__all__ = [
    "ParallelismExecutor",
    "SingleThreadedExecutor",
    "build_executor",
    "get_executor_class",
]
