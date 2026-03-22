# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org

import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Set, Union

import networkx
import torch

from mergekitty.task import Task


class ExecutorBase:
    DUMMY_TASK_VALUE = "!!DUMMY!!"

    math_device: torch.device
    storage_device: torch.device
    targets: List[Task]
    schedule: List[Task]
    dependencies: Dict[Task, Set[Task]]
    dependents: Dict[Task, Set[Task]]
    task_arguments: Dict[Task, Dict[str, Task]]
    _task_timing_mode: str
    _slow_task_threshold_seconds: float

    def __init__(
        self,
        tasks: List[Task],
        math_device: torch.device | str = torch.device("cpu"),
        storage_device: torch.device | str = torch.device("cpu"),
    ):
        self.math_device = torch.device(math_device)
        self.storage_device = torch.device(storage_device)
        self._task_timing_mode = (
            os.environ.get("MERGEKITTY_TASK_TIMINGS", "").strip().lower()
        )
        try:
            self._slow_task_threshold_seconds = float(
                os.environ.get("MERGEKITTY_SLOW_TASK_SECONDS", "5")
            )
        except ValueError:
            self._slow_task_threshold_seconds = 5.0
        self.schedule = self._make_schedule(tasks)
        self.targets = tasks
        self.dependents = self._build_dependents(self.dependencies)

    def _compare_key(self, task: Union[Task, str]):
        if task == ExecutorBase.DUMMY_TASK_VALUE:
            return ("", 0)
        return (
            task.group_label() or "",
            -task.priority(),
        )

    def _make_schedule(self, targets: List[Task]) -> List[Task]:
        self.schedule = []
        self.task_arguments = {}
        self.dependencies = self._build_dependencies(targets)

        edge_tups = []
        for node in self.dependencies:
            for dependency in self.dependencies[node]:
                edge_tups.append((dependency, node))

        for task in targets:
            edge_tups.append((ExecutorBase.DUMMY_TASK_VALUE, task))

        graph = networkx.DiGraph(edge_tups)
        return [
            task
            for task in networkx.lexicographical_topological_sort(
                graph, key=self._compare_key
            )
            if task != ExecutorBase.DUMMY_TASK_VALUE
        ]

    def _build_dependencies(self, targets: List[Task]) -> Dict[Task, Set[Task]]:
        task_dependencies: Dict[Task, Set[Task]] = {}
        to_process = list(targets)
        while to_process:
            child = to_process.pop()
            if child in task_dependencies:
                continue

            arguments = child.arguments()
            self.task_arguments[child] = arguments
            task_dependencies[child] = set()
            for dep in arguments.values():
                task_dependencies[child].add(dep)
                to_process.append(dep)
        return task_dependencies

    def _build_dependents(
        self, dependencies: Dict[Task, Set[Task]]
    ) -> Dict[Task, Set[Task]]:
        dependents = {task: set() for task in dependencies}
        for task, task_dependencies in dependencies.items():
            for dependency in task_dependencies:
                dependents.setdefault(dependency, set()).add(task)
        return dependents

    def _storage_device_for(self, execution_device: torch.device) -> torch.device:
        if self.storage_device.type != "cuda" or self.storage_device.index is not None:
            return self.storage_device
        if execution_device.type == "cuda":
            return execution_device
        return self.storage_device

    def _move_value_to_device(self, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device != device:
                return value.to(device)
            return value

        if isinstance(value, Mapping):
            return {
                key: self._move_value_to_device(child, device)
                for key, child in value.items()
            }

        if isinstance(value, list):
            return [self._move_value_to_device(child, device) for child in value]

        if isinstance(value, tuple):
            return tuple(self._move_value_to_device(child, device) for child in value)

        return value

    def _prepare_result(self, result: Any, execution_device: torch.device) -> Any:
        storage_device = self._storage_device_for(execution_device)
        if isinstance(result, torch.Tensor) and result.device != storage_device:
            return result.to(storage_device)
        return result

    def _task_label(self, task: Task) -> str:
        label = type(task).__name__
        for attr_name in ("tensor_name", "tensor", "name"):
            value = getattr(task, attr_name, None)
            if value:
                return f"{label}({value})"

        weight_info = getattr(task, "weight_info", None)
        if weight_info is not None and getattr(weight_info, "name", None):
            return f"{label}({weight_info.name})"

        return label

    def _log_task_phase(self, task: Task, phase: str, duration_seconds: float) -> None:
        if not self._task_timing_mode:
            return

        if (
            self._task_timing_mode != "all"
            and duration_seconds < self._slow_task_threshold_seconds
        ):
            return

        logging.info(
            "Task %s %s took %.3fs",
            self._task_label(task),
            phase,
            duration_seconds,
        )
