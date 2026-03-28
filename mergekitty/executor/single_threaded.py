# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org

from typing import Any, Dict, Iterator, List, Set, Tuple
import time

import torch

import tqdm

from mergekitty.executor._base import ExecutorBase
from mergekitty.task import Task


class SingleThreadedExecutor(ExecutorBase):
    """
    Schedules and executes a set of tasks and their dependencies.

    Handles scheduling, execution, the movement of data between devices, and the lifecycle of intermediate results.

    Attributes:
        math_device (torch.device): Device used for tensor computations.
        storage_device (torch.device): Device used for storing intermediate results.
        targets (List[Task]): List of target tasks to be executed.
        schedule (List[Task]): Calculated execution schedule of tasks.
        dependencies (Dict[Task, Set[Task]]): Dependencies of each task.
    """

    math_device: torch.device
    storage_device: torch.device
    targets: List[Task]
    schedule: List[Task]
    dependencies: Dict[Task, Set[Task]]

    def __init__(
        self,
        tasks: List[Task],
        math_device: torch.device | str = torch.device("cpu"),
        storage_device: torch.device | str = torch.device("cpu"),
    ):
        """
        Initializes the Executor with a list of tasks and device configurations.

        Args:
            tasks (List[Task]): The list of tasks to be executed.
            math_device (torch.device, optional): The device for tensor computations. Defaults to CPU.
            storage_device (torch.device, optional): The device for storing results. Defaults to CPU.
        """
        super().__init__(
            tasks=tasks,
            math_device=math_device,
            storage_device=storage_device,
        )

    def run(self, quiet: bool = False) -> Iterator[Tuple[Task, Any]]:
        """
        Execute the computed schedule and yield the target values.

        Yields:
            Iterator[Tuple[Task, Any]]: An iterator of task-result pairs.
        """
        # determine last usage of each value, so they can be evicted afterwards
        last_use_index = {}
        for idx, task in reversed(list(enumerate(self.schedule))):
            for t in self.dependencies[task]:
                if t not in last_use_index:
                    last_use_index[t] = idx
            if task not in last_use_index:
                last_use_index[task] = idx

        values: Dict[Task, Any] = {}
        for idx, task in (
            pbar := tqdm.tqdm(
                list(enumerate(self.schedule)),
                disable=quiet,
                desc="Executing graph",
            )
        ):
            use_math_device = task.uses_accelerator()

            prep_start = time.perf_counter()
            arguments = {}
            for name, dep in task.arguments().items():
                value = values[dep]

                # ensure any input tensors are on math device if task asks for it
                if use_math_device:
                    if (
                        isinstance(value, torch.Tensor)
                        and value.device != self.math_device
                    ):
                        value = value.to(self.math_device)
                    elif isinstance(value, dict):
                        for key in value:
                            if (
                                isinstance(value[key], torch.Tensor)
                                and value[key].device != self.math_device
                            ):
                                value[key] = value[key].to(self.math_device)

                arguments[name] = value
                del value
            self._log_task_phase(task, "prepare", time.perf_counter() - prep_start)

            execute_start = time.perf_counter()
            result = task.execute(**arguments)
            self._log_task_phase(task, "execute", time.perf_counter() - execute_start)

            store_start = time.perf_counter()
            res = self._prepare_result(
                task,
                result,
                execution_device=self.math_device
                if use_math_device
                else torch.device("cpu"),
            )
            self._log_task_phase(task, "store", time.perf_counter() - store_start)
            del result
            del arguments

            values[task] = res
            del res

            if task in self.targets:
                yield (task, values[task])

            # evict unreferenced values
            expired = []
            for key in values:
                if idx >= last_use_index[key]:
                    expired.append(key)

            for key in expired:
                del values[key]

        del values
        del pbar

    def execute(self) -> None:
        """
        Execute all tasks and discard results.
        """
        for task, value in self.run():
            pass
