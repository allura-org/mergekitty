# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025-2026 Allura-org

import heapq
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
import tqdm

from mergekitty.executor._base import ExecutorBase
from mergekitty.task import Task


class ParallelismExecutor(ExecutorBase):
    max_workers: int
    _cuda_devices: Tuple[torch.device, ...]

    def __init__(
        self,
        tasks: List[Task],
        math_device: torch.device | str = torch.device("cpu"),
        storage_device: torch.device | str = torch.device("cpu"),
        max_workers: Optional[int] = None,
    ):
        super().__init__(
            tasks=tasks,
            math_device=math_device,
            storage_device=storage_device,
        )

        if self.math_device.type == "cuda":
            if self.math_device.index is not None:
                self._cuda_devices = (self.math_device,)
            else:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self._cuda_devices = tuple(
                        torch.device("cuda", index) for index in range(device_count)
                    )
                else:
                    self._cuda_devices = (self.math_device,)
            default_parallelism = len(self._cuda_devices)
        else:
            self._cuda_devices = ()
            default_parallelism = os.cpu_count() or 1

        task_parallelism = len(self.schedule) or 1
        self.max_workers = min(max_workers or default_parallelism, task_parallelism)
        self.max_workers = max(1, self.max_workers)

    def run(self, quiet: bool = False) -> Iterator[Tuple[Task, Any]]:
        if not self.schedule:
            return

        ready_heap: List[Tuple[int, Task]] = []
        schedule_index = {task: idx for idx, task in enumerate(self.schedule)}
        remaining_dependencies = {
            task: len(self.dependencies[task]) for task in self.schedule
        }
        remaining_consumers = {
            task: len(self.dependents.get(task, set())) for task in self.dependencies
        }
        target_set = set(self.targets)
        target_schedule = [task for task in self.schedule if task in target_set]
        pending_target_results: Dict[Task, Any] = {}
        next_target_index = 0
        yielded_targets: Set[Task] = set()
        values: Dict[Task, Any] = {}
        running: Dict[Future[Any], Tuple[Task, torch.device]] = {}
        available_devices = list(self._cuda_devices)

        for task in self.schedule:
            if remaining_dependencies[task] == 0:
                heapq.heappush(ready_heap, (schedule_index[task], task))

        executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="mergekitty-executor",
        )
        pbar = tqdm.tqdm(
            total=len(self.schedule),
            disable=quiet,
            desc="Executing graph",
        )

        def maybe_evict(task: Task):
            if task not in values:
                return
            if remaining_consumers.get(task, 0) > 0:
                return
            if task in target_set and task not in yielded_targets:
                return
            del values[task]

        def reserve_device(task: Task) -> Optional[torch.device]:
            if not task.uses_accelerator():
                return torch.device("cpu")
            if self.math_device.type != "cuda":
                return self.math_device
            if len(self._cuda_devices) <= 1:
                return self._cuda_devices[0]
            if not available_devices:
                return None
            return available_devices.pop()

        def release_device(task: Task, execution_device: torch.device):
            if not task.uses_accelerator():
                return
            if self.math_device.type != "cuda":
                return
            if len(self._cuda_devices) <= 1:
                return
            available_devices.append(execution_device)

        def prepare_arguments(
            task: Task, execution_device: torch.device
        ) -> Dict[str, Any]:
            prepare_start = time.perf_counter()
            arguments = {}
            for name, dep in task.arguments().items():
                value = values[dep]
                if task.uses_accelerator():
                    value = self._move_value_to_device(value, execution_device)
                arguments[name] = value

            for dep in self.dependencies[task]:
                remaining_consumers[dep] -= 1
                maybe_evict(dep)

            self._log_task_phase(
                task,
                "prepare",
                time.perf_counter() - prepare_start,
            )
            return arguments

        def submit_ready_tasks():
            deferred: List[Tuple[int, Task]] = []
            while ready_heap and len(running) < self.max_workers:
                index, task = heapq.heappop(ready_heap)
                execution_device = reserve_device(task)
                if execution_device is None:
                    deferred.append((index, task))
                    continue

                arguments = prepare_arguments(task, execution_device)
                future = executor.submit(
                    self._execute_task,
                    task,
                    arguments,
                    execution_device,
                )
                running[future] = (task, execution_device)

            for entry in deferred:
                heapq.heappush(ready_heap, entry)

        try:
            submit_ready_tasks()

            while running:
                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task, execution_device = running.pop(future)
                    try:
                        result = future.result()
                    finally:
                        release_device(task, execution_device)

                    values[task] = result
                    pbar.update(1)

                    if task in target_set:
                        pending_target_results[task] = result
                        while (
                            next_target_index < len(target_schedule)
                            and target_schedule[next_target_index]
                            in pending_target_results
                        ):
                            next_target = target_schedule[next_target_index]
                            yielded_targets.add(next_target)
                            yield (
                                next_target,
                                pending_target_results.pop(next_target),
                            )
                            maybe_evict(next_target)
                            next_target_index += 1

                    for dependent in self.dependents.get(task, set()):
                        remaining_dependencies[dependent] -= 1
                        if remaining_dependencies[dependent] == 0:
                            heapq.heappush(
                                ready_heap, (schedule_index[dependent], dependent)
                            )

                    submit_ready_tasks()
        finally:
            for future in running:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            pbar.close()

    def execute(self) -> None:
        for _task, _value in self.run():
            pass

    def _execute_task(
        self,
        task: Task,
        arguments: Dict[str, Any],
        execution_device: torch.device,
    ) -> Any:
        if task.uses_accelerator() and execution_device.type == "cuda":
            torch.cuda.set_device(execution_device)

        execute_start = time.perf_counter()
        result = task.execute(**arguments)
        self._log_task_phase(task, "execute", time.perf_counter() - execute_start)

        store_start = time.perf_counter()
        prepared = self._prepare_result(task, result, execution_device)
        self._log_task_phase(task, "store", time.perf_counter() - store_start)
        return prepared
