# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025 Allura-org

import torch
from typing import Any, Dict, Iterator, List, Set, Tuple, Union
import multiprocessing
import networkx
import tqdm


from mergekitty.task import Task

# TODO: this cannot *possibly* be the best way to do this. there must be a better way. i refuse to believe that this is the best way to do this.
RANK = None
VALUES = None


class MultiThreadedExecutor:
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
        math_devices: List[torch.device] = [torch.device("cpu")],
        storage_devices: List[torch.device] = [torch.device("cpu")],
    ):
        """

        Initializes the Executor with a list of tasks and device configurations.

        Args:
            tasks (List[Task]): The list of tasks to be executed.
            math_device (torch.device, optional): The device for tensor computations. Defaults to CPU.
            storage_device (torch.device, optional): The device for storing results. Defaults to CPU.
        """
        assert len(math_devices) == len(storage_devices), (
            "math_devices and storage_devices must have the same length (the amount of processes to spawn)"
        )
        self.math_devices = math_devices
        self.storage_devices = storage_devices
        self.schedule = self._make_schedule(tasks)
        self.targets = tasks

    @staticmethod
    def _initialize_process(ranks: multiprocessing.Queue, values: Dict[Task, Any]):
        global RANK, VALUES
        RANK = ranks.get()
        VALUES = values

    def _execute_task(self, task_idx: Tuple[int, Task]) -> Tuple[Task, Any]:
        global RANK, VALUES

        _idx, task = task_idx
        rank = RANK
        math_device = self.math_devices[rank]
        storage_device = self.storage_devices[rank]

        use_math_device = task.uses_accelerator()

        arguments = {}
        for name, dep in task.arguments().items():
            value = VALUES[dep]

            if use_math_device:
                if isinstance(value, torch.Tensor) and value.device != math_device:
                    value = value.to(math_device)
                elif isinstance(value, dict):
                    for key in value:
                        if (
                            isinstance(value[key], torch.Tensor)
                            and value[key].device != math_device
                        ):
                            value[key] = value[key].to(math_device)

            arguments[name] = value
            del value

        res = task.execute(**arguments)
        del arguments

        if isinstance(res, torch.Tensor) and res.device != storage_device:
            res = res.to(storage_device)

        VALUES[task] = res

        return (task, res)

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

        manager = multiprocessing.Manager()
        values = manager.dict()
        ranks = multiprocessing.Queue()
        for i in range(len(self.math_devices)):
            ranks.put(i)

        with multiprocessing.Pool(
            len(self.math_devices),
            initializer=self._initialize_process,
            initargs=(ranks, values),
        ) as pool:
            it = pool.imap(self._execute_task, list(enumerate(self.schedule)))
            for task, _ in tqdm.tqdm(
                it, total=len(self.schedule), disable=quiet, desc="Executing graph"
            ):
                if task in self.targets:
                    yield (task, values[task])

    def execute(self) -> None:
        """
        Execute all tasks and discard results.
        """
        for task, value in self.run():
            pass

    DUMMY_TASK_VALUE = "!!DUMMY!!"

    # TODO: i wonder if this is worth multiprocessing too?
    def _make_schedule(self, targets: List[Task]) -> List[Task]:
        self.schedule = []
        self.dependencies = self._build_dependencies(targets)

        edge_tups = []
        for node in self.dependencies:
            for dependency in self.dependencies[node]:
                edge_tups.append((dependency, node))

        for task in targets:
            # add edges from a dummy node to each target to guarantee
            # they will be included in the final schedule
            edge_tups.append((MultiThreadedExecutor.DUMMY_TASK_VALUE, task))

        def _compare_key(task: Union[Task, str]):
            if task == MultiThreadedExecutor.DUMMY_TASK_VALUE:
                return ("", 0)
            return (
                task.group_label() or "",
                -task.priority(),
            )

        graph = networkx.DiGraph(edge_tups)
        res = [
            t
            for t in networkx.lexicographical_topological_sort(graph, key=_compare_key)
            if t != MultiThreadedExecutor.DUMMY_TASK_VALUE
        ]
        return res

    def _build_dependencies(self, targets: List[Task]) -> Dict[Task, Set[Task]]:
        task_dependencies: Dict[Task, Set[Task]] = {}
        to_process = list(targets)
        while to_process:
            child = to_process.pop()
            if child in task_dependencies:
                continue

            task_dependencies[child] = set()
            for _, dep in child.arguments().items():
                task_dependencies[child].add(dep)
                to_process.append(dep)
        return task_dependencies
