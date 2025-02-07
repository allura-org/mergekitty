# Partly Copyright (C) 2025 Arcee AI
# Partly Copyright (C) 2025 Allura-org
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
"""
Module for computational graph execution.

Classes:
    Task: Abstract base class representing a computational task.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from pydantic import BaseModel
from typing_extensions import Generic, TypeVar

ValueT = TypeVar("ValueT")


class Task(ABC, BaseModel, Generic[ValueT], frozen=True):
    """
    Abstract base class representing a task in a computational graph.

    This class should be extended to define specific tasks. Each task can have arguments (dependencies) and a defined execution strategy.

    Attributes:
        Generic[ValueT] (TypeVar): The type of the value that the task returns upon execution.

    Methods:
        arguments: Abstract method to define task arguments (dependencies).
        execute: Abstract method to execute the task.
        priority: Returns the priority of the task for scheduling purposes.
        group_label: Returns an optional label for task grouping.
    """

    @abstractmethod
    def arguments(self) -> Dict[str, "Task"]:
        """
        Returns a dictionary of arguments required for this task. The keys of the dictionary
        are argument names, and the values are Task instances. These keys correspond to the
        keyword argument names expected by the execute method.

        For example, if this method returns {'input1': taskA, 'input2': taskB}, the execute
        method should expect to be called as execute(input1=valueA, input2=valueB), where
        valueA and valueB are the outputs of taskA and taskB respectively.

        Returns:
            Dict[str, "Task"]: A dictionary mapping argument names to Task instances.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ValueT:
        """
        Executes the task using the results of its dependencies.

        The keyword arguments (**kwargs) for this method are dynamically determined based on
        the dictionary returned by the 'arguments' method. Each key in the 'arguments' method's
        return dictionary becomes a keyword argument in this method, with its value being
        the result of the corresponding task's execution.

        Returns:
            ValueT: The result of the task execution.
        """
        ...

    def priority(self) -> int:
        """
        Returns the priority of the task for scheduling.

        Higher numbers indicate higher priority. Default is 0.

        Returns:
            int: The priority of the task.
        """
        return 0

    def group_label(self) -> Optional[str]:
        """
        Returns an optional label used for grouping tasks together.

        Returns:
            Optional[str]: The group label of the task, if any.
        """
        return None

    def uses_accelerator(self) -> bool:
        """
        Returns True if the task can take advantage of matrix operation
        acceleration (such as on a GPU).
        """
        return False
