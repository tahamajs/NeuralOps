# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Base Trainer Module

Defines the abstract base trainer interface and unified high-level Trainer class for model training.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Fit the model on the provided dataset."""
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Run evaluate and return results."""
        pass


class Trainer(BaseTrainer):
    """Unified high-level Trainer for fine-tuning, compression recovery, and surgery pipelines."""

    def __init__(
        self,
        model: Any,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        config: dict[str, Any] | None = None,
        results_manager: Any | None = None,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.results_manager = results_manager
        self.device = device
        self.kwargs = kwargs
        self._delegate = None
        if train_dataset is not None:
            from .custom_loop_trainer import CustomLoopTrainer

            self._delegate = CustomLoopTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                config=self.config,
                results_manager=self.results_manager,
                device=self.device,
                **kwargs,
            )

    def train(self) -> dict[str, Any]:
        """Fit the model on the provided dataset."""
        if self._delegate is not None:
            return self._delegate.train()
        return {"status": "trained", "model": self.model, "config": self.config}

    def evaluate(self) -> dict[str, Any]:
        """Run evaluation on the model."""
        if self._delegate is not None:
            return self._delegate.evaluate()
        return {"status": "evaluated", "model": self.model}
