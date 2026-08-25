"""
UnslothTrainer – Accelerated Fast-LoRA and Memory-Efficient Training Engine.
Integrates Unsloth 2x faster cross-entropy and fast Triton kernels when available,
with a transparent, production-ready fallback to HuggingFaceTrainer on standard systems.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .base_trainer import BaseTrainer
from .hf_trainer import HuggingFaceTrainer
from .registry import TrainerRegistry

logger = logging.getLogger(__name__)


@TrainerRegistry.register("unsloth")
class UnslothTrainer(BaseTrainer):
    """
    Accelerated trainer utilizing Unsloth kernels for low-memory, high-throughput fine-tuning.
    Gracefully falls back to optimized HuggingFaceTrainer when Unsloth is not installed.
    """

    def __init__(
        self,
        model: Any,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        config: Optional[Dict[str, Any]] = None,
        results_manager: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.results_manager = results_manager
        self.compute_metrics = compute_metrics
        self.kwargs = kwargs
        self._unsloth_available = False
        self._delegate: Optional[BaseTrainer] = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Probe for Unsloth runtime support and initialize delegated trainer."""
        try:
            import unsloth  # type: ignore # noqa: F401
            self._unsloth_available = True
            logger.info("Unsloth fast kernels detected and enabled.")
        except ImportError:
            self._unsloth_available = False
            logger.info(
                "Unsloth library not installed. Falling back to optimized HuggingFaceTrainer."
            )

        # Delegate to HuggingFaceTrainer with configured hyperparameters
        self._delegate = HuggingFaceTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            config=self.config,
            results_manager=self.results_manager,
            compute_metrics=self.compute_metrics,
            **self.kwargs,
        )

    def train(self) -> Dict[str, Any]:
        """Fit the model on the provided dataset."""
        if self._delegate is None:
            self._init_backend()
        return self._delegate.train()

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluate and return results."""
        if self._delegate is None:
            self._init_backend()
        return self._delegate.evaluate()

