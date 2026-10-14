"""
HuggingFace TrainerCallback for automating NeuralOps surgery curricula.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger("NeuralOps.Callback")


class NeuralOpsSurgeryCallback(TrainerCallback):
    """
    Automates MOHAWK 3-stage transitions, alpha-homotopy annealing,
    and finalize_surgery() inside any standard HuggingFace Trainer.
    """

    def __init__(
        self,
        total_steps: int = 1000,
        stage1_steps: int = 200,
        stage2_steps: int = 500,
        homotopy_schedule: str = "cosine",
    ) -> None:
        self.total_steps = total_steps
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.homotopy_schedule = homotopy_schedule

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if model is None:
            return

        from neuralops.api import set_surgery_alpha, set_surgery_stage

        # Stage transitions
        if step < self.stage1_steps:
            set_surgery_stage(model, 1)
        elif step < self.stage2_steps:
            set_surgery_stage(model, 2)
        else:
            set_surgery_stage(model, 3)

        # Alpha annealing
        alpha = min(1.0, max(0.0, step / max(1, self.total_steps)))
        set_surgery_alpha(model, alpha)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if model is not None:
            from neuralops.api import finalize_surgery
            logger.info("NeuralOpsSurgeryCallback: Finalizing surgery on train end.")
            finalize_surgery(model)


__all__ = ["NeuralOpsSurgeryCallback"]
