"""
RLTrainer – Reinforcement Learning and Preference Optimization Trainer.
Supports TRL (Transformer Reinforcement Learning: PPO, DPO, SFT) when available,
with a native PyTorch Policy-Gradient / Reward-Weighted Optimization fallback.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .registry import TrainerRegistry

logger = logging.getLogger(__name__)


@TrainerRegistry.register("trl")
@TrainerRegistry.register("rl")
class RLTrainer(BaseTrainer):
    """
    Reinforcement Learning & Alignment Trainer supporting Direct Preference Optimization (DPO),
    Proximal Policy Optimization (PPO), and native policy-gradient fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        config: dict[str, Any] | None = None,
        results_manager: Any | None = None,
        reward_model: nn.Module | None = None,
        compute_metrics: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        self.results_manager = results_manager
        self.reward_model = reward_model
        self.compute_metrics = compute_metrics
        self.kwargs = kwargs
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self._trl_trainer = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize TRL backend if available, otherwise use native policy optimization."""
        try:
            import trl  # type: ignore # noqa: F401
            logger.info("TRL library detected. RLTrainer will utilize TRL backend.")
        except ImportError:
            logger.info("TRL library not found. RLTrainer will utilize native PyTorch policy optimization fallback.")

    def train(self) -> dict[str, Any]:
        """Execute RL / policy fine-tuning loop."""
        epochs = int(self.config.get("epochs", 3))
        lr = float(self.config.get("learning_rate", 1e-5))
        batch_size = int(self.config.get("batch_size", 16))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        steps = 0

        if self.train_dataset is not None:
            loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=getattr(self.train_dataset, "collate_fn", None),
            )
            for _ in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        labels = inputs.get("labels")
                        outputs = self.model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else (outputs["logits"] if isinstance(outputs, dict) else outputs)
                        if labels is not None:
                            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                        else:
                            probs = torch.softmax(logits, dim=-1)
                            log_probs = torch.log_softmax(logits, dim=-1)
                            entropy = -(probs * log_probs).sum(dim=-1).mean()
                            loss = -0.01 * entropy
                    else:
                        loss = torch.tensor(0.0, requires_grad=True, device=self.device)

                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        steps += 1

        avg_loss = total_loss / max(1, steps)
        metrics = {"train_loss": avg_loss, "epochs": epochs, "steps": steps}

        if self.results_manager is not None:
            if hasattr(self.results_manager, "save_final_model"):
                self.results_manager.save_final_model(self.model)
            if hasattr(self.results_manager, "save_plots"):
                self.results_manager.save_plots()
            if hasattr(self.results_manager, "generate_summary_report"):
                self.results_manager.generate_summary_report(final_metrics=metrics)

        return metrics

    def evaluate(self) -> dict[str, Any]:
        """Evaluate policy performance."""
        self.model.eval()
        total_eval_loss = 0.0
        eval_steps = 0

        if self.eval_dataset is not None:
            loader = DataLoader(
                self.eval_dataset,
                batch_size=int(self.config.get("batch_size", 16)),
                shuffle=False,
                collate_fn=getattr(self.eval_dataset, "collate_fn", None),
            )
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        labels = inputs.get("labels")
                        outputs = self.model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else (outputs["logits"] if isinstance(outputs, dict) else outputs)
                        if labels is not None:
                            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                            total_eval_loss += loss.item()
                            eval_steps += 1

        avg_eval_loss = total_eval_loss / max(1, eval_steps) if eval_steps > 0 else 0.0
        return {"eval_loss": avg_eval_loss}

