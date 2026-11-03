#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TTT-Online: Test-Time Training Adaptation per Benchmark Prompt.

Implements self-supervised online adaptation on the evaluation prompt itself
prior to autoregressive generation or classification:
    S_{test} = S_0 - eta * grad_S ||x_{prompt} - hat{x}_{prompt}||_2^2
Enables zero-shot domain adaptation directly on the evaluation set without
requiring external fine-tuning sets.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NeuralOps.Trainer.Regimes.TTTOnline")


class TTTOnlineAdapter:
    """
    Online Test-Time Training (TTT) manager that adapts recurrent state
    or attention projection weights on test prompts.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 1e-3,
        inner_steps: int = 1,
        target_substrings: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.target_substrings = target_substrings or ["out_proj", "o_proj", "v_proj"]

    def _get_target_parameters(self) -> List[nn.Parameter]:
        params = []
        for name, p in self.model.named_parameters():
            if any(sub in name for sub in self.target_substrings) and p.requires_grad:
                params.append(p)
        return params

    def adapt_on_prompt(
        self,
        prompt_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask_ratio: float = 0.15,
    ) -> Dict[str, float]:
        """
        Executes self-supervised reconstruction adaptation step on prompt:
            L_{self} = ||x - hat{x}||_2^2 on masked token positions
        """
        target_params = self._get_target_parameters()
        if not target_params:
            logger.debug("No eligible target parameters found for TTT adaptation.")
            return {"ttt_loss": 0.0}

        self.model.eval()

        if isinstance(prompt_batch, dict):
            input_tensor = prompt_batch.get("input_ids")
            if input_tensor is None:
                input_tensor = prompt_batch.get("inputs_embeds")
        elif isinstance(prompt_batch, torch.Tensor):
            input_tensor = prompt_batch
        else:
            return {"ttt_loss": 0.0}

        if input_tensor is None or input_tensor.ndim not in (2, 3):
            return {"ttt_loss": 0.0}

        # Backup original parameter data
        backups = [p.data.clone() for p in target_params]

        total_loss = 0.0
        for _ in range(self.inner_steps):
            # Self-supervised next-token or feature-reconstruction loss
            if isinstance(prompt_batch, dict):
                outputs = self.model(**prompt_batch)
            else:
                outputs = self.model(prompt_batch)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                logits = outputs[0]
            else:
                break

            # Handle discrete token IDs vs continuous feature embeddings
            if input_tensor.dtype in (torch.int32, torch.int64, torch.long, torch.int) and input_tensor.ndim == 2:
                # Next token prediction self-supervised cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
            else:
                # Continuous reconstruction / associative state prediction loss ||x - hat{x}||_2^2
                target_x = input_tensor
                pred_x = logits
                if pred_x.shape[-1] != target_x.shape[-1]:
                    # Target dimension projection or slice if shapes differ
                    min_dim = min(pred_x.shape[-1], target_x.shape[-1])
                    pred_x = pred_x[..., :min_dim]
                    target_x = target_x[..., :min_dim]
                loss = F.mse_loss(pred_x, target_x, reduction="mean")
            total_loss += loss.item()

            grads = torch.autograd.grad(
                loss, target_params, retain_graph=False, create_graph=False, allow_unused=True
            )

            with torch.no_grad():
                for p, g in zip(target_params, grads):
                    if g is not None:
                        p.data.add_(g, alpha=-self.inner_lr)

        avg_loss = total_loss / max(1, self.inner_steps)
        return {"ttt_loss": avg_loss}

    def reset_parameters(self, backups: List[torch.Tensor]) -> None:
        """Restores parameters to pre-prompt state."""
        target_params = self._get_target_parameters()
        with torch.no_grad():
            for p, b in zip(target_params, backups):
                p.data.copy_(b)


__all__ = ["TTTOnlineAdapter"]
