#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Adaptation Heal Operation Contract.

Specification Section 6.12: adaptation.heal@1.0.0
Connects to DoRA (Weight-Decomposed Low-Rank Adaptation) directional healing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import OperationExecutionError
from neuralops.operations.registry import register_operation
from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer

logger = logging.getLogger("NeuralOps.Operations.Adaptation.Heal")

CONTRACT = OperationContract(
    name="adaptation.heal",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["calibration_tokens"],
        "properties": {
            "calibration_tokens": {"type": "integer", "minimum": 1024},
            "mode": {"type": "string", "enum": ["norm_only", "bias_only", "full_quick"]},
        },
        "additionalProperties": False,
    },
    preconditions=["model.is_mutated"],
    postconditions=["model.is_calibrated"],
    declares_mutations=["*.ln_*", "*.norm*"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation(override=True)
class AdaptationHealOperation(Operation):
    """Operation for healing mutated/compressed models via DoRA directional adaptation."""

    def __init__(self) -> None:
        """Initialize AdaptationHealOperation with canonical contract."""
        super().__init__(contract=CONTRACT)

    def apply(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply DoRA directional adaptation and calibration to heal model.

        Args:
            model: Mutated or compressed neural network to heal.
            config: Parameters including rank, lora_alpha, targets, calibration_tokens, mode,
                    dataloader / calibration_batches, steps, and collapse_after_heal.

        Returns:
            Tuple of (healed_model, evidence_dict).

        Raises:
            OperationExecutionError: If adaptation healing fails.
        """
        try:
            rank = int(config.get("rank", 8))
            lora_alpha = float(config.get("lora_alpha", 16.0))
            calibration_tokens = int(config.get("calibration_tokens", 1024))
            mode = str(config.get("mode", "full_quick"))
            target_substrings: Optional[List[str]] = config.get("targets", None)
            collapse_after_heal = bool(config.get("collapse_after_heal", True))
            steps = int(config.get("steps", 10))
            lr = float(config.get("lr", 1e-4))
            dataloader = config.get("dataloader", config.get("calibration_data", None))

            # Instantiate DoRA healer
            healer = DoRAHealingHealer(
                model=model,
                rank=rank,
                lora_alpha=lora_alpha,
                target_substrings=target_substrings,
            )
            injected_count = healer.inject_dora_adapters()

            # Mode adjustments: configure trainable parameters based on healing mode
            if mode == "norm_only":
                for p in model.parameters():
                    p.requires_grad = False
                for name, mod in model.named_modules():
                    if any(n in name.lower() for n in ("norm", "ln_", "layernorm", "rmsnorm")):
                        for p in mod.parameters():
                            p.requires_grad = True
            elif mode == "bias_only":
                for name, p in model.named_parameters():
                    p.requires_grad = "bias" in name.lower()
            else:
                # full_quick: DoRA adapters (A, B, magnitude) + norms/biases are trainable
                for p in model.parameters():
                    p.requires_grad = False
                for wrapper in healer.injected_wrappers.values():
                    wrapper.magnitude.requires_grad = True
                    wrapper.lora_A.requires_grad = True
                    wrapper.lora_B.requires_grad = True
                    if wrapper.bias is not None:
                        wrapper.bias.requires_grad = True

            trainable_params = [p for p in model.parameters() if p.requires_grad]

            # Run micro-calibration / healing steps if dataloader or calibration tensors provided
            losses: List[float] = []
            if dataloader is not None and trainable_params:
                optimizer = torch.optim.AdamW(trainable_params, lr=lr)
                step = 0
                for batch in dataloader:
                    if step >= steps:
                        break
                    optimizer.zero_grad()
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    elif isinstance(batch, (tuple, list)):
                        outputs = model(*batch)
                    else:
                        outputs = model(batch)

                    # Compute surrogate healing loss (e.g. logit norm / entropy / MSE if targets given)
                    if hasattr(outputs, "logits"):
                        loss = outputs.logits.square().mean()
                    elif isinstance(outputs, torch.Tensor):
                        loss = outputs.square().mean()
                    else:
                        loss = torch.tensor(0.0, requires_grad=True)

                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach().item()))
                    step += 1

            if collapse_after_heal and injected_count > 0:
                healer.collapse_adapters()
                collapsed = True
            else:
                collapsed = False

            evidence: Dict[str, Any] = {
                "operation": "adaptation.heal",
                "calibration_tokens": calibration_tokens,
                "mode": mode,
                "rank": rank,
                "lora_alpha": lora_alpha,
                "injected_dora_adapters": injected_count,
                "collapsed": collapsed,
                "steps_run": len(losses),
                "losses": losses,
            }
            logger.info(
                "Successfully executed adaptation.heal (mode=%s, injected=%d, collapsed=%s)",
                mode,
                injected_count,
                collapsed,
            )
            return model, evidence

        except Exception as exc:
            if isinstance(exc, OperationExecutionError):
                raise
            raise OperationExecutionError(f"Failed to execute adaptation.heal: {exc}") from exc
