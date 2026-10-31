#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L3 Behavioral Attention Pattern Similarity Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


def _has_attention_mechanisms(model: nn.Module) -> bool:
    for name, _ in model.named_modules():
        lowered = name.lower()
        if "attn" in lowered or "attention" in lowered or "self_attn" in lowered:
            return True
    return False


class AttentionPatternsGate(Gate):
    """Measures Frobenius norm / cosine similarity of attention weight distributions."""

    name = "attention_patterns"
    level = "L3"
    severity = GateSeverity.WARNING

    def __init__(self, min_similarity: float = 0.80, **kwargs: Any) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.min_similarity = min_similarity

    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> GateResult:
        t0 = time.perf_counter()
        active_model = model_after or model
        base_model = model_before or baseline_model

        if active_model is None or base_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped attention_patterns: requires both model_before and model_after",
                duration_seconds=time.perf_counter() - t0,
            )

        # Check if models contain attention modules per spec §7.1:
        # "returns skipped=True on models without attention (e.g., Mamba), doesn't fail"
        if not _has_attention_mechanisms(active_model) or not _has_attention_mechanisms(base_model):
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Model does not contain recognizable attention mechanisms; skipped",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = float(context.get("min_attention_similarity", self.min_similarity)) if context else self.min_similarity

        active_model.eval()
        base_model.eval()
        dummy_in = _get_dummy_input(active_model, batch_size=2, seq_len=16)

        captured_base: List[torch.Tensor] = []
        captured_test: List[torch.Tensor] = []

        def _hook_b(module: nn.Module, inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor) and out.ndim >= 3:
                captured_base.append(out.detach())
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                captured_base.append(out[0].detach())

        def _hook_t(module: nn.Module, inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor) and out.ndim >= 3:
                captured_test.append(out.detach())
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                captured_test.append(out[0].detach())

        hooks = []
        for name, mod in base_model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                hooks.append(mod.register_forward_hook(_hook_b))
                break

        for name, mod in active_model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                hooks.append(mod.register_forward_hook(_hook_t))
                break

        try:
            with torch.no_grad():
                _ = base_model(dummy_in)
                _ = active_model(dummy_in)
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Forward pass exception during attention pattern collection: {e}",
                duration_seconds=time.perf_counter() - t0,
            )
        finally:
            for h in hooks:
                h.remove()

        if not captured_base or not captured_test:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Attention activations not captured; skipping gate",
                duration_seconds=time.perf_counter() - t0,
            )

        tb = captured_base[0].view(-1).float()
        tt = captured_test[0].view(-1).float()
        if tb.numel() != tt.numel():
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Attention output dimension changed; pattern similarity skipped",
                duration_seconds=time.perf_counter() - t0,
            )

        sim = F.cosine_similarity(tb.unsqueeze(0), tt.unsqueeze(0)).item()
        passed = sim >= threshold

        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=sim,
            threshold=threshold,
            message=f"Attention pattern similarity: {sim:.4f} (threshold >= {threshold:.4f})",
            duration_seconds=time.perf_counter() - t0,
        )


__all__ = ["AttentionPatternsGate"]
