#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Connectivity and Orphan Module Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


class ConnectivityGate(Gate):
    """Verifies that all modules in the model tree are properly linked with no orphans."""

    name = "connectivity"
    level = "L1"
    severity = GateSeverity.ERROR

    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        active_model = model_after or model
        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message="No model provided to connectivity gate",
                duration_seconds=time.perf_counter() - t0,
            )

        total_modules = len(list(active_model.modules()))
        named_modules = len(list(active_model.named_modules()))
        registered_set = set(active_model.modules())

        # Check for unlinked / orphan modules in __dict__ of the model or submodules
        orphan_found = False
        orphan_name = ""

        for m_name, m in active_model.named_modules():
            # Check if any attribute in __dict__ is an nn.Module but not in registered_set
            for attr_name, val in m.__dict__.items():
                if isinstance(val, nn.Module) and val not in registered_set:
                    orphan_found = True
                    orphan_name = f"{m_name}.{attr_name}" if m_name else attr_name
                    break
            if orphan_found:
                break

        passed = (total_modules == named_modules) and not orphan_found
        msg = (
            f"All {named_modules} modules properly rooted"
            if passed
            else (
                f"Orphan module '{orphan_name}' detected in model graph"
                if orphan_found
                else f"Module count mismatch: {total_modules} total vs {named_modules} named"
            )
        )

        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=named_modules,
            threshold=total_modules,
            message=msg,
            duration_seconds=time.perf_counter() - t0,
        )


# Backward-compatible alias
NoOrphanModulesGate = ConnectivityGate

__all__ = ["ConnectivityGate", "NoOrphanModulesGate"]
