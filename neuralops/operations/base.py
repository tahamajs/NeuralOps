#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Operation Abstract Base Class and Contract Definitions.

Implements Specification Section 5.1 (Registration Contract Interface).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class OperationContract:
    name: str
    version: str
    input_state: str
    output_state: str
    params_schema: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    declares_mutations: List[str] = field(default_factory=list)
    deterministic_given_seed: bool = True
    composable_with_self: bool = False
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    ordering_constraints: List[str] = field(default_factory=list)


class Operation(ABC):
    """Abstract Base Class for all versioned NeuralOps transformation operations."""

    name: str
    version: str
    input_state: str
    output_state: str
    params_schema: Dict[str, Any]
    preconditions: List[str]
    postconditions: List[str]
    declares_mutations: List[str]
    deterministic_given_seed: bool
    composable_with_self: bool

    contract: Optional[OperationContract] = None
    requires: List[str] = []
    produces: List[str] = []
    mutates: List[str] = []
    conflicts: List[str] = []
    ordering_constraints: List[str] = []

    def __init__(self, contract: Optional[OperationContract] = None):
        c = contract or getattr(self, "contract", None)
        if c is not None:
            self.contract = c
            self.name = c.name
            self.version = c.version
            self.input_state = c.input_state
            self.output_state = c.output_state
            self.params_schema = c.params_schema
            self.preconditions = c.preconditions
            self.postconditions = c.postconditions
            self.declares_mutations = c.declares_mutations
            self.deterministic_given_seed = c.deterministic_given_seed
            self.composable_with_self = c.composable_with_self
            self.mutates = list(c.declares_mutations)
            self.requires = list(getattr(c, "requires", []))
            self.produces = list(getattr(c, "produces", []))
            self.conflicts = list(getattr(c, "conflicts", []))
            self.ordering_constraints = list(getattr(c, "ordering_constraints", []))

    def simulate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Simulate output tensor shape given input shape."""
        return input_shape

    def validate_preconditions(self, model_state: Dict[str, Any]) -> List[str]:
        """Verify whether the model satisfies all static prerequisites for this operation."""
        errors: List[str] = []
        curr_state = model_state.get("state", "loaded")
        if self.input_state != "*" and curr_state != self.input_state:
            errors.append(
                f"State mismatch for '{self.name}': requires '{self.input_state}', currently in '{curr_state}'"
            )
        return errors

    def infer_output_state(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict post-operation model state dictionary without running weights mutation."""
        updated = dict(model_state)
        updated["state"] = self.output_state
        return updated

    def estimated_time_and_memory(self, model: Any) -> Dict[str, Any]:
        """Estimate execution latency (ms) and peak VRAM allocation (MB)."""
        param_count = sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else 100_000_000
        # Default rough baseline estimate
        return {
            "estimated_time_ms": round(param_count / 1_000_000 * 5.0, 2),
            "estimated_vram_mb": round(param_count * 4 / (1024 * 1024) * 1.2, 2),
        }

    @abstractmethod
    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Execute the operation on model according to config.

        Returns:
            Tuple of (transformed_model, evidence_dict).
        """
        raise NotImplementedError
