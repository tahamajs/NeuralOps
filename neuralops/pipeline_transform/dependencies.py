#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dependency Graph and Pre-Execution Conflict Detection Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from neuralops.operations.registry import OPERATION_REGISTRY
from neuralops.recipe.loader import load_recipe
from neuralops.recipe.schema import Recipe


class ConflictSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class Conflict:
    """Pre-execution structural, ordering, or mutation incompatibility."""

    severity: ConflictSeverity
    code: str
    message: str
    stage_id: Optional[str] = None
    operation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "stage_id": self.stage_id,
            "operation": self.operation,
        }


@dataclass
class DependencyNode:
    """Node in transformation dependency DAG."""

    stage_id: str
    operation_type: str
    version: str
    config: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    requires: Set[str] = field(default_factory=set)
    produces: Set[str] = field(default_factory=set)
    mutates: Set[str] = field(default_factory=set)
    conflicts: Set[str] = field(default_factory=set)
    ordering_constraints: Set[str] = field(default_factory=set)


class DependencyGraph:
    """Directed Acyclic Graph representing recipe operation dependencies."""

    def __init__(self) -> None:
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, node: DependencyNode) -> None:
        self.nodes[node.stage_id] = node
        self.edges.setdefault(node.stage_id, [])
        for dep in node.depends_on:
            self.edges.setdefault(dep, []).append(node.stage_id)


class ConflictDetector:
    """Analyzes a recipe dependency graph and detects structural and ordering conflicts."""

    def __init__(self, recipe: Recipe, adapter: Optional[Any] = None) -> None:
        self.recipe = recipe
        self.adapter = adapter
        self.graph = DependencyGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        for idx, op_spec in enumerate(self.recipe.operations):
            st_id = op_spec.id or f"stage_{idx}"
            op_val = OPERATION_REGISTRY.get(op_spec.type, {}).get(op_spec.version)
            op_instance = op_val() if isinstance(op_val, type) else op_val

            reqs = set(getattr(op_instance, "requires", []))
            prods = set(getattr(op_instance, "produces", []))
            muts = set(getattr(op_instance, "mutates", []))
            confls = set(getattr(op_instance, "conflicts", []))
            ord_c = set(getattr(op_instance, "ordering_constraints", []))

            # Infer defaults for standard known operations if not explicit
            if op_spec.type == "weights.compress":
                prods.add("low_rank_projection")
                ord_c.add("after:attention.replace")
            elif op_spec.type == "attention.replace":
                confls.add("low_rank_projection")

            node = DependencyNode(
                stage_id=st_id,
                operation_type=op_spec.type,
                version=op_spec.version,
                config=dict(op_spec.config),
                depends_on=list(op_spec.depends_on),
                requires=reqs,
                produces=prods,
                mutates=muts,
                conflicts=confls,
                ordering_constraints=ord_c,
            )
            self.graph.add_node(node)

    def detect_conflicts(self) -> List[Conflict]:
        conflicts: List[Conflict] = []
        produced_features: Set[str] = set()
        seen_operations: List[str] = []

        # Simulated tensor shape tracking: (batch, seq, hidden)
        current_shape: Tuple[int, ...] = (2, 16, 64)
        if hasattr(self.adapter, "hidden_size"):
            current_shape = (2, 16, int(self.adapter.hidden_size))

        for idx, op_spec in enumerate(self.recipe.operations):
            st_id = op_spec.id or f"stage_{idx}"
            node = self.graph.nodes.get(st_id)
            if node is None:
                continue

            op_val = OPERATION_REGISTRY.get(op_spec.type, {}).get(op_spec.version)
            op_instance = op_val() if isinstance(op_val, type) else op_val

            # 1. Ordering conflict: e.g. compress before attention.replace
            for constraint in node.ordering_constraints:
                if constraint.startswith("after:"):
                    target_op = constraint.split("after:", 1)[1]
                    # If this op is run, but target_op appears LATER in the recipe
                    later_ops = [o.type for o in self.recipe.operations[idx + 1 :]]
                    if target_op in later_ops:
                        conflicts.append(
                            Conflict(
                                severity=ConflictSeverity.ERROR,
                                code="E_CONFLICT_ORDER",
                                message=(
                                    f"Ordering violation at stage '{st_id}' ({node.operation_type}): "
                                    f"must be executed after '{target_op}', but '{target_op}' is scheduled later in pipeline."
                                ),
                                stage_id=st_id,
                                operation=node.operation_type,
                            )
                        )

            # 2. Incompatible state/artifact conflict:
            # E.g., attention.replace conflicts with previously produced low_rank_projection
            for confl_feature in node.conflicts:
                if confl_feature in produced_features:
                    conflicts.append(
                        Conflict(
                            severity=ConflictSeverity.ERROR,
                            code="E_CONFLICT_MUTATION",
                            message=(
                                f"Incompatible transformation at stage '{st_id}' ({node.operation_type}): "
                                f"conflicts with prior artifact '{confl_feature}' produced by earlier stage. "
                                f"Compression factors are invalidated by subsequent attention surgery."
                            ),
                            stage_id=st_id,
                            operation=node.operation_type,
                        )
                    )

            # 3. Missing requires
            for req in node.requires:
                if req not in produced_features:
                    conflicts.append(
                        Conflict(
                            severity=ConflictSeverity.ERROR,
                            code="E_CONFLICT_DEPENDENCY",
                            message=(
                                f"Missing prerequisite at stage '{st_id}' ({node.operation_type}): "
                                f"requires '{req}' which has not been produced."
                            ),
                            stage_id=st_id,
                            operation=node.operation_type,
                        )
                    )

            # 4. Shape propagation simulation
            if op_instance is not None:
                try:
                    new_shape = op_instance.simulate_output_shape(current_shape)
                    # Check shape validity
                    if len(new_shape) != len(current_shape):
                        conflicts.append(
                            Conflict(
                                severity=ConflictSeverity.ERROR,
                                code="E_CONFLICT_SHAPE_MISMATCH",
                                message=(
                                    f"Shape rank mismatch at stage '{st_id}' ({node.operation_type}): "
                                    f"expected {len(current_shape)}D tensor, simulated {len(new_shape)}D."
                                ),
                                stage_id=st_id,
                                operation=node.operation_type,
                            )
                        )
                    current_shape = new_shape
                except Exception as e:
                    conflicts.append(
                        Conflict(
                            severity=ConflictSeverity.WARNING,
                            code="E_CONFLICT_SHAPE_SIMULATION_FAILED",
                            message=f"Shape simulation warning at stage '{st_id}': {e}",
                            stage_id=st_id,
                            operation=node.operation_type,
                        )
                    )

            # Register produced features and seen operations
            produced_features.update(node.produces)
            seen_operations.append(node.operation_type)

        # Sort conflicts by severity: ERROR before WARNING
        conflicts.sort(key=lambda c: 0 if c.severity == ConflictSeverity.ERROR else 1)
        return conflicts


def analyze_recipe(
    recipe: Union[str, Recipe, Dict[str, Any]],
    adapter: Optional[Any] = None,
) -> List[Conflict]:
    """Pre-execution analysis of operation ordering, structural dependencies, and conflicts."""
    if isinstance(recipe, str):
        recipe_obj = load_recipe(recipe)
    elif isinstance(recipe, dict):
        recipe_obj = Recipe.from_dict(recipe)
    else:
        recipe_obj = recipe

    detector = ConflictDetector(recipe_obj, adapter=adapter)
    return detector.detect_conflicts()


__all__ = [
    "Conflict",
    "ConflictDetector",
    "ConflictSeverity",
    "DependencyGraph",
    "DependencyNode",
    "analyze_recipe",
]
