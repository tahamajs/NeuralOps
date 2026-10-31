#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Execution Plan and Stage Data Models for Transactional Transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from neuralops.operations.base import OperationContract


@dataclass
class Stage:
    """An executable transformation step within an execution plan."""

    id: str
    operation_type: str
    version: str
    config: Dict[str, Any]
    contract: Optional[OperationContract] = None
    depends_on: List[str] = field(default_factory=list)
    estimated_parameters_delta: int = 0

    def to_dict(self) -> Dict[str, Any]:
        contract_dict = None
        if self.contract is not None:
            if hasattr(self.contract, "to_dict"):
                contract_dict = self.contract.to_dict()
            elif isinstance(self.contract, OperationContract):
                contract_dict = asdict(self.contract)

        return {
            "id": self.id,
            "operation_type": self.operation_type,
            "version": self.version,
            "config": dict(self.config),
            "contract": contract_dict,
            "depends_on": list(self.depends_on),
            "estimated_parameters_delta": self.estimated_parameters_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage:
        contract_data = data.get("contract")
        contract = None
        if contract_data and isinstance(contract_data, dict):
            contract = OperationContract(
                name=contract_data.get("name", ""),
                version=contract_data.get("version", "1.0.0"),
                input_state=contract_data.get("input_state", "*"),
                output_state=contract_data.get("output_state", "*"),
                params_schema=contract_data.get("params_schema", {}),
                preconditions=contract_data.get("preconditions", []),
                postconditions=contract_data.get("postconditions", []),
                declares_mutations=contract_data.get("declares_mutations", []),
                deterministic_given_seed=contract_data.get("deterministic_given_seed", True),
                composable_with_self=contract_data.get("composable_with_self", False),
            )
        return cls(
            id=data["id"],
            operation_type=data["operation_type"],
            version=data.get("version", "1.0.0"),
            config=dict(data.get("config", {})),
            contract=contract,
            depends_on=list(data.get("depends_on", [])),
            estimated_parameters_delta=data.get("estimated_parameters_delta", 0),
        )


@dataclass
class ExecutionPlan:
    """A fully-validated, staged transformation plan."""

    recipe_name: str
    target_model: str
    stages: List[Stage]
    warnings: List[str] = field(default_factory=list)
    estimated_parameters_delta: int = 0
    recipe_hash: str = ""
    validation_spec: Dict[str, Any] = field(default_factory=dict)
    execution_spec: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "target_model": self.target_model,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": list(self.warnings),
            "estimated_parameters_delta": self.estimated_parameters_delta,
            "recipe_hash": self.recipe_hash,
            "validation_spec": dict(self.validation_spec),
            "execution_spec": dict(self.execution_spec),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionPlan:
        stages = [Stage.from_dict(s) for s in data.get("stages", [])]
        return cls(
            recipe_name=data.get("recipe_name", ""),
            target_model=data.get("target_model", ""),
            stages=stages,
            warnings=list(data.get("warnings", [])),
            estimated_parameters_delta=data.get("estimated_parameters_delta", 0),
            recipe_hash=data.get("recipe_hash", ""),
            validation_spec=dict(data.get("validation_spec", {})),
            execution_spec=dict(data.get("execution_spec", {})),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = ["ExecutionPlan", "Stage"]
