#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Operation Registry and Lifecycle Management.

Implements Specification Section 5 (Operation Registry Protocol).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import (
    E_REGISTRY_DUPLICATE,
    E_REGISTRY_UNKNOWN_OP,
    E_REGISTRY_UNKNOWN_VERSION,
    OperationError,
)

logger = logging.getLogger("NeuralOps.Operations.Registry")

# Internal storage: {op_name: {version: Operation_instance}}
_OPERATION_REGISTRY: Dict[str, Dict[str, Operation]] = {}
OPERATION_REGISTRY = _OPERATION_REGISTRY


def register_operation(
    operation_or_cls: Optional[Union[Operation, Type[Operation], Callable[..., Any]]] = None,
    *,
    override: bool = False,
) -> Any:
    """Register an operation contract and its executable implementation.

    Can be used as a function call or decorator on Operation classes.
    """

    def _decorator(target: Union[Operation, Type[Operation], Callable[..., Any]]) -> Any:
        instance: Operation
        if isinstance(target, type) and issubclass(target, Operation):
            instance = target()
        elif isinstance(target, Operation):
            instance = target
        else:
            raise TypeError(f"Target must be an Operation instance or subclass, got {type(target).__name__}")

        name = instance.name
        ver = instance.version

        if name not in _OPERATION_REGISTRY:
            _OPERATION_REGISTRY[name] = {}

        if ver in _OPERATION_REGISTRY[name] and not override:
            raise OperationError(
                f"Operation '{name}' at version '{ver}' is already registered",
                E_REGISTRY_DUPLICATE,
                operation_name=name,
            )

        _OPERATION_REGISTRY[name][ver] = instance
        logger.debug(f"Registered operation {name}@{ver}")
        return target

    if operation_or_cls is not None:
        return _decorator(operation_or_cls)
    return _decorator


def get_operation(name: str, version: Optional[str] = "1.0.0") -> Operation:
    """Retrieve an operation instance by name and exact version."""
    if name == "attention.transplant" and "attention.transplant" not in _OPERATION_REGISTRY:
        name = "attention.replace"
    if name not in _OPERATION_REGISTRY:
        raise OperationError(
            f"Operation '{name}' is not registered in the active registry",
            E_REGISTRY_UNKNOWN_OP,
            operation_name=name,
        )

    versions = _OPERATION_REGISTRY[name]
    if version not in versions:
        avail = ", ".join(sorted(versions.keys()))
        raise OperationError(
            f"Operation '{name}' has no registered version '{version}'. Available versions: [{avail}]",
            E_REGISTRY_UNKNOWN_VERSION,
            operation_name=name,
        )

    return versions[version]


def list_operations() -> List[str]:
    """Return sorted list of all registered operations in 'name@version' format."""
    ops: List[str] = []
    for name, v_map in sorted(_OPERATION_REGISTRY.items()):
        for ver in sorted(v_map.keys()):
            ops.append(f"{name}@{ver}")
    return sorted(ops)


def get_operation_contracts() -> Dict[str, OperationContract]:
    """Return a dictionary of all registered contracts keyed by 'name@version'."""
    contracts: Dict[str, OperationContract] = {}
    for name, v_map in _OPERATION_REGISTRY.items():
        for ver, op in v_map.items():
            if op.contract is not None:
                contracts[f"{name}@{ver}"] = op.contract
    return contracts
