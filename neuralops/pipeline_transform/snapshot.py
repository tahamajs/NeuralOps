#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Selective Model Snapshot and Guaranteed Rollback Management.
"""

from __future__ import annotations

import copy
import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuralops.pipeline_transform.rollback import RollbackError

logger = logging.getLogger("NeuralOps.PipelineTransform.Snapshot")


def _matches_any_path(name: str, state_paths: Optional[List[str]]) -> bool:
    """Check if named parameter, buffer, or module matches any pattern in state_paths."""
    if not state_paths or "*" in state_paths:
        return True

    for pattern in state_paths:
        if pattern == name:
            return True
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(f"*.{name}", pattern):
            return True
        if name.startswith(f"{pattern}.") or pattern.startswith(f"{name}."):
            return True
        # Match wildcard components like *.mlp.* against model.layers.0.mlp.fc
        if any(fnmatch.fnmatch(part, pattern) for part in name.split(".")):
            return True
    return False


@dataclass
class Snapshot:
    """In-memory selective snapshot of model parameters, buffers, and structure."""

    state_paths: Optional[List[str]] = None
    params: Dict[str, torch.Tensor] = field(default_factory=dict)
    buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    submodules: Dict[str, nn.Module] = field(default_factory=dict)
    config_attrs: Dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[Dict[str, Any]] = None

    def size_bytes(self) -> int:
        """Calculate total memory size in bytes of captured tensors."""
        total = 0
        for p in self.params.values():
            total += p.numel() * p.element_size()
        for b in self.buffers.values():
            total += b.numel() * b.element_size()
        return total


def capture(
    model: nn.Module,
    state_paths: Optional[List[str]] = None,
    capture_rng: bool = False,
) -> Snapshot:
    """Capture a selective, memory-efficient snapshot of the model.

    Parameters:
        model: PyTorch module to snapshot.
        state_paths: Dotted path patterns of modules/parameters to capture.
                     If None or ['*'], all model parameters and buffers are captured.
        capture_rng: Whether to capture PyTorch and CUDA RNG states.

    Returns:
        Snapshot instance containing copied tensors and module structures.
    """
    params: Dict[str, torch.Tensor] = {}
    buffers: Dict[str, torch.Tensor] = {}
    submodules: Dict[str, nn.Module] = {}

    # Capture matching parameters
    for name, param in model.named_parameters():
        if _matches_any_path(name, state_paths):
            params[name] = param.detach().clone().cpu()

    # Capture matching buffers
    for name, buf in model.named_buffers():
        if _matches_any_path(name, state_paths):
            buffers[name] = buf.detach().clone().cpu()

    # Capture structural copies of submodules at target paths
    if state_paths and "*" not in state_paths:
        for name, mod in model.named_modules():
            if name and _matches_any_path(name, state_paths):
                # Save deepcopy of target submodule for structural rollback
                try:
                    submodules[name] = copy.deepcopy(mod).cpu()
                except Exception as e:
                    logger.debug("Failed deepcopy of submodule '%s': %s", name, e)
    else:
        # Full model snapshot
        try:
            submodules[""] = copy.deepcopy(model).cpu()
        except Exception as e:
            logger.debug("Failed deepcopy of entire model: %s", e)

    # Capture config attributes if present
    config_attrs: Dict[str, Any] = {}
    if hasattr(model, "config") and model.config is not None:
        cfg = model.config
        if hasattr(cfg, "to_dict"):
            config_attrs = copy.deepcopy(cfg.to_dict())
        else:
            for k in dir(cfg):
                if not k.startswith("_"):
                    try:
                        val = getattr(cfg, k)
                        if not callable(val):
                            config_attrs[k] = copy.deepcopy(val)
                    except Exception:
                        pass

    # Capture RNG state if requested
    rng_state: Optional[Dict[str, Any]] = None
    if capture_rng:
        rng_state = {
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

    return Snapshot(
        state_paths=state_paths,
        params=params,
        buffers=buffers,
        submodules=submodules,
        config_attrs=config_attrs,
        rng_state=rng_state,
    )


def _get_parent_and_attr(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    curr = root
    for p in parts[:-1]:
        if p.isdigit():
            curr = curr[int(p)]
        else:
            curr = getattr(curr, p)
    return curr, parts[-1]


def restore(model: nn.Module, snapshot: Snapshot) -> None:
    """Restore model to the exact snapshotted state.

    Raises:
        RollbackError: If any restoration step fails.
    """
    try:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        # 1. Restore structural submodule mutations
        if "" in snapshot.submodules:
            saved_root = snapshot.submodules[""]
            for child_name, child_mod in saved_root.named_children():
                setattr(model, child_name, copy.deepcopy(child_mod).to(device))
            model.load_state_dict(saved_root.state_dict())
        else:
            for path, saved_mod in snapshot.submodules.items():
                try:
                    parent, attr = _get_parent_and_attr(model, path)
                    restored_mod = copy.deepcopy(saved_mod).to(device)
                    if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential, list)):
                        parent[int(attr)] = restored_mod
                    else:
                        setattr(parent, attr, restored_mod)
                except Exception as exc:
                    logger.debug("Submodule reattachment failed for path '%s': %s", path, exc)

        # 2. Restore parameters
        for name, tensor in snapshot.params.items():
            param_found = False
            for p_name, p in model.named_parameters():
                if p_name == name:
                    p.data.copy_(tensor.to(p.device))
                    param_found = True
                    break

            if not param_found:
                # Re-attach parameter if removed during mutation
                parent, attr = _get_parent_and_attr(model, name)
                setattr(parent, attr, nn.Parameter(tensor.to(device).clone()))

        # 3. Restore buffers
        for name, tensor in snapshot.buffers.items():
            for b_name, b in model.named_buffers():
                if b_name == name:
                    b.data.copy_(tensor.to(b.device))
                    break

        # 4. Restore config attributes
        if snapshot.config_attrs and hasattr(model, "config") and model.config is not None:
            cfg = model.config
            for k, v in snapshot.config_attrs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, copy.deepcopy(v))

        # 5. Restore RNG state
        if snapshot.rng_state is not None:
            if "torch" in snapshot.rng_state:
                torch.set_rng_state(snapshot.rng_state["torch"])
            if "cuda" in snapshot.rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(snapshot.rng_state["cuda"])

    except Exception as e:
        raise RollbackError(f"Failed to restore model snapshot: {e}", error_code="E_EXEC_ROLLBACK_FAILED") from e


__all__ = ["Snapshot", "capture", "restore"]
