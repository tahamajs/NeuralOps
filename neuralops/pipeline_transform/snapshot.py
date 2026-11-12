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
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
    """In-memory or disk-backed selective snapshot of model parameters, buffers, and structure."""

    state_paths: Optional[List[str]] = None
    params: Dict[str, torch.Tensor] = field(default_factory=dict)
    buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    submodules: Dict[str, nn.Module] = field(default_factory=dict)
    config_attrs: Dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[Dict[str, Any]] = None
    disk_path: Optional[str] = None
    _temp_dir: Optional[str] = None

    def size_bytes(self) -> int:
        """Calculate total memory size in bytes of captured tensors."""
        if self.disk_path and os.path.exists(self.disk_path):
            return os.path.getsize(self.disk_path)
        total = 0
        for p in self.params.values():
            total += p.numel() * p.element_size()
        for b in self.buffers.values():
            total += b.numel() * b.element_size()
        return total

    def cleanup(self) -> None:
        """Clean up temporary disk artifacts if spilled to disk."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as e:
                logger.debug("Failed cleaning snapshot temp dir %s: %s", self._temp_dir, e)
            self._temp_dir = None
            self.disk_path = None

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        state_paths: Optional[List[str]] = None,
        capture_rng: bool = False,
        spill_to_disk: bool = False,
        disk_dir: Optional[str] = None,
    ) -> Snapshot:
        """Capture a snapshot using the module capture function."""
        return capture(
            model=model,
            state_paths=state_paths,
            capture_rng=capture_rng,
            spill_to_disk=spill_to_disk,
            disk_dir=disk_dir,
        )

    def restore(self, model: nn.Module) -> nn.Module:
        """Restore model using this snapshot."""
        restore(model, self)
        return model


def capture(
    model: nn.Module,
    state_paths: Optional[List[str]] = None,
    capture_rng: bool = False,
    spill_to_disk: bool = False,
    disk_dir: Optional[str] = None,
) -> Snapshot:
    """Capture a selective, memory-efficient snapshot of the model.

    Parameters:
        model: PyTorch module to snapshot.
        state_paths: Dotted path patterns of modules/parameters to capture.
                     If None or ['*'], all model parameters and buffers are captured.
        capture_rng: Whether to capture PyTorch and CUDA RNG states.
        spill_to_disk: If True, writes tensor state dictionaries to disk using SafeTensors
                       to minimize host RAM pressure and enable zero-copy mmap during restore.
        disk_dir: Directory where temporary SafeTensors snapshot files are written.

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
                    except Exception:  # allowed-swallow: ignore unreadable dynamic attributes
                        pass

    # Capture RNG state if requested
    rng_state: Optional[Dict[str, Any]] = None
    if capture_rng:
        rng_state = {
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

    disk_path: Optional[str] = None
    temp_dir_to_track: Optional[str] = None

    if spill_to_disk:
        try:
            from safetensors.torch import save_file

            temp_dir = tempfile.mkdtemp(prefix="neuralops_snap_", dir=disk_dir)
            temp_dir_to_track = temp_dir
            snap_file = os.path.join(temp_dir, f"snapshot_{uuid.uuid4().hex}.safetensors")

            # Combine params and buffers with prefixes for SafeTensors storage
            tensors_to_save: Dict[str, torch.Tensor] = {}
            for k, v in params.items():
                tensors_to_save[f"p:{k}"] = v.contiguous()
            for k, v in buffers.items():
                tensors_to_save[f"b:{k}"] = v.contiguous()

            save_file(tensors_to_save, snap_file)
            disk_path = snap_file
            # Release in-memory tensor dictionaries to free host RAM
            params = {}
            buffers = {}
            logger.info("Spilled snapshot to disk SafeTensors: %s", snap_file)
        except Exception as e:
            logger.warning("Failed to spill snapshot to disk SafeTensors: %s; keeping in RAM", e)
            if temp_dir_to_track and os.path.exists(temp_dir_to_track):
                shutil.rmtree(temp_dir_to_track, ignore_errors=True)
                temp_dir_to_track = None
            disk_path = None

    return Snapshot(
        state_paths=state_paths,
        params=params,
        buffers=buffers,
        submodules=submodules,
        config_attrs=config_attrs,
        rng_state=rng_state,
        disk_path=disk_path,
        _temp_dir=temp_dir_to_track,
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
        if snapshot.disk_path and os.path.exists(snapshot.disk_path):
            from safetensors import safe_open

            with safe_open(snapshot.disk_path, framework="pt") as f:
                for full_key in f.keys():
                    if full_key.startswith("p:"):
                        name = full_key[2:]
                        tensor = f.get_tensor(full_key)
                        param_found = False
                        for p_name, p in model.named_parameters():
                            if p_name == name:
                                p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
                                param_found = True
                                break
                        if not param_found:
                            parent, attr = _get_parent_and_attr(model, name)
                            setattr(parent, attr, nn.Parameter(tensor.to(device=device, dtype=tensor.dtype).clone()))
                    elif full_key.startswith("b:"):
                        name = full_key[2:]
                        tensor = f.get_tensor(full_key)
                        for b_name, b in model.named_buffers():
                            if b_name == name:
                                b.data.copy_(tensor.to(device=b.device, dtype=b.dtype))
                                break
        else:
            for name, tensor in snapshot.params.items():
                param_found = False
                for p_name, p in model.named_parameters():
                    if p_name == name:
                        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
                        param_found = True
                        break

                if not param_found:
                    # Re-attach parameter if removed during mutation
                    parent, attr = _get_parent_and_attr(model, name)
                    setattr(parent, attr, nn.Parameter(tensor.to(device=device, dtype=tensor.dtype).clone()))

            # 3. Restore buffers
            for name, tensor in snapshot.buffers.items():
                for b_name, b in model.named_buffers():
                    if b_name == name:
                        b.data.copy_(tensor.to(device=b.device, dtype=b.dtype))
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


ModelSnapshot = Snapshot

__all__ = ["Snapshot", "ModelSnapshot", "capture", "restore"]
