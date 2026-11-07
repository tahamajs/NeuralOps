#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SafeTensors serialization and deserialization for surgerized models and checkpoints.

Provides fail-closed, memory-mapped persistence of neural weights with embedded
architectural and provenance metadata (surgery parameters, parent model, git SHA,
and retention metrics).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

LOG = logging.getLogger(__name__)


def save_surgerized_safetensors(
    model: Any,
    path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save surgerized model state_dict or tensor mapping to SafeTensors format.

    Args:
        model: A torch.nn.Module, or a state_dict dictionary mapping tensor names to tensors.
        path: Destination file or directory path (if directory, saves as model.safetensors).
        metadata: Optional provenance dictionary (e.g. base_model, transplanted_attention, git_commit).

    Returns:
        Path to the saved .safetensors file.

    Raises:
        RuntimeError: If safetensors is unavailable or serialization fails.
    """
    target = Path(path)
    if target.is_dir() or target.suffix != ".safetensors":
        if target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "model.safetensors"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    # Prepare stringified metadata for SafeTensors header
    st_metadata: Dict[str, str] = {}
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                st_metadata[str(k)] = str(v)
            else:
                try:
                    st_metadata[str(k)] = json.dumps(v, default=str)
                except Exception:
                    st_metadata[str(k)] = str(v)

    # Extract state dict
    if hasattr(model, "state_dict") and callable(model.state_dict):
        raw_state = model.state_dict()
    elif isinstance(model, dict):
        raw_state = model
    else:
        raise TypeError(f"Expected nn.Module or dict of tensors, got {type(model).__name__}")

    # Ensure tensors are contiguous and on CPU if necessary
    clean_state: Dict[str, Any] = {}
    for k, v in raw_state.items():
        if hasattr(v, "detach"):
            clean_state[k] = v.detach().contiguous()
        else:
            clean_state[k] = v

    # Convert numpy arrays to torch tensors or vice versa based on input
    has_torch = False
    try:
        import torch

        has_torch = True
    except ImportError:
        pass

    # Inspect whether inputs are torch tensors or numpy arrays
    has_numpy_tensors = any(
        hasattr(v, "__array__") and not (has_torch and isinstance(v, torch.Tensor)) for v in clean_state.values()
    )

    if has_numpy_tensors:
        try:
            from safetensors.numpy import save_file as np_save_file

            np_state: Dict[str, Any] = {}
            for k, v in clean_state.items():
                if hasattr(v, "cpu"):
                    np_state[k] = v.cpu().numpy()
                elif hasattr(v, "__array__"):
                    np_state[k] = v
                else:
                    np_state[k] = v
            np_save_file(np_state, str(target), metadata=st_metadata if st_metadata else None)
            return target
        except ImportError:
            pass
        except Exception as exc:
            raise RuntimeError(f"Failed to serialize numpy state to SafeTensors at {target}: {exc}") from exc

    try:
        from safetensors.torch import save_file as st_save_file

        st_save_file(clean_state, str(target), metadata=st_metadata if st_metadata else None)
        return target
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to serialize surgerized model weights to SafeTensors at {target}: {exc}") from exc

    # Fallback to safetensors numpy / rust serialize_file if torch integration module not directly imported
    try:
        from safetensors.numpy import save_file as np_save_file

        # Convert torch tensors to numpy arrays
        np_state: Dict[str, Any] = {}
        for k, v in clean_state.items():
            if hasattr(v, "cpu"):
                np_state[k] = v.cpu().numpy()
            elif hasattr(v, "numpy") and callable(v.numpy):
                np_state[k] = v.numpy()
            else:
                np_state[k] = v
        np_save_file(np_state, str(target), metadata=st_metadata if st_metadata else None)
        return target
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to serialize numpy state to SafeTensors at {target}: {exc}") from exc

    raise RuntimeError(
        f"SafeTensors serialization failed: safetensors package is required to save {target}. "
        "Placeholder files are strictly forbidden by AGENTS.md §6.1."
    )


def load_surgerized_safetensors(
    path: Path | str,
    device: str = "cpu",
    framework: str = "pt",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Load weights and embedded provenance metadata from a SafeTensors file.

    Supports both PyTorch and NumPy frameworks. If framework='pt' is requested but PyTorch
    is not installed, gracefully attempts to return numpy arrays via framework='numpy'.

    Args:
        path: Path to the .safetensors file or directory containing model.safetensors.
        device: Target device for loaded tensors (e.g., 'cpu', 'cuda:0').
        framework: Target tensor framework ('pt' for PyTorch, 'numpy' or 'np' for NumPy).

    Returns:
        Tuple of (state_dict, metadata_dict).

    Raises:
        FileNotFoundError: If the specified safetensors file does not exist.
        RuntimeError: If deserialization fails.
    """
    target = Path(path)
    if target.is_dir():
        target = target / "model.safetensors"

    if not target.exists():
        raise FileNotFoundError(f"SafeTensors checkpoint not found at: {target}")

    metadata: Dict[str, str] = {}
    tensors: Dict[str, Any] = {}

    framework_clean = "np" if framework in ("numpy", "np") else "pt"

    # Attempt primary safe_open loading
    try:
        from safetensors import safe_open

        # Try designated framework
        try:
            with safe_open(
                str(target), framework=framework_clean, device=device if framework_clean == "pt" else "cpu"
            ) as f:
                metadata = f.metadata() or {}
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            return tensors, metadata
        except Exception as primary_exc:
            # If PyTorch was requested but failed (e.g. torch not installed or CPU fallback), try np
            if framework_clean == "pt":
                try:
                    with safe_open(str(target), framework="np") as f:
                        metadata = f.metadata() or {}
                        for k in f.keys():
                            tensors[k] = f.get_tensor(k)
                    return tensors, metadata
                except Exception:
                    pass
            raise primary_exc
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to load SafeTensors file at {target}: {exc}") from exc

    # Direct framework loaders
    if framework_clean == "pt":
        try:
            from safetensors.torch import load_file

            tensors = load_file(str(target), device=device)
            return tensors, metadata
        except Exception:
            pass

    try:
        from safetensors.numpy import load_file as np_load_file

        tensors = np_load_file(str(target))
        return tensors, metadata
    except Exception as exc:
        raise RuntimeError(f"Failed to load SafeTensors file at {target}: {exc}") from exc


__all__ = [
    "save_surgerized_safetensors",
    "load_surgerized_safetensors",
]
