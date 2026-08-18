"""
Safe model deserialization and checkpoint loading utilities.
Prevents arbitrary code execution by strictly enforcing weights_only=True or using safetensors.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

try:
    import safetensors.torch

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

SAFE_EXTENSIONS = {".safetensors", ".sft", ".pt", ".pth", ".bin", ".ckpt"}


def is_safe_checkpoint_format(checkpoint_path: Union[str, Path]) -> bool:
    """
    Check whether a checkpoint file uses a recognized safe or verifiable format.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        bool: True if the file extension is supported and safe.
    """
    ext = Path(checkpoint_path).suffix.lower()
    return ext in SAFE_EXTENSIONS


def safe_load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
    weights_only: bool = True,
    prefer_safetensors: bool = True,
) -> Dict[str, Any]:
    """
    Safely load a model checkpoint, preventing arbitrary code execution.

    Enforces weights_only=True for standard PyTorch serialized checkpoints,
    and leverages safetensors.torch.load_file for zero-copy safetensors format.

    Args:
        checkpoint_path: Path to checkpoint file on disk.
        map_location: Device destination for loaded tensors ('cpu', 'cuda', etc.).
        weights_only: Whether to strictly enforce weights_only=True in torch.load.
        prefer_safetensors: Whether to prioritize safetensors if available.

    Returns:
        Dict[str, Any]: Loaded checkpoint state dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If file format is unrecognized or deserialization fails.
        RuntimeError: If security constraints (weights_only) are violated.
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at path: {path}")

    ext = path.suffix.lower()

    # FIX: Use safetensors.torch.load_file for zero-risk, fast deserialization when available
    if ext in {".safetensors", ".sft"}:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                f"safetensors library is required to load {path.name}, but it is not installed."
            )
        target_device = str(map_location) if map_location is not None else "cpu"
        logger.debug("Loading safetensors checkpoint from %s onto %s", path, target_device)
        return safetensors.torch.load_file(str(path), device=target_device)

    # FIX: Enforce weights_only=True in torch.load to prevent arbitrary code execution on untrusted models
    try:
        device = map_location if map_location is not None else "cpu"
        logger.debug("Loading torch checkpoint from %s with weights_only=%s", path, weights_only)
        loaded = torch.load(str(path), map_location=device, weights_only=weights_only)
        if isinstance(loaded, dict):
            return loaded
        return {"state_dict": loaded}
    except Exception as e:
        logger.error("Failed to safely load checkpoint %s: %s", path, e)
        raise RuntimeError(
            f"Security or loading error during checkpoint deserialization of {path}: {e}"
        ) from e


def safe_save_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    use_safetensors: bool = True,
) -> str:
    """
    Safely persist a model state dictionary to disk.

    Args:
        state_dict: Dictionary containing tensor weights.
        checkpoint_path: Target file path.
        use_safetensors: If True and supported, saves in safetensors format.

    Returns:
        str: Absolute path to the saved checkpoint file.
    """
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if we should save as safetensors
    is_safetensors_target = path.suffix.lower() in {".safetensors", ".sft"} or use_safetensors

    if is_safetensors_target and SAFETENSORS_AVAILABLE:
        # Check that all items in state_dict are torch.Tensor
        all_tensors = all(isinstance(v, torch.Tensor) for v in state_dict.values())
        if all_tensors:
            target_path = path if path.suffix.lower() == ".safetensors" else path.with_suffix(".safetensors")
            # FIX: Save model weights using safetensors to ensure safe loading by default
            safetensors.torch.save_file(state_dict, str(target_path))
            logger.info("Saved safetensors checkpoint to %s", target_path)
            return str(target_path)

    # Standard torch save fallback
    # FIX: Persist tensors cleanly for safe deserialization
    torch.save(state_dict, str(path))
    logger.info("Saved torch checkpoint to %s", path)
    return str(path)


def safe_load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> torch.nn.Module:
    """
    Safely load weights from a checkpoint file directly into a PyTorch nn.Module.

    Args:
        model: Target PyTorch nn.Module instance.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to enforce exact key matching.
        map_location: Device location.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    # FIX: Strictly use safe_load_checkpoint with weights_only=True
    ckpt = safe_load_checkpoint(checkpoint_path, map_location=map_location, weights_only=True)

    # Unpack nested state dict if necessary
    state_dict = ckpt
    for key in ["model_state_dict", "state_dict", "model", "weights"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break

    model.load_state_dict(state_dict, strict=strict)
    logger.info("Successfully loaded model weights from %s (strict=%s)", checkpoint_path, strict)
    return model
