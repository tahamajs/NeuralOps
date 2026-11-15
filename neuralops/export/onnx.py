#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ONNX Exporter with Dynamic Shape Support for Recurrent & Attention Architectures.

Provides robust ONNX serialization configured for dynamic sequence lengths, batch sizes,
and recurrent hidden state tensors across diverse neural architectures.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Export.ONNX")


class _TensorOnlyExportWrapper(nn.Module):
    """Unwraps HuggingFace ModelOutput dataclass/dict outputs into pure tensor tuples."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.model(*args, **kwargs)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict):
            if "logits" in out:
                return out["logits"]
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
        if isinstance(out, (list, tuple)):
            filtered = [x for x in out if isinstance(x, torch.Tensor)]
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                return tuple(filtered)
        return out


def build_dynamic_axes_for_recurrent(
    input_names: List[str],
    output_names: List[str],
    has_recurrent_state: bool = False,
    state_names: Optional[List[str]] = None,
) -> Dict[str, Dict[int, str]]:
    """Build dynamic axes dictionary supporting dynamic batch size, sequence length, and recurrent states.

    Args:
        input_names: List of input tensor names.
        output_names: List of output tensor names.
        has_recurrent_state: Whether model maintains recurrent hidden states.
        state_names: Names of recurrent state tensors if provided.

    Returns:
        Mapping of tensor names to dimension index to dimension name mappings.
    """
    dynamic_axes: Dict[str, Dict[int, str]] = {}

    for name in input_names:
        if name in ("input_ids", "inputs", "x"):
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "mask" in name:
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif has_recurrent_state and state_names and name in state_names:
            dynamic_axes[name] = {0: "batch_size"}
        else:
            dynamic_axes[name] = {0: "batch_size"}

    for name in output_names:
        if name in ("logits", "output", "hidden_states"):
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif has_recurrent_state and state_names and name in state_names:
            dynamic_axes[name] = {0: "batch_size"}
        else:
            dynamic_axes[name] = {0: "batch_size"}

    return dynamic_axes


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    sample_input: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    verify_with_ort: bool = True,
    is_recurrent: bool = False,
) -> Path:
    """Export PyTorch model to ONNX format with dynamic shape support.

    Args:
        model: PyTorch model module to export.
        output_path: Destination path for .onnx file or directory.
        sample_input: Example input tensor or tuple of tensors for tracing.
        opset_version: Target ONNX opset version (default 17).
        dynamic_axes: Custom dynamic axes mapping. If None, automatically derived.
        input_names: Optional custom list of input node names.
        output_names: Optional custom list of output node names.
        verify_with_ort: Whether to load and verify with onnxruntime if installed.
        is_recurrent: Whether the model exhibits recurrent state dynamics.

    Returns:
        Path to the exported ONNX model file.

    Raises:
        RuntimeError: If export or verification fails.
    """
    out_p = Path(output_path)
    if out_p.is_dir() or not out_p.suffix:
        out_p.mkdir(parents=True, exist_ok=True)
        target_file = out_p / "model.onnx"
    else:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        target_file = out_p

    if sample_input is None:
        sample_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)

    cpu_model = model.to("cpu").eval()
    if isinstance(sample_input, torch.Tensor):
        if sample_input.device != torch.device("cpu"):
            sample_input = sample_input.to("cpu")
    elif isinstance(sample_input, (list, tuple)):
        sample_input = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t for t in sample_input)

    wrapped = _TensorOnlyExportWrapper(cpu_model)

    if input_names is None:
        if isinstance(sample_input, torch.Tensor):
            if sample_input.dtype in (torch.int32, torch.int64):
                input_names = ["input_ids"]
            else:
                input_names = ["inputs"]
        else:
            input_names = [f"input_{i}" for i in range(len(sample_input))]

    if output_names is None:
        output_names = ["logits"]

    if dynamic_axes is None:
        dynamic_axes = build_dynamic_axes_for_recurrent(
            input_names=input_names,
            output_names=output_names,
            has_recurrent_state=is_recurrent,
        )

    effective_opset = opset_version
    is_pt_legacy = torch.__version__ < "2.0"
    if is_pt_legacy and effective_opset > 14:
        effective_opset = 14

    with tempfile.NamedTemporaryFile(
        prefix=".neuralops-onnx-", suffix=".onnx", dir=target_file.parent, delete=False
    ) as temp_f:
        temp_onnx_file = Path(temp_f.name)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            export_kwargs: Dict[str, Any] = {
                "opset_version": effective_opset,
                "input_names": input_names,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
            }
            if not is_pt_legacy:
                export_kwargs["dynamo"] = False
            torch.onnx.export(
                cpu_model,
                sample_input,
                str(temp_onnx_file),
                **export_kwargs,
            )
    except Exception:
        # Fallback with wrapped model if cpu_model direct export failed
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                export_kwargs = {
                    "opset_version": effective_opset,
                    "input_names": input_names,
                    "output_names": output_names,
                    "dynamic_axes": dynamic_axes,
                }
                if not is_pt_legacy:
                    export_kwargs["dynamo"] = False
                torch.onnx.export(
                    wrapped,
                    sample_input,
                    str(temp_onnx_file),
                    **export_kwargs,
                )
        except Exception as exc:
            temp_onnx_file.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX export failed: {exc}") from exc

    try:
        os.replace(temp_onnx_file, target_file)
    except OSError as err:
        temp_onnx_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to publish ONNX artifact to {target_file}: {err}") from err

    if verify_with_ort:
        try:
            import onnxruntime as ort

            _ = ort.InferenceSession(str(target_file))
            logger.info("ONNX artifact %s successfully verified with onnxruntime", target_file)
        except ImportError:
            logger.debug("onnxruntime not installed, skipping session verification")
        except Exception as verif_err:
            raise RuntimeError(f"ONNX export verification failed for {target_file}: {verif_err}") from verif_err

    return target_file


__all__ = [
    "build_dynamic_axes_for_recurrent",
    "export_to_onnx",
]
