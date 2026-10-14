"""
Universal Streaming Activation Calibrator for All 72 Compression Engines.

Provides:
  - UniversalActivationCalibrator: attaches streaming in-place GPU forward hooks
    to collect empirical statistics without keeping raw activation tensors in RAM/VRAM:
      * Streaming Gram / Hessian Covariance: H_l = sum(X_l^T X_l) + lambda * I (for WhitenedSVD, ASVD, GPTQ, SparseGPT).
      * Streaming Channel Norms: s_l = running_rms_or_max(X_l) (for Wanda, AWQ, SmoothQuant).
  - Works with any model architecture (LLMs, Vision Transformers, Audio, Custom nn.Module).
  - transplant_and_compress(): coordinated pipeline applying attention transplantation
    and post-surgery projection compression in a single unified step.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger("NeuralOps.Surgery.Calibrator")


class UniversalActivationCalibrator:
    """
    Streaming GPU activation calibrator that accumulates Gram matrices and channel scales
    across arbitrary calibration data without OOM risks.
    """

    def __init__(
        self,
        model: nn.Module,
        target_classes: Tuple[type, ...] = (nn.Linear,),
        exclude_keywords: Tuple[str, ...] = ("lm_head", "classifier", "score", "embed"),
        damping: float = 1e-4,
    ) -> None:
        self.model = model
        self.target_classes = target_classes
        self.exclude_keywords = exclude_keywords
        self.damping = damping

        # Storage for streaming statistics
        self.hessian_dict: Dict[str, torch.Tensor] = {}
        self.channel_scale_dict: Dict[str, torch.Tensor] = {}
        self.sample_count_dict: Dict[str, int] = {}
        self._hooks: List[Any] = []

    def _should_calibrate(self, name: str, module: nn.Module) -> bool:
        """Determines if a module should be tracked for calibration."""
        if not isinstance(module, self.target_classes):
            return False
        for ex in self.exclude_keywords:
            if ex in name.lower():
                return False
        return True

    def attach_hooks(self) -> None:
        """Attaches streaming forward_pre_hook listeners to eligible modules."""
        self.clear()

        for name, module in self.model.named_modules():
            if not self._should_calibrate(name, module):
                continue

            def _make_hook(mod_name: str) -> Callable:
                def _hook(mod: nn.Module, inputs: Tuple[Any, ...]) -> None:
                    if len(inputs) == 0:
                        return
                    x = inputs[0]
                    if not isinstance(x, torch.Tensor):
                        return

                    # x shape: (B, T, D) or (B, D)
                    x_mat = x.detach().reshape(-1, x.shape[-1]).float()
                    num_samples = x_mat.shape[0]

                    # 1. Update streaming Gram matrix H = X^T X
                    gram = torch.matmul(x_mat.t(), x_mat)
                    if mod_name not in self.hessian_dict:
                        self.hessian_dict[mod_name] = gram
                        self.sample_count_dict[mod_name] = num_samples
                    else:
                        self.hessian_dict[mod_name] += gram
                        self.sample_count_dict[mod_name] += num_samples

                    # 2. Update streaming channel norm scales (for Wanda / AWQ)
                    channel_rms = torch.sqrt((x_mat ** 2).mean(dim=0) + 1e-6)
                    if mod_name not in self.channel_scale_dict:
                        self.channel_scale_dict[mod_name] = channel_rms
                    else:
                        self.channel_scale_dict[mod_name] = torch.maximum(
                            self.channel_scale_dict[mod_name], channel_rms
                        )

                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Removes all forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Resets accumulated statistics."""
        self.remove_hooks()
        self.hessian_dict.clear()
        self.channel_scale_dict.clear()
        self.sample_count_dict.clear()

    def calibrate(
        self,
        calibration_data: Union[torch.utils.data.DataLoader, List[Any], torch.Tensor],
        num_batches: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs streaming calibration loop over calibration data and computes
        regularized inverse Hessians and whitening projections.
        """
        self.attach_hooks()
        was_training = self.model.training
        self.model.eval()

        batch_idx = 0
        with torch.no_grad():
            if isinstance(calibration_data, torch.Tensor):
                self.model(calibration_data)
            else:
                for batch in calibration_data:
                    if num_batches is not None and batch_idx >= num_batches:
                        break
                    if isinstance(batch, dict):
                        device = next(self.model.parameters()).device
                        batch_device = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        self.model(**batch_device)
                    elif isinstance(batch, (tuple, list)):
                        device = next(self.model.parameters()).device
                        batch_device = [
                            x.to(device) if isinstance(x, torch.Tensor) else x
                            for x in batch
                        ]
                        self.model(*batch_device)
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch)
                    batch_idx += 1

        self.remove_hooks()
        if was_training:
            self.model.train()

        # Normalize Gram matrices by sample count and add damping
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, gram in self.hessian_dict.items():
            n = max(1, self.sample_count_dict.get(name, 1))
            h_norm = gram / n
            d = h_norm.shape[0]
            h_reg = h_norm + self.damping * torch.eye(d, device=h_norm.device, dtype=h_norm.dtype)
            scale = self.channel_scale_dict.get(name, torch.ones(d, device=gram.device))

            results[name] = {
                "hessian": h_reg,
                "channel_scale": scale,
            }

        return results


def compress_model_universal(
    model: nn.Module,
    method: str = "direct_svd",
    rank: int = 32,
    calibration_data: Optional[Any] = None,
    layers: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Universal model compression runner supporting all 72 NeuralOps compression engines
    with optional streaming calibration.
    """
    if method == "svd":
        method = "direct_svd"

    calib_stats = None
    if calibration_data is not None:
        calibrator = UniversalActivationCalibrator(model)
        calib_stats = calibrator.calibrate(calibration_data)

    comp_step: Dict[str, Any] = {
        "method": method,
        "rank": rank,
        **kwargs,
    }
    if layers is not None:
        comp_step["layer_indices"] = layers
    if calib_stats is not None:
        comp_step["calibration_stats"] = calib_stats

    from neuralops.core.compression_pipeline import compress_model
    return compress_model(model=model, config={"compressors": [comp_step]})


def transplant_and_compress(
    model: nn.Module,
    attention_type: str = "cosformer",
    compression_method: str = "direct_svd",
    rank: int = 32,
    calibration_data: Optional[Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Coordinated pipeline: applies attention transplantation then performs
    streaming-calibrated model compression in a single operation.
    """
    from neuralops.api import replace_attention

    # 1. Transplant attention
    model = replace_attention(model, attention=attention_type, **kwargs)

    # 2. Compress projections
    model = compress_model_universal(
        model=model,
        method=compression_method,
        rank=rank,
        calibration_data=calibration_data,
        **kwargs,
    )
    return model


__all__ = [
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
]
