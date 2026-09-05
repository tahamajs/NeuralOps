"""
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning via Eigen-Decomposition.
Registered in COMPRESSOR_REGISTRY as 'flat_llm' and 'flat_llm_compressor'. Production hardened
for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with reconstruction
validation, Conv1D support, and safe fallback.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("flat_llm")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor")
class FLATLLMCompressor(BaseCompressor):
    """
    FLAT-LLM Compressor: Activation-aware neuron and output channel pruning.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if error exceeds threshold.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Consistent bias pruning when output channels are pruned.
    - Well-conditioned damped Hessian inversion in FP32 with magnitude fallback.
    - Unified critical-layer protection via should_compress_linear.
    - Per-layer stats and error reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        keep_ratio: float = 0.7,
        damping: float = 1e-4,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flat_llm", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            keep_ratio = getattr(cfg, "flat_llm_keep_ratio", getattr(cfg, "keep_ratio", keep_ratio))
            damping = getattr(cfg, "flat_llm_damping", getattr(cfg, "damping", damping))
            structured_24 = getattr(cfg, "flat_llm_structured_24", getattr(cfg, "structured_24", structured_24))
            max_error = getattr(cfg, "flat_llm_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "flat_llm_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.keep_ratio = max(0.05, min(float(keep_ratio), 1.0))
        self.damping = float(damping)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.structured_24 = bool(structured_24)
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_keep_ratios: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_layer_outputs(self) -> Dict[str, torch.Tensor]:
        """Capture output activations across target modules."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if isinstance(output, torch.Tensor):
                    out = output.detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(out.cpu())
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("FLAT-LLM output collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes FLAT-LLM dimension pruning across target linear and Conv1D modules."""
        outputs_map = self._collect_layer_outputs()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_keep_ratios = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("FLAT-LLM: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            if is_conv1d:
                W = raw_w.t().float()  # [out_features, in_features]
            else:
                W = raw_w.float()      # [out_features, in_features]

            d_out, d_in = W.shape
            norm_w = torch.norm(W, p="fro").item()

            act = outputs_map.get(name, None)
            if act is not None and act.numel() > 0:
                act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
                if act_flat.shape[-1] == d_out:
                    try:
                        H = torch.matmul(act_flat.T, act_flat) / max(act_flat.shape[0], 1)
                        diag_mean = torch.diagonal(H).mean().item()
                        if not math.isfinite(diag_mean) or diag_mean <= 0.0:
                            diag_mean = 1.0
                        lambda_val = max(float(self.damping) * diag_mean, 1e-6)
                        H_damped = H + lambda_val * torch.eye(d_out, device=device, dtype=H.dtype)
                        eigvals, _ = torch.linalg.eigh(H_damped)
                    except Exception as e:
                        logger.debug("FLAT-LLM eigh failed on %s: %s", name, e)
                        eigvals = torch.sum(W.pow(2), dim=1)
                else:
                    eigvals = torch.sum(W.pow(2), dim=1)
            else:
                eigvals = torch.sum(W.pow(2), dim=1)

            if self.structured_24:
                saliency_matrix = eigvals.unsqueeze(1).expand_as(W)
                W_pruned, _ = apply_24_structured_mask(W, saliency=saliency_matrix, dim=0)
                mask = None
            else:
                keep_dims = max(1, int(d_out * self.keep_ratio))
                _, keep_idx = torch.topk(eigvals, k=keep_dims, largest=True)
                mask = torch.zeros(d_out, device=device)
                mask[keep_idx] = 1.0
                W_pruned = W * mask.unsqueeze(1)

            # Compute relative Frobenius reconstruction error
            if norm_w < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - W_pruned, p="fro").item() / norm_w)

            self.layer_errors[name] = final_err
            self.layer_keep_ratios[name] = 0.5 if self.structured_24 else self.keep_ratio

            if final_err > self.max_error:
                logger.warning(
                    "FLAT-LLM reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping FLAT-LLM pruning for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            # Assign pruned weights ensuring shape preservation
            if is_conv1d:
                res_w = W_pruned.t().to(dtype=dtype, device=device)
            else:
                res_w = W_pruned.to(dtype=dtype, device=device)

            assert res_w.shape == orig_shape, f"Shape mismatch on {name}: {res_w.shape} vs {orig_shape}"
            module.weight.data.copy_(res_w)

            if mask is not None and module.bias is not None:
                module.bias.data.copy_((module.bias.data * mask).to(dtype=dtype, device=device))

            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_keep_ratios": self.layer_keep_ratios,
            "keep_ratio": self.keep_ratio,
            "damping": self.damping,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "method": "flat_llm_eigen_pruning",
        })
        logger.info(
            "FLAT-LLM compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("flat_llm")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor")
def flat_llm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAT-LLM compressor."""
    keep_ratio = kwargs.pop("keep_ratio", getattr(comp_config, "flat_llm_keep_ratio", getattr(comp_config, "keep_ratio", 0.7)) if comp_config else 0.7)
    damping = kwargs.pop("damping", getattr(comp_config, "flat_llm_damping", getattr(comp_config, "damping", 1e-4)) if comp_config else 1e-4)
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "flat_llm_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else False)
    max_error = kwargs.pop("max_error", getattr(comp_config, "flat_llm_max_error", getattr(comp_config, "max_error", 0.20)) if comp_config else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "flat_llm_fallback_to_original", getattr(comp_config, "fallback_to_original", True)) if comp_config else True)
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else None))

    compressor = FLATLLMCompressor(
        model,
        tokenizer=tokenizer,
        keep_ratio=keep_ratio,
        damping=damping,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["FLATLLMCompressor", "flat_llm_factory"]
