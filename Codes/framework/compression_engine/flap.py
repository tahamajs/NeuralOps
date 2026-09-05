"""
FLAP: Fluctuation-Based Adaptive Structured Pruning.
Registered in COMPRESSOR_REGISTRY as 'flap' and 'flap_compressor'. Production hardened
for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with
reconstruction validation, Conv1D support, and safe fallback.
"""

from __future__ import annotations

import logging
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


@COMPRESSOR_REGISTRY.register("flap")
@COMPRESSOR_REGISTRY.register("flap_compressor")
class FLAPCompressor(BaseCompressor):
    """
    FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if error exceeds threshold.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - FP32 computation for all variance, saliency, and mask calculations.
    - Per-layer stats and error reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.2,
        metric_type: str = "WIFV",
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flap", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = getattr(cfg, "flap_sparsity", getattr(cfg, "sparsity", sparsity))
            metric_type = getattr(cfg, "flap_metric_type", getattr(cfg, "metric_type", metric_type))
            structured_24 = getattr(cfg, "flap_structured_24", getattr(cfg, "structured_24", structured_24))
            max_error = getattr(cfg, "flap_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "flap_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.metric_type = metric_type.upper()
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
        self.layer_sparsities: Dict[str, float] = {}

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

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture layer input activations across target modules."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach().float()
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                    in_f = module.weight.shape[0] if is_conv1d else module.weight.shape[1]
                    flat = inp.reshape(-1, in_f)
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(flat.cpu())
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
                logger.debug("FLAP activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def _compute_fluctuation_metric(
        self,
        weight: torch.Tensor,
        activations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes per-column fluctuation saliency score in FP32."""
        d_out, d_in = weight.shape
        device = weight.device

        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float().to(device)
            if X.shape[-1] == d_in:
                inp_var = torch.var(X, dim=0).clamp(min=1e-8)
            else:
                inp_var = torch.ones(d_in, device=device)
        else:
            inp_var = torch.ones(d_in, device=device)

        W = weight.float()
        if self.metric_type == "IFV":
            scores = inp_var
        elif self.metric_type == "WIFV":
            scores = inp_var * torch.sum(W.pow(2), dim=0)
        elif self.metric_type == "WIFN":
            scores = torch.sum(torch.abs(W), dim=0) * torch.sqrt(inp_var)
        else:
            scores = inp_var * torch.sum(W.pow(2), dim=0)

        return scores

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes FLAP structured column pruning across target linear and Conv1D modules."""
        activations_map = self._collect_layer_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("FLAP: No compressible layers found.")
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

            act = activations_map.get(name, None)
            scores = self._compute_fluctuation_metric(W, act)

            if self.structured_24:
                saliency_matrix = scores.unsqueeze(0).expand_as(W)
                W_pruned, _ = apply_24_structured_mask(W, saliency=saliency_matrix, dim=1)
            else:
                n_keep = max(1, int(d_in * (1.0 - self.sparsity)))
                _, keep_idx = torch.topk(scores, k=n_keep, largest=True)
                mask = torch.zeros(d_in, device=device)
                mask[keep_idx] = 1.0
                W_pruned = W * mask.unsqueeze(0)

            # Compute relative Frobenius reconstruction error
            if norm_w < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - W_pruned, p="fro").item() / norm_w)

            self.layer_errors[name] = final_err
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity

            if final_err > self.max_error:
                logger.warning(
                    "FLAP reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping FLAP pruning for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            # Assign pruned weights ensuring shape preservation
            if is_conv1d:
                res_w = W_pruned.t().to(dtype=dtype, device=device)
            else:
                res_w = W_pruned.to(dtype=dtype, device=device)

            assert res_w.shape == orig_shape, f"Shape mismatch on {name}: {res_w.shape} vs {orig_shape}"
            module.weight.data.copy_(res_w)
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
            "layer_sparsities": self.layer_sparsities,
            "metric_type": self.metric_type,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "method": "flap_fluctuation_pruning",
        })
        logger.info(
            "FLAP compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("flap")
@COMPRESSOR_REGISTRY.register("flap_compressor")
def flap_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAP compressor."""
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "flap_sparsity", getattr(comp_config, "sparsity", 0.2)) if comp_config else 0.2)
    metric_type = kwargs.pop("metric_type", getattr(comp_config, "flap_metric_type", getattr(comp_config, "metric_type", "WIFV")) if comp_config else "WIFV")
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "flap_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else False)
    max_error = kwargs.pop("max_error", getattr(comp_config, "flap_max_error", getattr(comp_config, "max_error", 0.20)) if comp_config else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "flap_fallback_to_original", getattr(comp_config, "fallback_to_original", True)) if comp_config else True)
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or (getattr(comp_config, "tokenizer", None) if comp_config else None))

    compressor = FLAPCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        metric_type=metric_type,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["FLAPCompressor", "flap_compressor_factory"]
