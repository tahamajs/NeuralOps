"""
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning via Eigen-Decomposition.
Registered in COMPRESSOR_REGISTRY as 'flat_llm' and 'flat_llm_compressor'.

Bug-fix map:
  FLT-FIX-01  structured_24 reports actual sparsity (no hard-coded 0.5).
  FLT-FIX-02  weight/bias copy under torch.no_grad().
  FLT-FIX-03  bias handling respects requires_grad.
  FLT-FIX-04  multiple calibration batches supported.
  FLT-FIX-05  eigvals fallback documented (weight-energy proxy).
  FLT-FIX-06  forward sanity check includes attention_mask.
  FLT-FIX-07  factory adds return_stats support.
  FLT-FIX-08  no silent mutation of LayerNorm/Embedding.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "out_proj", "dense",
}


@COMPRESSOR_REGISTRY.register("flat_llm")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor")
class FLATLLMCompressor(BaseCompressor):
    """
    FLAT-LLM Compressor: Activation-aware neuron and output channel pruning.

    Pruning metric:
        H = E[x_out^T x_out]          (output covariance across calibration)
        S = diag(H)                    (per-output-channel energy)
        keep = top_k(S, keep_ratio)    (retain highest-energy channels)
        W_pruned = diag(keep) @ W      (zero out pruned rows)

    structured_24 alternative:
        Apply 2:4 sparsity per column with S as the saliency matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        keep_ratio: float = 0.7,
        damping: float = 1e-4,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        structured_24: bool = False,
        target_modules: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        num_calibration_batches: int = 8,       # FLT-FIX-04
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="flat_llm", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.cfg = cfg

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        raw_kr = _cfg_get(cfg, "flat_llm_keep_ratio",
                          _cfg_get(cfg, "keep_ratio", keep_ratio))
        self.keep_ratio = max(0.05, min(float(raw_kr), 1.0))
        self.damping = float(_cfg_get(cfg, "flat_llm_damping",
                                      _cfg_get(cfg, "damping", damping)))
        self.structured_24 = bool(_cfg_get(cfg, "flat_llm_structured_24",
                                           _cfg_get(cfg, "structured_24", structured_24)))
        self.max_error = float(_cfg_get(cfg, "flat_llm_max_error",
                                        _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "flat_llm_target_attention_only",
                     _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "flat_llm_fallback_to_original",
                     _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.calibration_loader = calibration_loader or _cfg_get(cfg, "calibration_loader", None)
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device) if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.num_calibration_batches = max(1, int(
            _cfg_get(cfg, "num_calibration_batches", num_calibration_batches)
        ))

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_keep_ratios: Dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in
               ["classifier", "score", "lm_head", "pooler", "embedding",
                "router", "gate", "norm", "layernorm"]):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    # FLT-FIX-04: multiple calibration batches
    # ------------------------------------------------------------------
    def _normalize_calibration_batches(self, calib_inputs: Any) -> List[Any]:
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, torch.Tensor):
            return [calib_inputs]
        if isinstance(calib_inputs, dict):
            return [calib_inputs]
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            return [calib_inputs]
        return [calib_inputs]

    def _collect_layer_outputs(self) -> Dict[str, torch.Tensor]:
        """
        Capture per-output-channel energy (mean of per-batch output variance).

        Returns: {name: tensor of shape (out_features,)}
        """
        self.model.eval()
        energy_sum: Dict[str, torch.Tensor] = {}
        batch_count: Dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    return
                out = output.detach().float()
                # Determine out_features.
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                out_f = target_mod.weight.shape[1] if is_conv1d else target_mod.weight.shape[0]

                flat = out.reshape(-1, out_f).cpu()
                if flat.size(0) >= 2:
                    var = flat.var(dim=0)
                else:
                    var = flat.abs().mean(dim=0)
                if name not in energy_sum:
                    energy_sum[name] = var
                    batch_count[name] = 1
                else:
                    energy_sum[name] += var
                    batch_count[name] += 1
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )
            batches = self._normalize_calibration_batches(calib_inputs)
            n = min(len(batches), self.num_calibration_batches)

            with torch.no_grad():
                for i in range(n):
                    b = batches[i]
                    try:
                        if isinstance(b, dict):
                            self.model(**b)
                        elif isinstance(b, torch.Tensor):
                            self.model(b)
                        elif isinstance(b, (list, tuple)):
                            self.model(*b)
                    except Exception as e:
                        logger.debug("FLAT-LLM calibration batch %d failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, esum in energy_sum.items():
            aggregated[name] = (esum / max(1, batch_count[name])).to(self.target_device)
        return aggregated

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        outputs_map = self._collect_layer_outputs()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_keep_ratios = {}

        total_orig_params = 0
        total_comp_params = 0

        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("FLAT-LLM: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().float().contiguous() if is_conv1d else raw_w.float()
            d_out, d_in = W.shape
            orig_layer_params = d_out * d_in
            total_orig_params += orig_layer_params
            norm_w = torch.norm(W, p="fro").item()

            # FLT-FIX-05: use output energy if available, else weight-energy proxy.
            act = outputs_map.get(name, None)
            if act is not None and act.numel() > 0 and act.shape[0] == d_out:
                eigvals = act.float().clamp(min=0.0).to(device)
            else:
                eigvals = torch.sum(W.pow(2), dim=1)

            # FLT-FIX-01: compute actual keep ratio from mask / structured output.
            if self.structured_24:
                saliency_matrix = eigvals.unsqueeze(1).expand_as(W)
                W_pruned, mask = apply_24_structured_mask(
                    W, saliency=saliency_matrix, dim=0
                )
                if mask is not None:
                    m = mask.float()
                    effective_keep = m.sum().item() / max(1, m.numel())
                else:
                    zeros = (W_pruned.abs() < 1e-12).float().sum().item()
                    effective_keep = 1.0 - zeros / max(1, W_pruned.numel())
                # For bias gating, derive a channel-level keep mask from nonzero rows.
                with torch.no_grad():
                    row_keep = (W_pruned.abs().sum(dim=1) > 1e-12).float()
            else:
                keep_dims = max(1, int(d_out * self.keep_ratio))
                _, keep_idx = torch.topk(eigvals, k=keep_dims, largest=True)
                row_keep = torch.zeros(d_out, device=device)
                row_keep[keep_idx] = 1.0
                W_pruned = W * row_keep.unsqueeze(1)
                effective_keep = keep_dims / d_out

            if norm_w < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - W_pruned, p="fro").item() / norm_w)

            self.layer_errors[name] = final_err
            self.layer_keep_ratios[name] = float(effective_keep)

            if final_err > self.max_error:
                logger.warning(
                    "FLAT-LLM reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err, name, self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("FLAT-LLM: fallback to original on %s.", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            if is_conv1d:
                res_w = W_pruned.t().to(dtype=dtype, device=device).contiguous()
            else:
                res_w = W_pruned.to(dtype=dtype, device=device).contiguous()

            if res_w.shape != orig_shape:
                raise RuntimeError(
                    f"FLAT-LLM: shape mismatch on {name}: "
                    f"{tuple(res_w.shape)} vs {tuple(orig_shape)}"
                )

            # FLT-FIX-02/03: safe no_grad updates, respects requires_grad.
            with torch.no_grad():
                target_mod.weight.data.copy_(res_w)
                if (row_keep is not None
                        and getattr(target_mod, "bias", None) is not None
                        and target_mod.bias is not None):
                    target_mod.bias.data.copy_(
                        (target_mod.bias.data.float() * row_keep).to(dtype=dtype, device=device)
                    )

            # LoRA trainability.
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if hasattr(module.lora_A, "weight") and module.lora_A.weight is not None:
                    module.lora_A.weight.requires_grad = True
                if hasattr(module.lora_B, "weight") and module.lora_B.weight is not None:
                    module.lora_B.weight.requires_grad = True

            self.compressed_layers.append(name)
            comp_layer_params = max(1, int(effective_keep * orig_layer_params))
            total_comp_params += comp_layer_params

        # FLT-FIX-06: forward sanity check with attention_mask.
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        first_linear = next(
                            (m for m in self.model.modules() if isinstance(m, nn.Linear)),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        self.model(torch.randn(
                            1, 8, in_dim, device=dev,
                            dtype=first_p.dtype if first_p is not None else torch.float32,
                        ))
        except Exception as e:
            logger.debug("FLAT-LLM forward sanity check failed: %s", e)
            forward_check_success = False

        effective_ratio = (total_comp_params / total_orig_params) if total_orig_params > 0 else 1.0

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "layer_errors": self.layer_errors,
            "layer_keep_ratios": self.layer_keep_ratios,
            "keep_ratio": self.keep_ratio,
            "damping": self.damping,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "effective_parameter_ratio": round(effective_ratio, 4),
            "forward_check_success": forward_check_success,
            "method": "flat_llm_eigen_pruning",
        })
        logger.info(
            "FLAT-LLM compression complete: %d compressed, %d fallback. "
            "Avg error: %.4f, Effective Param Ratio: %.2fx",
            len(self.compressed_layers), len(self.fallback_layers),
            avg_err, effective_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("flat_llm_factory")
@COMPRESSOR_REGISTRY.register("flat_llm_compressor_factory")
def flat_llm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for FLAT-LLM compressor."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    keep_ratio = kwargs.pop("keep_ratio",
                 _cfg_get(comp_config, "flat_llm_keep_ratio",
                 _cfg_get(comp_config, "keep_ratio", 0.7)))
    damping = kwargs.pop("damping",
              _cfg_get(comp_config, "flat_llm_damping",
              _cfg_get(comp_config, "damping", 1e-4)))
    structured_24 = kwargs.pop("structured_24",
                     _cfg_get(comp_config, "flat_llm_structured_24",
                     _cfg_get(comp_config, "structured_24", False)))
    max_error = kwargs.pop("max_error",
                _cfg_get(comp_config, "flat_llm_max_error",
                _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "flat_llm_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "flat_llm_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    tokenizer = kwargs.pop("tokenizer",
                getattr(model, "tokenizer", None)
                or _cfg_get(comp_config, "tokenizer", None))
    num_calib = kwargs.pop("num_calibration_batches",
                _cfg_get(comp_config, "num_calibration_batches", 8))

    compressor = FLATLLMCompressor(
        model,
        tokenizer=tokenizer,
        keep_ratio=keep_ratio,
        damping=damping,
        structured_24=structured_24,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        num_calibration_batches=num_calib,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["FLATLLMCompressor", "flat_llm_factory"]