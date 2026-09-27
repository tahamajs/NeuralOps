"""
CR (Column-Row / Interpolative Decomposition) Compressor.
Decomposes W ≈ C @ P, where C are actual columns of W.
Registered in COMPRESSOR_REGISTRY as 'cr' and 'cr_compressor'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.model_utils import replace_module
from framework.core.pruning_utils import adaptive_max_error
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from dict, object, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CRLinear(nn.Module):
    """
    Interpolative Decomposition / Column-Row Factorized Linear Layer.
    W ≈ C @ P where:
    - C in R^{out_features x rank}: column skeleton from W
    - P in R^{rank x in_features}: interpolative projection matrix

    Forward pass:
    y = ((x @ P^T) @ C^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.rank = max(1, min(rank, in_features, out_features))

        self.C = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.P = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.C, -bound, bound)
        nn.init.uniform_(self.P, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.C, self.P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution:
        y = ((x @ P^T) @ C^T) + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_p = F.linear(x_flat, self.P)
        out = F.linear(x_p, self.C, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return torch.matmul(self.C, self.P)


    @classmethod
    def from_linear(cls, linear: Union[nn.Linear, Any], rank: int = 64) -> "CRLinear":
        """
        Perform Column Interpolative Decomposition on linear or Conv1D layer weight matrix.
        """
        target_mod = getattr(linear, "base_layer", getattr(linear, "linear", linear))
        raw_w = target_mod.weight.data
        bias = target_mod.bias.data if getattr(target_mod, "bias", None) is not None else None
        cls_name = target_mod.__class__.__name__.lower()
        wrapper_name = linear.__class__.__name__.lower()

        is_conv1d = "conv1d" in cls_name or "conv1d" in wrapper_name or hasattr(target_mod, "nf") or hasattr(linear, "nf")
        if is_conv1d:
            # Conv1D weights are [in_features, out_features] -> transpose to [out_features, in_features]
            W = raw_w.t().contiguous()
        else:
            W = raw_w.contiguous()

        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        cr_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        W_float = W.float()

        # Step 1: Select skeleton columns via SciPy QRCP or robust PyTorch pivoted selection
        col_idx = None
        try:
            import scipy.linalg
            W_np = W_float.cpu().numpy()
            _, _, piv = scipy.linalg.qr(W_np, mode="economic", pivoting=True)
            col_idx = torch.tensor(piv[:r], dtype=torch.long, device=W.device)
        except Exception as e:
            logger.debug("CRLinear pivoted QR failed; using greedy column selection: %s", e)

        if col_idx is None:
            # Pure PyTorch QRCP-equivalent greedy pivoted column selection
            remaining_cols = list(range(in_f))
            selected_cols = []
            norms = torch.sum(W_float ** 2, dim=0)

            for _ in range(r):
                best_col_idx = int(torch.argmax(norms).item())
                selected_cols.append(best_col_idx)
                # Zero out selected norm
                norms[best_col_idx] = -1.0
                if len(selected_cols) >= r:
                    break
                # Orthogonalize remaining columns against selected column
                v = W_float[:, best_col_idx:best_col_idx+1]
                v_norm_sq = torch.sum(v ** 2).clamp_min(1e-12)
                proj = torch.matmul(v.T, W_float) / v_norm_sq
                W_float = W_float - torch.matmul(v, proj)
                norms = torch.sum(W_float ** 2, dim=0)
                for sc in selected_cols:
                    norms[sc] = -1.0

            col_idx = torch.tensor(selected_cols, dtype=torch.long, device=W.device)

        W_float_orig = W.float()
        C_data = W_float_orig[:, col_idx]  # (out_f, r)

        # Step 2: Tikhonov regularized pseudo-inverse: (C^T C + eps * I)^{-1} C^T
        CtC = torch.matmul(C_data.T, C_data)  # (r, r)
        diag_mean = torch.trace(CtC) / max(1, r)
        eps = max(1e-4 * diag_mean.item(), 1e-4)
        reg_eye = eps * torch.eye(r, device=C_data.device, dtype=C_data.dtype)
        try:
            C_pinv = torch.linalg.solve(CtC + reg_eye, C_data.T)  # (r, out_f)
        except Exception:
            C_pinv = torch.linalg.pinv(C_data, rcond=1e-4)

        P_data = torch.matmul(C_pinv, W_float_orig)  # (r, in_f)

        # Step 3: Symmetrically normalize factor magnitudes to ensure FP16 stability
        norm_C = torch.norm(C_data, p="fro").clamp_min(1e-8)
        norm_P = torch.norm(P_data, p="fro").clamp_min(1e-8)
        scale = torch.sqrt(norm_C / norm_P)
        C_data = C_data / scale
        P_data = P_data * scale

        cr_layer.C.data.copy_(C_data.to(dtype=W.dtype))
        cr_layer.P.data.copy_(P_data.to(dtype=W.dtype))

        if bias is not None and cr_layer.bias is not None:
            cr_layer.bias.data.copy_(bias)

        return cr_layer


@COMPRESSOR_REGISTRY.register("cr_compressor")
class CRCompressor(BaseCompressor):
    """
    CR (Column-Row / Interpolative Decomposition) Compressor.
    Decomposes linear layers into skeleton column bases and interpolative projections.

    Args:
        model: PyTorch model.
        rank: Rank parameter (number of skeleton columns).
        max_error: Maximum relative Frobenius reconstruction error (default 0.20).
        fallback_to_original: If True, keep original layer when error exceeds max_error (default True).
        max_retries: Number of rank increase retries before fallback (default 2).
        target_attention_only: If True, restrict compression to attention projections only.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_attention_only: bool = False,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cr", **kwargs)
        c_rank = _cfg_get(cfg, "cr_rank", _cfg_get(cfg, "rank", rank))
        c_max_error = _cfg_get(cfg, "cr_max_error", _cfg_get(cfg, "max_error", max_error))
        c_fallback = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
        c_target_attn = _cfg_get(cfg, "cr_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))

        if c_max_error is None or c_max_error == 0.20:
            c_max_error = adaptive_max_error(c_rank, 768, 768, "distilbert")

        self.rank = int(c_rank) if c_rank is not None else 64
        self.max_error = float(c_max_error)
        self.fallback_to_original = bool(c_fallback)
        self.max_retries = max(0, int(max_retries))
        self.target_attention_only = bool(c_target_attn)
        self.cfg = cfg
        self.target_layers = target_layers or _cfg_get(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Filter candidate layers based on classification head exclusion and target_attention_only."""
        cls_name = module.__class__.__name__.lower()
        has_base = hasattr(module, "base_layer")
        base_cls = module.base_layer.__class__.__name__.lower() if has_base else ""

        is_linear = isinstance(module, nn.Linear) or "linear" in cls_name
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_lora = has_base and (
            isinstance(module.base_layer, nn.Linear)
            or "conv1d" in base_cls
            or hasattr(module.base_layer, "nf")
        )

        if not (is_linear or is_conv1d or is_lora):
            return False

        if isinstance(module, CRLinear) or (has_base and isinstance(module.base_layer, CRLinear)):
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor", "pre_classifier", "head"]):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False

        if self.target_layers and not any(t in name for t in self.target_layers):
            return False

        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using interpolative decomposition."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                if self.target_attention_only:
                    is_lin_like = isinstance(module, nn.Linear) or "conv1d" in module.__class__.__name__.lower() or hasattr(module, "base_layer")
                    if is_lin_like:
                        for p in module.parameters():
                            p.requires_grad = False
                continue

            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            raw_w = target_mod.weight.data
            orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = orig_w.shape
            max_rank = min(out_f, in_f)

            if max_rank < 16:
                continue

            # Parameter reduction check: r * (in_f + out_f) must be less than in_f * out_f
            target_r = min(self.rank, max_rank)
            if target_r * (in_f + out_f) >= 0.85 * (in_f * out_f):
                target_r = max(1, int(0.65 * (in_f * out_f) / (in_f + out_f)))

            cur_r = target_r
            best_cr_layer = None
            best_err = float("inf")
            best_r = cur_r
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)

            for _ in range(self.max_retries + 1):
                try:
                    candidate = CRLinear.from_linear(target_mod, rank=cur_r)
                    recon_w = candidate.get_weight().float()
                    err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()
                    if err < best_err:
                        best_err = err
                        best_cr_layer = candidate
                        best_r = cur_r

                    if err <= self.max_error:
                        break

                    next_r = min(int(cur_r * 1.5), max_rank - 1)
                    if next_r <= cur_r or next_r * (in_f + out_f) >= 0.85 * (in_f * out_f):
                        break
                    cur_r = next_r
                except Exception as ex:
                    logger.debug("CR decomposition error on %s with rank %d: %s", name, cur_r, ex)
                    break

            if best_err > self.max_error:
                logger.warning(
                    "CRCompressor: Layer %s error %.4f exceeded max_error %.4f",
                    name, best_err, self.max_error
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    # Freeze fallback layers during recovery
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if best_cr_layer is not None:
                new_layer = best_cr_layer
                compressed_layer = new_layer
                # Preserve existing LoRA adapter weights if the source module has them
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                    if hasattr(new_layer, 'lora_A') and new_layer.lora_A is not None:
                        if hasattr(src_lA, 'data') and src_lA.shape == new_layer.lora_A.shape:
                            new_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(new_layer, 'lora_B') and new_layer.lora_B is not None:
                        if hasattr(src_lB, 'data') and src_lB.shape == new_layer.lora_B.shape:
                            new_layer.lora_B.data.copy_(src_lB.data)

                if hasattr(module, 'base_layer'):
                    module.base_layer = compressed_layer  # put compressed layer inside wrapper
                    # module itself (LoRALinear wrapper) remains the replacement
                    new_module = module
                else:
                    new_module = compressed_layer

                if hasattr(module, 'base_layer'):
                    for p_name, p in module.named_parameters():
                        if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                            p.requires_grad = True
                else:
                    if name == "":
                        self.model = new_module
                    else:
                        replace_module(self.model, name, new_module)

                self.compressed_layers.append(name)
                self.layer_errors[name] = float(best_err)
                self.layer_ranks[name] = int(best_r)
                total_orig_decomp_params += (in_f * out_f)
                total_comp_decomp_params += (best_r * (in_f + out_f))

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        effective_param_ratio = (
            total_comp_decomp_params / total_orig_decomp_params if total_orig_decomp_params > 0 else 1.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "cr_rank": self.rank,
                "target_attention_only": self.target_attention_only,
                "effective_parameter_ratio": round(effective_param_ratio, 4),
                "used_interpolative_decomposition_count": len(self.compressed_layers),
            }
        )
        logger.info(
            "Compressor cr finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats

    def compress_linear(self, linear: Union[nn.Linear, Any]) -> CRLinear:
        """Compress a single linear layer into CRLinear."""
        return CRLinear.from_linear(linear, rank=self.rank)


@COMPRESSOR_REGISTRY.register("cr")
@COMPRESSOR_REGISTRY.register("cr_factory")
@COMPRESSOR_REGISTRY.register("cr_compressor_factory")
def cr_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CR compressor."""
    effective_rank = kwargs.pop("rank", rank or _cfg_get(comp_config, "cr_rank", _cfg_get(comp_config, "rank", 64)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "cr_max_error", _cfg_get(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "cr_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    compressor = CRCompressor(
        model,
        rank=effective_rank,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats:
        return compressed_model, stats
    return compressed_model
