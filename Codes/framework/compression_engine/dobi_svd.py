"""
Dobi-SVD: Differentiable SVD layer with learnable spectral gating.
Registered as 'dobi' and 'dobi_svd'. Production hardened for multi-architecture
compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support,
and safe fallback.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class DobiSVDLinearLayer(nn.Module):
    is_custom_attention: bool = True
    """
    Differentiable SVD Linear Layer (Dobi-SVD).

    Decomposes linear transformations into low-rank factorized parameters
    (U: [out_features, rank], V: [rank, in_features]) with a learnable continuous
    spectral gate vector alpha: [rank], allowing differentiable rank pruning and fine-tuning:
        y = (x @ (V * sigmoid(alpha)).T) @ U.T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        init_gate_val: float = 2.5,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        # Learnable spectral gate parameter initialized to +2.5 so sigma(alpha) ~ 0.92
        self.gate = nn.Parameter(torch.full((self.rank,), fill_value=init_gate_val, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        """Initialize U and V parameters directly from SVD factors."""
        with torch.no_grad():
            self.U.copy_(U_factor)
            self.V.copy_(V_factor)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        g = torch.sigmoid(self.gate).unsqueeze(1)
        return torch.matmul(self.U, self.V * g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass with spectral gating matching Linear and Conv1D."""
        if x.shape[-1] != self.in_features:
            if x.shape[-1] < self.in_features:
                x = F.pad(x, (0, self.in_features - x.shape[-1]))
            else:
                x = x[..., :self.in_features]
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        g = torch.sigmoid(self.gate).unsqueeze(1)  # [rank, 1]
        V_gated = self.V * g                       # [rank, in_features]
        latent = F.linear(x_flat, V_gated)         # [..., rank]
        out = F.linear(latent, self.U, self.bias)  # [..., out_features]
        return out.view(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "q_lin", "k_lin", "v_lin", "out_lin",
    "c_attn", "c_proj",
    "query", "key", "value",
}


@COMPRESSOR_REGISTRY.register("dobi_class")
@COMPRESSOR_REGISTRY.register("dobi_svd_class")
@COMPRESSOR_REGISTRY.register("dobi_compressor_class")
class DobiSVDCompressor(BaseCompressor):
    """
    Dobi-SVD Compressor: Differentiable SVD with learnable spectral gating.

    Extracts truncated SVD factors from pretrained weights with symmetric
    square-root singular value distribution (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T),
    compensates for initial gate value to ensure near-lossless initialization, and
    replaces layers with DobiSVDLinearLayer containing learnable gating parameters.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic rank escalation retries when error exceeds threshold.
    - Safe fallback to original module if error remains high.
    - Fallback and non-target layers frozen (requires_grad=False) during recovery.
    - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
    - target_attention_only parameter: restricts SVD factorization to attention projections.
    - Strict rank bounds guaranteeing parameter reduction.
    - Critical-layer protection via should_compress_linear.
    - Detailed per-layer stats, avg error, and effective parameter ratio.
    """

    def __init__(
        self,
        model: nn.Module,
        init_rank_ratio: float = 0.5,
        rank: Optional[int] = None,
        init_gate_val: float = 2.5,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="dobi", **kwargs)
        cfg_obj = cfg if cfg is not None else kwargs.get("comp_config", None)
        self.cfg = cfg_obj

        init_rank_ratio = _cfg_get(cfg_obj, "dobi_init_rank_ratio", _cfg_get(cfg_obj, "init_rank_ratio", init_rank_ratio))
        rank = _cfg_get(cfg_obj, "dobi_rank", _cfg_get(cfg_obj, "rank", rank))
        init_gate_val = _cfg_get(cfg_obj, "init_gate_val", init_gate_val)
        max_error = _cfg_get(cfg_obj, "dobi_max_error", _cfg_get(cfg_obj, "max_error", max_error))
        fallback_to_original = _cfg_get(
            cfg_obj, "dobi_fallback_to_original", _cfg_get(cfg_obj, "fallback_to_original", fallback_to_original)
        )
        target_attention_only = _cfg_get(
            cfg_obj, "dobi_target_attention_only", _cfg_get(cfg_obj, "target_attention_only", target_attention_only)
        )

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.init_rank_ratio = float(init_rank_ratio)
        self.rank = rank
        self.init_gate_val = float(init_gate_val)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.max_retries = int(max_retries)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_errors: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(k in name_lower for k in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, DobiSVDLinearLayer) or isinstance(module, DobiSVDLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _determine_rank(self, in_f: int, out_f: int, target_r: Optional[int]) -> int:
        """Compute target rank guaranteeing parameter reduction and physical bounds."""
        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r, 128))

        if target_r is not None:
            r = target_r
        else:
            r = int(round(min(in_f, out_f) * self.init_rank_ratio))

        return max(1, min(r, max_r))

    def compress_linear(
        self,
        layer: nn.Module,
        rank: Optional[int] = None,
    ) -> Tuple[Optional[DobiSVDLinearLayer], float, int]:
        """Decompose a single linear or Conv1D layer into DobiSVDLinearLayer."""
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = target_mod.__class__.__name__.lower()
        raw_w = target_mod.weight.data

        # Determine in_features and out_features accurately for Linear, Conv1D, and fused QKV
        in_f = None
        out_f = None
        if target_mod.bias is not None and getattr(target_mod.bias, "ndim", 0) == 1 and raw_w.ndim == 2:
            bias_len = target_mod.bias.shape[0]
            if raw_w.shape[1] == bias_len and raw_w.shape[0] != bias_len:
                # Conv1D format: [in_features, out_features]
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            elif raw_w.shape[0] == bias_len and raw_w.shape[1] != bias_len:
                # Linear format: [out_features, in_features]
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

        if in_f is None or out_f is None:
            if hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
                in_f, out_f = target_mod.in_features, target_mod.out_features
            elif hasattr(target_mod, "nf") and hasattr(target_mod, "nx"):
                in_f, out_f = target_mod.nx, target_mod.nf
            elif "conv1d" in cls_name or (raw_w.ndim == 2 and raw_w.shape[0] < raw_w.shape[1]):
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

        # W must strictly be [out_features, in_features]
        if raw_w.shape == (out_f, in_f):
            W = raw_w.float()
        elif raw_w.shape == (in_f, out_f):
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()

        assert W.shape == (out_f, in_f), f"W shape {W.shape} does not match ({out_f}, {in_f})"

        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r, 128))

        current_r = self._determine_rank(in_f, out_f, rank)
        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()
        if norm_w < 1e-4:
            logger.warning("DobiSVD: weight tensor has near-zero Frobenius norm (%.6e < 1e-4)", norm_w)
            return None, 1.0, 0

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("DobiSVD SVD computation failed: %s", e)
            return None, 1.0, 0

        best_layer: Optional[DobiSVDLinearLayer] = None
        best_err = 1.0
        chosen_r = current_r

        # Initial gate scaling factor to achieve exact initial SVD energy
        init_gate_scale = 1.0 / (1.0 / (1.0 + math.exp(-self.init_gate_val)))

        retries = 0 if rank is not None else self.max_retries
        for attempt in range(retries + 1):
            r = max(1, min(current_r, max_r))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            # Pre-scale V by 1/sigma(alpha) so initial gated output matches SVD reconstruction
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :] * init_gate_scale).to(dtype=dtype, device=device)

            dobi_layer = DobiSVDLinearLayer(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=target_mod.bias is not None,
                init_gate_val=self.init_gate_val,
                dtype=dtype,
                device=device,
            )
            dobi_layer.set_low_rank_factors(U_r, V_r)
            assert dobi_layer.U.shape == (out_f, r), f"U shape mismatch: {dobi_layer.U.shape} vs ({out_f}, {r})"
            assert dobi_layer.V.shape == (r, in_f), f"V shape mismatch: {dobi_layer.V.shape} vs ({r}, {in_f})"

            if target_mod.bias is not None and dobi_layer.bias is not None:
                tb = target_mod.bias.data
                db = dobi_layer.bias.data
                min_b = min(tb.numel(), db.numel())
                db[:min_b].copy_(tb[:min_b])

            W_recon = dobi_layer.weight.float()
            recon_norm = torch.norm(W_recon, p="fro").item()
            if recon_norm < 1e-4:
                logger.warning("DobiSVD: reconstructed weight Frobenius norm too small (%.6e < 1e-4)", recon_norm)
                continue

            if norm_w < 1e-12:
                err = 0.0
            else:
                err = float(torch.norm(W - W_recon, p="fro").item() / norm_w)

            best_err = err
            chosen_r = r
            best_layer = dobi_layer

            if err <= self.max_error or r >= max_r:
                break

            if attempt < retries:
                next_r = min(int(r * 1.5), max_r)
                if next_r <= r:
                    break
                current_r = next_r

        if best_layer is not None:
            best_layer.reconstruction_error = best_err
            return best_layer, best_err, chosen_r
        return None, best_err, chosen_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all target linear and Conv1D layers across the model."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}

        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("Dobi-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        # If target_attention_only=True, freeze non-target linear/conv layers during recovery
        if self.target_attention_only:
            for name, module in list(self.model.named_modules()):
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if is_linear or is_conv1d:
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            raw_w = target_mod.weight.data
            cls_name = target_mod.__class__.__name__.lower()
            in_f = None
            out_f = None
            if target_mod.bias is not None and getattr(target_mod.bias, "ndim", 0) == 1 and raw_w.ndim == 2:
                bias_len = target_mod.bias.shape[0]
                if raw_w.shape[1] == bias_len and raw_w.shape[0] != bias_len:
                    in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                elif raw_w.shape[0] == bias_len and raw_w.shape[1] != bias_len:
                    out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            if in_f is None or out_f is None:
                if hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
                    in_f, out_f = target_mod.in_features, target_mod.out_features
                elif hasattr(target_mod, "nf") and hasattr(target_mod, "nx"):
                    in_f, out_f = target_mod.nx, target_mod.nf
                elif "conv1d" in cls_name or (raw_w.ndim == 2 and raw_w.shape[0] < raw_w.shape[1]):
                    in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                else:
                    out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            orig_params = out_f * in_f
            total_orig_decomp_params += orig_params

            new_layer, err, r = self.compress_linear(module, rank=self.rank)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if new_layer is None or err > self.max_error:
                logger.warning(
                    "Dobi-SVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping Dobi-SVD compression for %s (fallback to original and frozen).", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_decomp_params += orig_params
                    continue

            # If wrapped in LoRALinear, replace the inner base_layer or linear
            has_base_layer = hasattr(module, "base_layer") or (hasattr(module, "linear") and (hasattr(module, "lora_A") or hasattr(module, "r")))
            if hasattr(module, "base_layer"):
                module.base_layer = new_layer
            elif hasattr(module, "linear") and (hasattr(module, "lora_A") or hasattr(module, "r")):
                module.linear = new_layer
            else:
                replace_module(self.model, name, new_layer)

            if has_base_layer:
                for p_name, p in module.named_parameters():
                    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                        p.requires_grad = True

            self.compressed_layers.append(name)
            # Parameters: U (out*r) + V (r*in) + gate (r)
            comp_params = r * (in_f + out_f) + r
            total_comp_decomp_params += comp_params

        effective_ratio = (
            total_comp_decomp_params / total_orig_decomp_params if total_orig_decomp_params > 0 else 1.0
        )
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "original_layer_count": len(target_layers),
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "init_rank_ratio": self.init_rank_ratio,
            "init_gate_val": self.init_gate_val,
            "effective_parameter_ratio": round(effective_ratio, 4),
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        logger.info(
            "Dobi-SVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("dobi")
@COMPRESSOR_REGISTRY.register("dobi_svd")
@COMPRESSOR_REGISTRY.register("dobi_compressor")
@COMPRESSOR_REGISTRY.register("dobi_svd_compressor")
@COMPRESSOR_REGISTRY.register("dobi_factory")
@COMPRESSOR_REGISTRY.register("dobi_svd_factory")
def dobi_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Dobi factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    try:
        init_rank_ratio = kwargs.pop("init_rank_ratio", _cfg_get(comp_config, "dobi_init_rank_ratio", _cfg_get(comp_config, "init_rank_ratio", 0.5)))
        rank = kwargs.pop("rank", _cfg_get(comp_config, "dobi_rank", _cfg_get(comp_config, "rank", None)))
        init_gate_val = kwargs.pop("init_gate_val", _cfg_get(comp_config, "init_gate_val", 2.5))
        max_error = kwargs.pop("max_error", _cfg_get(comp_config, "dobi_max_error", _cfg_get(comp_config, "max_error", 0.35)))
        fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "dobi_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)))
        target_attention_only = kwargs.pop(
            "target_attention_only",
            _cfg_get(comp_config, "dobi_target_attention_only", _cfg_get(comp_config, "target_attention_only", True)),
        )

        compressor = DobiSVDCompressor(
            model,
            init_rank_ratio=init_rank_ratio,
            rank=rank,
            init_gate_val=init_gate_val,
            max_error=max_error,
            fallback_to_original=fallback,
            target_attention_only=target_attention_only,
            cfg=comp_config,
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        if return_stats:
            return compressed_model, stats
        return compressed_model
    except Exception as e:
        logger.error("dobi_factory failed: %s", e, exc_info=True)
        return (model, {}) if return_stats else model


@COMPRESSOR_REGISTRY.register("inject_dobi")
@COMPRESSOR_REGISTRY.register("dobi_inject")
def inject_dobi(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for Dobi-SVD. Returns None on failure."""
    target = layer if layer is not None else model
    if target is None:
        return None
    try:
        res = dobi_factory(target, **kwargs)
        if isinstance(res, tuple):
            return res[0]
        return res
    except Exception as e:
        logger.error("inject_dobi failed: %s", e, exc_info=True)
        return None


def dobi_inject(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook alias for dobi."""
    return inject_dobi(model, layer, **kwargs)


dobi_svd_factory = dobi_factory
dobi_svd_compressor_factory = dobi_factory

__all__ = [
    "DobiSVDLinearLayer",
    "DobiSVDCompressor",
    "dobi_factory",
    "dobi_svd_factory",
    "dobi_svd_compressor_factory",
    "inject_dobi",
    "dobi_inject",
]
