"""
NMF: Non-negative Matrix Factorization Linear Compressor.

Registered in COMPRESSOR_REGISTRY as 'nmf', 'nmf_compressor'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


# ===========================================================================
# NMF Linear Layer
# ===========================================================================
class NMFLinear(nn.Module):
    """
    Linear layer with weight factorized via sign-aware NMF:

        W = H @ V * W_sign

    where H ≥ 0, V ≥ 0, and W_sign ∈ {-1, +1} stores the sign of the original
    weight. Since `torch.sign` cannot be reconstructed by NMF, we factor the
    magnitude |W| with non-negative factors and remember the sign separately.

    Storage: H (out × r), V (r × in), W_sign (out × in as int8 or fp16).
    Note: W_sign storage is non-trivial (out × in elements). If storage is
    the priority, use a signed-NMF variant that avoids storing W_sign.
    """

    # NMF-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # NMF-FIX-02: W_sign always registered (default: all +1).
        # Stored as fp16 to keep memory reasonable; sign is exact in fp16.
        _dtype = dtype if dtype is not None else torch.float32
        self.H = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=_dtype, device=device),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=_dtype, device=device),
            requires_grad=False,
        )
        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.H, 0.0, bound)
        nn.init.uniform_(self.V, 0.0, bound)

        self.register_buffer(
            "W_sign",
            torch.ones(out_features, in_features, dtype=_dtype, device=device),
            persistent=True,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    (self.lora_r, in_features), dtype=torch.float32, device=device
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (out_features, self.lora_r), dtype=torch.float32, device=device
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix: H @ V * W_sign."""
        W_mag = torch.matmul(self.H.float(), self.V.float())
        return (W_mag * self.W_sign.float()).to(self.H.dtype)

    def get_weight(self) -> torch.Tensor:
        return self.weight

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # NMF-FIX-06: dtype-safe
        x_v = F.linear(x_flat, self.V.to(dtype=x_flat.dtype))
        F.linear(x_v, self.H.to(dtype=x_v.dtype))
        # Sign-aware reconstruction: y = x @ W^T where W = H@V * W_sign.
        # (H@V * W_sign) @ x = (H@V) @ (W_sign * x) element-wise? No —
        # we must build the effective weight for F.linear.
        # Alternative: rely on `weight` for correctness.
        W_eff = self.weight.to(dtype=x_flat.dtype)
        out = F.linear(x_flat, W_eff)

        if self.bias is not None:
            out = out + self.bias.to(dtype=x_flat.dtype)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x_flat.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out.view(*orig_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 64,
        max_iter: int = 30,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> NMFLinear:
        """
        Fit sign-aware NMF on the given Linear / Conv1D / LoRALinear module
        using multiplicative updates (Lee & Seung 2001) with NNDSVD init.
        """
        base_layer = getattr(linear, "base_layer", linear)
        raw_w = base_layer.weight.data
        orig_dtype = raw_w.dtype
        orig_device = raw_w.device

        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous()
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
        else:
            W = raw_w
            out_f, in_f = W.shape

        bias = (
            base_layer.bias.data
            if getattr(base_layer, "bias", None) is not None
            else None
        )

        # ---------------- Rank budget (NMF-FIX-08) ----------------
        max_r = min(out_f, in_f)
        # rank*(in+out) < in*out  →  rank < in*out / (in+out)
        param_cap = max(1, (in_f * out_f) // max(1, in_f + out_f))
        r = max(1, min(int(rank), max_r, param_cap))

        # ---------------- Sign-aware factorization ----------------
        # Decompose |W| with NMF; store sign(W) separately.
        W_f = W.float()
        W_sign = torch.sign(W_f)
        W_sign[W_sign == 0] = 1.0  # NMF-FIX-10
        W_abs = W_f.abs().clamp_min(1e-6)

        # NMF-FIX-05: compute on GPU if available, return to orig device.
        compute_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        W_abs_c = W_abs.to(compute_device)

        # ---- NNDSVD init ----
        k = min(r + 10, max_r)
        try:
            U, S, V = torch.svd_lowrank(W_abs_c, q=k, niter=2)
            Vt = V.t().contiguous()[:r, :]
            U = U[:, :r]
            S = S[:r]

            H_init = torch.zeros(out_f, r, device=compute_device, dtype=W_abs_c.dtype)
            V_init = torch.zeros(r, in_f, device=compute_device, dtype=W_abs_c.dtype)
            if r > 0 and S.numel() > 0:
                s0 = S[0].clamp_min(1e-12)
                H_init[:, 0] = (s0.sqrt() * U[:, 0].abs()).clamp_min(1e-6)
                V_init[0, :] = (s0.sqrt() * Vt[0, :].abs()).clamp_min(1e-6)
            for j in range(1, r):
                x, y = U[:, j], Vt[j, :]
                x_p, x_n = x.clamp_min(0), (-x).clamp_min(0)
                y_p, y_n = y.clamp_min(0), (-y).clamp_min(0)
                mp = x_p.norm() * y_p.norm()
                mn = x_n.norm() * y_n.norm()
                if mp >= mn:
                    u = x_p / x_p.norm().clamp_min(1e-10)
                    v = y_p / y_p.norm().clamp_min(1e-10)
                    sigma = mp
                else:
                    u = x_n / x_n.norm().clamp_min(1e-10)
                    v = y_n / y_n.norm().clamp_min(1e-10)
                    sigma = mn
                H_init[:, j] = (
                    S[j].clamp_min(1e-12).sqrt() * sigma.sqrt() * u
                ).clamp_min(1e-6)
                V_init[j, :] = (
                    S[j].clamp_min(1e-12).sqrt() * sigma.sqrt() * v
                ).clamp_min(1e-6)
        except Exception:
            torch.manual_seed(42)
            H_init = torch.rand(out_f, r, device=compute_device) * 0.1 + 1e-4
            V_init = torch.rand(r, in_f, device=compute_device) * 0.1 + 1e-4

        # ---- Multiplicative updates ----
        n_iter = min(max(1, int(max_iter)), 50)
        H = H_init.clone()
        V = V_init.clone()
        try:
            for _ in range(n_iter):
                HT = H.t().contiguous()
                HTW = torch.matmul(HT, W_abs_c)
                HTHV = torch.matmul(torch.matmul(HT, H), V).clamp_min(1e-10)
                V = V * (HTW / HTHV)

                VT = V.t().contiguous()
                WVT = torch.matmul(W_abs_c, VT)
                HVVT = torch.matmul(torch.matmul(H, V), VT).clamp_min(1e-10)
                H = H * (WVT / HVVT)
        except Exception:
            pass  # keep H_init, V_init on failure

        # ---- Build target layer (NMF-FIX-05: return to orig device/dtype) ----
        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(
            linear, "lora_alpha", getattr(linear, "alpha", lora_alpha)
        )

        nmf_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=(bias is not None),
            dtype=orig_dtype,
            device=orig_device,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
        )

        with torch.no_grad():
            nmf_layer.H.data.copy_(H.detach().to(device=orig_device, dtype=orig_dtype))
            nmf_layer.V.data.copy_(V.detach().to(device=orig_device, dtype=orig_dtype))
            nmf_layer.W_sign.data.copy_(W_sign.to(device=orig_device, dtype=orig_dtype))
            if bias is not None and nmf_layer.bias is not None:
                nmf_layer.bias.data.copy_(bias.to(device=orig_device, dtype=orig_dtype))

        # Preserve existing LoRA weights if present
        if (
            hasattr(linear, "lora_A")
            and hasattr(linear, "lora_B")
            and nmf_layer.lora_A is not None
            and nmf_layer.lora_B is not None
        ):
            with torch.no_grad():
                src_A = (
                    linear.lora_A.weight.data
                    if hasattr(linear.lora_A, "weight")
                    else linear.lora_A.data
                )
                src_B = (
                    linear.lora_B.weight.data
                    if hasattr(linear.lora_B, "weight")
                    else linear.lora_B.data
                )
                if (
                    src_A.shape == nmf_layer.lora_A.shape
                    and src_B.shape == nmf_layer.lora_B.shape
                ):
                    nmf_layer.lora_A.copy_(
                        src_A.to(nmf_layer.lora_A.device, dtype=nmf_layer.lora_A.dtype)
                    )
                    nmf_layer.lora_B.copy_(
                        src_B.to(nmf_layer.lora_B.device, dtype=nmf_layer.lora_B.dtype)
                    )
            nmf_layer.lora_A.requires_grad = True
            nmf_layer.lora_B.requires_grad = True

        return nmf_layer


# ===========================================================================
# Config helper
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "self_attn",
}


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("nmf")
@COMPRESSOR_REGISTRY.register("nmf_compressor")
class NMFCompressor(BaseCompressor):
    """
    NMF Compressor: decomposes linear layer magnitudes into non-negative
    low-rank factors, preserving the sign pattern of the original weight.
    """

    # NMF-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 64,
        max_iter: int = 100,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        cfg: Any | None = None,
        target_layers: list[str] | None = None,
        target_attention_only: bool = False,
        max_error: float = 0.70,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(), name="nmf", **kwargs
        )
        conf = cfg or kwargs.get("comp_config")

        self.rank = int(rank or _cfg_get(conf, "nmf_rank", _cfg_get(conf, "rank", 64)))
        self.max_iter = int(max_iter or _cfg_get(conf, "nmf_max_iter", 100))
        self.lora_r = int(_cfg_get(conf, "lora_r", lora_r))
        self.lora_alpha = float(_cfg_get(conf, "lora_alpha", lora_alpha))
        self.cfg = conf
        self.target_layers = target_layers or _cfg_get(conf, "target_layers", None)
        if self.target_layers is not None and len(self.target_layers) == 0:
            self.target_layers = None
        self.target_attention_only = bool(
            target_attention_only or _cfg_get(conf, "target_attention_only", False)
        )
        self.max_error = float(_cfg_get(conf, "max_error", max_error))
        self.fallback_to_original = bool(
            _cfg_get(conf, "fallback_to_original", fallback_to_original)
        )
        self.compressed_layers: list[dict[str, Any]] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(
            k in name_l
            for k in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "layernorm",
                "rmsnorm",
                "batchnorm",
            )
        ):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding, NMFLinear)):
            return False

        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        try:
            from framework.utils.compat import should_compress_linear

            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_attention_only and not any(k in name_l for k in _ATTN_NAMES):
            return False

        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    # ------------------------------------------------------------------
    def compress(
        self, model: nn.Module | None = None
    ) -> tuple[nn.Module, dict[str, Any]]:
        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        from framework.core.model_utils import replace_module

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []

        # NMF-FIX-07: accurate accounting.
        total_orig_storage = 0
        total_comp_storage = 0
        seen_weights = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_module = getattr(module, "base_layer", module)
            w = getattr(base_module, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = w.data
            orig_device = raw_w.device
            orig_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(base_module, "bias", None) is not None:
                orig_bytes += base_module.bias.numel() * base_module.bias.element_size()
            total_orig_storage += orig_bytes

            # rank budget
            if raw_w.ndim == 2:
                cls_name = base_module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(base_module, "nf")
                W_shape = (raw_w.shape[1], raw_w.shape[0]) if is_conv1d else raw_w.shape
            else:
                W_shape = raw_w.shape
            max_rank = min(W_shape)
            r = min(self.rank, max_rank)

            try:
                nmf_layer = NMFLinear.from_linear(
                    module,
                    rank=r,
                    max_iter=self.max_iter,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )

                # NMF-FIX-09: error on the exact forward path.
                W_rec = nmf_layer.weight.float().to(orig_device)
                W_orig = raw_w.float().to(orig_device)
                if is_conv1d:
                    W_orig = W_orig.t().contiguous()
                    W_rec = (
                        W_rec.t().contiguous() if W_rec.shape != W_orig.shape else W_rec
                    )

                num = torch.norm(W_orig - W_rec, p="fro")
                den = torch.norm(W_orig, p="fro").clamp_min(1e-12)
                err = float((num / den).item())

                if self.fallback_to_original and err > self.max_error:
                    logger.warning(
                        "NMF: err=%.4f on %s > max_error=%.4f; fallback.",
                        err,
                        name,
                        self.max_error,
                    )
                    self.fallback_layers.append(name)
                    total_comp_storage += orig_bytes
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                # NMF-FIX-04: preserve wrapper.
                if hasattr(module, "base_layer"):
                    module.base_layer = nmf_layer
                else:
                    replace_module(self.model, name, nmf_layer)

                self.compressed_layers.append(
                    {
                        "name": name,
                        "rank": nmf_layer.rank,
                        "relative_error": float(err),
                    }
                )

                # Storage: H + V + W_sign + bias + LoRA
                comp_bytes = (
                    nmf_layer.H.numel() * nmf_layer.H.element_size()
                    + nmf_layer.V.numel() * nmf_layer.V.element_size()
                    + nmf_layer.W_sign.numel() * nmf_layer.W_sign.element_size()
                )
                if nmf_layer.bias is not None:
                    comp_bytes += nmf_layer.bias.numel() * nmf_layer.bias.element_size()
                if nmf_layer.lora_A is not None:
                    comp_bytes += (
                        nmf_layer.lora_A.numel() * nmf_layer.lora_A.element_size()
                    )
                if nmf_layer.lora_B is not None:
                    comp_bytes += (
                        nmf_layer.lora_B.numel() * nmf_layer.lora_B.element_size()
                    )
                total_comp_storage += comp_bytes

            except Exception as e:
                logger.warning("NMF: failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                total_comp_storage += orig_bytes
                for p in module.parameters():
                    p.requires_grad = False

        # Forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    self.model(dummy)
        except Exception as e:
            logger.warning("NMF forward sanity check failed: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        storage_ratio = (
            total_orig_storage / total_comp_storage if total_comp_storage > 0 else 1.0
        )
        effective_ratio = (
            total_comp_storage / total_orig_storage if total_orig_storage > 0 else 1.0
        )

        stats = self.get_compression_stats()
        stats.update(
            {
                "num_compressed": len(self.compressed_layers),
                "num_compressed_layers": len(self.compressed_layers),
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "nmf_rank": self.rank,
                "max_iter": self.max_iter,
                "compressed_layers": self.compressed_layers,
                "orig_storage_bytes": total_orig_storage,
                "comp_storage_bytes": total_comp_storage,
                "storage_compression_ratio": round(storage_ratio, 3),
                "effective_parameter_ratio": round(effective_ratio, 4),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
                "forward_check_success": forward_check_success,
                "method": "nmf",
            }
        )
        self.stats = stats
        return self.model, stats

    # ------------------------------------------------------------------
    def compress_linear(self, linear: nn.Module) -> NMFLinear:
        return NMFLinear.from_linear(
            linear,
            rank=self.rank,
            max_iter=self.max_iter,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("nmf_factory")
def nmf_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    effective_rank = rank or _cfg_get(
        comp_config, "nmf_rank", _cfg_get(comp_config, "rank", 64)
    )
    max_iter = _cfg_get(comp_config, "nmf_max_iter", 100)
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.70))
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "fallback_to_original", True),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = NMFCompressor(
        model,
        rank=effective_rank,
        max_iter=max_iter,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "NMFLinear",
    "NMFCompressor",
    "nmf_factory",
]
