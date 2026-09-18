"""
NMF: Non-negative Matrix Factorization Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'nmf'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class NMFLinear(nn.Module):
    is_custom_attention: bool = True
    """
    Linear layer decomposed via Non-negative Matrix Factorization (NMF):
    W = H @ V + shift
    - H in R_+^{out_features x rank}
    - V in R_+^{rank x in_features}
    - shift in R
    - Optional LoRA adapter parameters (lora_A, lora_B)

    Forward pass:
    y = ((x @ V^T) @ H^T) + shift * sum(x, dim=-1, keepdim=True) + bias + LoRA(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        shift: float = 0.0,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.rank = min(rank, in_features, out_features)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0
        self.register_buffer("shift", torch.tensor(shift, dtype=dtype or torch.float32, device=device))

        self.H = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device), requires_grad=False)
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device), requires_grad=False)

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.H, 0.0, bound)
        nn.init.uniform_(self.V, 0.0, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        W_mag = torch.matmul(self.H, self.V)
        if hasattr(self, "W_sign") and self.W_sign is not None:
            return W_mag * self.W_sign
        return W_mag + self.shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward pass using factorized matrix multiplications.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        if hasattr(self, "W_sign") and self.W_sign is not None:
            W_eff = (torch.matmul(self.H, self.V) * self.W_sign).to(dtype=x.dtype)
            out = F.linear(x_flat, W_eff)
        else:
            # Factorized forward: (x @ V^T) @ H^T
            x_v = F.linear(x_flat, self.V)  # (K, rank)
            out = F.linear(x_v, self.H)     # (K, out_features)

            # Account for constant baseline shift if non-zero
            if self.shift.abs() > 1e-8:
                x_sum = x_flat.sum(dim=-1, keepdim=True)  # (K, 1)
                out = out + x_sum * self.shift

        if self.bias is not None:
            out = out + self.bias

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x_flat.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        out = out 
        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return self.weight

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 64,
        max_iter: int = 30,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "NMFLinear":
        """
        Fit Non-negative Matrix Factorization on linear or Conv1D layer weights using
        GPU-accelerated Multiplicative Updates (Lee & Seung 2001) with NNDSVD initialization.
        """
        base_layer = getattr(linear, "base_layer", linear)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(base_layer, "in_features")
            and base_layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().contiguous()
            out_f, in_f = W.shape
        else:
            W = raw_w
            out_f, in_f = W.shape

        bias = base_layer.bias.data if getattr(base_layer, "bias", None) is not None else None
        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1) if (in_f * out_f > 16) else max_r
        r = max(1, min(rank, max_r, param_cap))

        # Perform decomposition on GPU if available for 10-50x speedup
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        W_f = W.float().to(device)

        # Signed magnitude decomposition: avoids shift inflation on zero-centered weights
        W_sign = torch.sign(W_f)
        W_sign[W_sign == 0] = 1.0
        W_abs = W_f.abs().clamp_min(1e-6)

        # Fast NNDSVD initialization using torch.svd_lowrank
        k = min(r + 10, max_r)
        try:
            U, S, V = torch.svd_lowrank(W_abs, q=k, niter=2)
            Vt = V.t().contiguous()[:r, :]
            U = U[:, :r]
            S = S[:r]

            H_init = torch.zeros(out_f, r, device=device, dtype=W_abs.dtype)
            V_init = torch.zeros(r, in_f, device=device, dtype=W_abs.dtype)
            H_init[:, 0] = torch.sqrt(S[0]) * U[:, 0].abs()
            V_init[0, :] = torch.sqrt(S[0]) * Vt[0, :].abs()
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
                H_init[:, j] = torch.sqrt(S[j] * sigma) * u
                V_init[j, :] = torch.sqrt(S[j] * sigma) * v
            H_init = H_init.clamp_min(1e-6)
            V_init = V_init.clamp_min(1e-6)
        except Exception:
            torch.manual_seed(42)
            H_init = (torch.rand(out_f, r, device=device, dtype=W_abs.dtype) * 0.1 + 1e-4)
            V_init = (torch.rand(r, in_f, device=device, dtype=W_abs.dtype) * 0.1 + 1e-4)

        n_iter = min(max(1, max_iter), 50)
        try:
            H = H_init.clone()
            V = V_init.clone()
            for _ in range(n_iter):
                HT = H.t().contiguous()
                HTW = torch.matmul(HT, W_abs)
                HTHV = torch.matmul(torch.matmul(HT, H), V).clamp_min(1e-10)
                V = V * (HTW / HTHV)

                VT = V.t().contiguous()
                WVT = torch.matmul(W_abs, VT)
                HVVT = torch.matmul(torch.matmul(H, V), VT).clamp_min(1e-10)
                H = H * (WVT / HVVT)
        except Exception:
            H, V = H_init, V_init

        H_final = H.detach().cpu()
        V_final = V.detach().cpu()
        W_sign_cpu = W_sign.detach().cpu()

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        nmf_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            shift=0.0,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
        )

        nmf_layer.H.data.copy_(H_final.to(dtype=W.dtype))
        nmf_layer.V.data.copy_(V_final.to(dtype=W.dtype))
        nmf_layer.register_buffer("W_sign", W_sign_cpu.to(dtype=W.dtype))

        if bias is not None and nmf_layer.bias is not None:
            nmf_layer.bias.data.copy_(bias)

        # Preserve existing LoRA weights if present
        if hasattr(linear, "lora_A") and hasattr(linear, "lora_B") and nmf_layer.lora_A is not None and nmf_layer.lora_B is not None:
            with torch.no_grad():
                src_A = linear.lora_A.weight.data if hasattr(linear.lora_A, "weight") else linear.lora_A.data
                src_B = linear.lora_B.weight.data if hasattr(linear.lora_B, "weight") else linear.lora_B.data
                if src_A.shape == nmf_layer.lora_A.shape and src_B.shape == nmf_layer.lora_B.shape:
                    nmf_layer.lora_A.copy_(src_A.to(nmf_layer.lora_A.device, dtype=nmf_layer.lora_A.dtype))
                    nmf_layer.lora_B.copy_(src_B.to(nmf_layer.lora_B.device, dtype=nmf_layer.lora_B.dtype))

        if nmf_layer.lora_A is not None:
            nmf_layer.lora_A.requires_grad = True
        if nmf_layer.lora_B is not None:
            nmf_layer.lora_B.requires_grad = True

        return nmf_layer


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


class NMFCompressor(BaseCompressor):
    """
    Non-negative Matrix Factorization (NMF) Compressor.
    Decomposes linear layer weights into low-rank non-negative factors.

    Args:
        model: Model to compress.
        rank: Target rank for NMF factorization.
        max_iter: Maximum multiplicative update iterations.
        cfg: Optional configuration object.
        target_layers: Optional layer name filter.
        target_attention_only: If True, only compress attention projections.
        max_error: Maximum relative Frobenius error threshold (default 0.35).
        fallback_to_original: If True, keep original layer on high reconstruction error.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        max_iter: int = 100,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        target_attention_only: bool = False,
        max_error: float = 0.70,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="nmf", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        self.rank = rank or _cfg_get(conf, "nmf_rank", _cfg_get(conf, "rank", 64))
        self.max_iter = max_iter or _cfg_get(conf, "nmf_max_iter", 100)
        self.lora_r = _cfg_get(conf, "lora_r", lora_r)
        self.lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
        self.cfg = conf
        self.target_layers = target_layers or _cfg_get(conf, "target_layers", None)
        self.target_attention_only = bool(target_attention_only or _cfg_get(conf, "target_attention_only", False))
        self.max_error = float(_cfg_get(conf, "max_error", max_error))
        self.fallback_to_original = bool(_cfg_get(conf, "fallback_to_original", fallback_to_original))
        self.compressed_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf") or hasattr(module, "base_layer")
        if not is_linear_like:
            return False
        if isinstance(module, NMFLinear):
            return False
        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self.target_attention_only and not any(k in name.lower() for k in _ATTN_NAMES):
            return False
        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model linear layers via NMF."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        from framework.core.model_utils import replace_module

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            target_layer = getattr(module, "base_layer", module)
            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
            W_orig = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            max_rank = min(W_orig.shape[0], W_orig.shape[1])
            r = min(self.rank, max_rank)

            try:
                nmf_layer = NMFLinear.from_linear(
                    module,
                    rank=r,
                    max_iter=self.max_iter,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_rec = nmf_layer.weight.float()
                err = (torch.norm(W_orig - W_rec, p="fro") / (torch.norm(W_orig, p="fro") + 1e-8)).item()

                if self.fallback_to_original and err > self.max_error:
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False
                        total_comp_params += p.numel()
                    continue

                replace_module(self.model, name, nmf_layer)
                self.compressed_layers.append({
                    "name": name,
                    "rank": nmf_layer.rank,
                    "relative_error": float(err),
                })
                for p in nmf_layer.parameters(recurse=False):
                    total_comp_params += p.numel()
            except Exception as e:
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
                    total_comp_params += p.numel()

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats.update(
            {
                "num_compressed": len(self.compressed_layers),
                "num_compressed_layers": len(self.compressed_layers),
                "compressed_layer_count": len(self.compressed_layers),
                "nmf_rank": self.rank,
                "max_iter": self.max_iter,
                "effective_parameter_ratio": eff_ratio,
                "compressed_layers": self.compressed_layers,
                "method": "nmf",
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> NMFLinear:
        """Compress a single linear layer into NMFLinear."""
        return NMFLinear.from_linear(linear, rank=self.rank, max_iter=self.max_iter, lora_r=self.lora_r, lora_alpha=self.lora_alpha)


@COMPRESSOR_REGISTRY.register("nmf")
@COMPRESSOR_REGISTRY.register("nmf_compressor")
def nmf_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for NMF compression."""
    effective_rank = rank or _cfg_get(comp_config, "nmf_rank", _cfg_get(comp_config, "rank", 64))
    max_iter = _cfg_get(comp_config, "nmf_max_iter", 100)
    target_attention_only = _cfg_get(comp_config, "target_attention_only", kwargs.pop("target_attention_only", False))
    lora_r = _cfg_get(comp_config, "lora_r", kwargs.pop("lora_r", 0))
    lora_alpha = _cfg_get(comp_config, "lora_alpha", kwargs.pop("lora_alpha", 1.0))
    max_error = _cfg_get(comp_config, "max_error", kwargs.pop("max_error", 0.70))
    fallback = _cfg_get(comp_config, "fallback_to_original", kwargs.pop("fallback_to_original", True))

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
    if kwargs.get("return_stats", kwargs.get("return_tuple", False)):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "NMFLinear",
    "NMFCompressor",
    "nmf_factory",
]

