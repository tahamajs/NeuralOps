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
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        g = torch.sigmoid(self.gate).unsqueeze(1)  # [rank, 1]
        V_gated = self.V * g                       # [rank, in_features]
        latent = F.linear(x_flat, V_gated)         # [..., rank]
        out = F.linear(latent, self.U, self.bias)  # [..., out_features]
        return out.view(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


@COMPRESSOR_REGISTRY.register("dobi")
@COMPRESSOR_REGISTRY.register("dobi_svd")
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
    - Strict rank bounds guaranteeing parameter reduction.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Critical-layer protection via should_compress_linear.
    - Detailed per-layer stats and error reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        init_rank_ratio: float = 0.5,
        rank: Optional[int] = None,
        init_gate_val: float = 2.5,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="dobi", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            init_rank_ratio = getattr(
                cfg, "dobi_init_rank_ratio", getattr(cfg, "init_rank_ratio", init_rank_ratio)
            )
            rank = getattr(cfg, "dobi_rank", getattr(cfg, "rank", rank))
            init_gate_val = getattr(cfg, "init_gate_val", init_gate_val)
            max_error = getattr(cfg, "dobi_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "dobi_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.init_rank_ratio = float(init_rank_ratio)
        self.rank = rank
        self.init_gate_val = float(init_gate_val)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.max_retries = int(max_retries)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_errors: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, DobiSVDLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _determine_rank(self, in_f: int, out_f: int, target_r: Optional[int]) -> int:
        """Compute target rank guaranteeing parameter reduction and physical bounds."""
        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

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
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()      # [out_features, in_features]

        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

        current_r = self._determine_rank(in_f, out_f, rank)
        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()

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

        for attempt in range(self.max_retries + 1):
            r = max(1, min(current_r, max_r))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            # Pre-scale V by 1/sigma(alpha) so initial gated output matches SVD reconstruction
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :] * init_gate_scale).to(dtype=dtype, device=device)

            dobi_layer = DobiSVDLinearLayer(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=layer.bias is not None,
                init_gate_val=self.init_gate_val,
                dtype=dtype,
                device=device,
            )
            dobi_layer.set_low_rank_factors(U_r, V_r)
            if layer.bias is not None and dobi_layer.bias is not None:
                dobi_layer.bias.data.copy_(layer.bias.data)

            W_recon = dobi_layer.weight.float()
            if norm_w < 1e-12:
                err = 0.0
            else:
                err = float(torch.norm(W - W_recon, p="fro").item() / norm_w)

            best_err = err
            chosen_r = r
            best_layer = dobi_layer

            if err <= self.max_error or r >= max_r:
                break

            if attempt < self.max_retries:
                next_r = min(int(r * 1.5), max_r)
                if next_r <= r:
                    break
                current_r = next_r

        return best_layer, best_err, chosen_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all target linear and Conv1D layers across the model."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("Dobi-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
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
                    logger.info("Skipping Dobi-SVD compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            replace_module(self.model, name, new_layer)
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
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
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


def dobi_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Dobi factory function."""
    init_rank_ratio = kwargs.pop("init_rank_ratio", getattr(comp_config, "dobi_init_rank_ratio", getattr(comp_config, "init_rank_ratio", 0.5)) if comp_config else 0.5)
    rank = kwargs.pop("rank", getattr(comp_config, "dobi_rank", getattr(comp_config, "rank", None)) if comp_config else None)
    init_gate_val = kwargs.pop("init_gate_val", getattr(comp_config, "init_gate_val", 2.5) if comp_config else 2.5)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20) if comp_config else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config else True)

    compressor = DobiSVDCompressor(
        model,
        init_rank_ratio=init_rank_ratio,
        rank=rank,
        init_gate_val=init_gate_val,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


dobi_svd_factory = dobi_factory
dobi_svd_compressor_factory = dobi_factory

__all__ = [
    "DobiSVDLinearLayer",
    "DobiSVDCompressor",
    "dobi_factory",
    "dobi_svd_factory",
    "dobi_svd_compressor_factory",
]
