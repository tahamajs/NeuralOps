"""
SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'soka'.

Mathematical Formulation:
1. Weight Matrix Factorization via SVD:
   $$W = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top = A B^\\top$$
   where $A = U_r \\Sigma_r^{1/2} \\in \\mathbb{R}^{d_{out} \\times r}$ and $B = V_r \\Sigma_r^{1/2} \\in \\mathbb{R}^{d_{in} \\times r}$.
2. Forward Execution:
   $$y = (x B) A^\\top + b$$
3. Singular Energy Threshold Rank Selection:
   $$r = \\min \\left\\{ k \\;\\middle|\\; \\frac{\\sum_{i=1}^k \\sigma_i^2}{\\sum_{i=1}^{\\min(d_{in}, d_{out})} \\sigma_i^2} \\ge \\alpha \\right\\}$$
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LowRankLinear(nn.Module):
    """
    Factorized Linear Layer executing as y = (x @ V) @ U^T + b.
    Reduces parameter storage from (d_in * d_out) to r * (d_in + d_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        # V: [in_features, rank] -> maps x to bottleneck
        # U: [out_features, rank] -> maps bottleneck to output
        self.V = nn.Parameter(torch.empty((in_features, self.rank), device=device, dtype=dtype))
        self.U = nn.Parameter(torch.empty((out_features, self.rank), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.V, std=0.02)
        nn.init.normal_(self.U, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U, self.V.t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        bottleneck = torch.matmul(x, self.V)  # [..., rank]
        out = torch.matmul(bottleneck, self.U.T)  # [..., out_features]
        if self.bias is not None:
            out = out + self.bias
        return out


class LowRankCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    SoKA Low-Rank Linear Compressor.
    Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        energy_threshold: float | None = None,
        max_error: float = 0.5,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="soka", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            rank = _cfg_get(cfg, "soka_rank", _cfg_get(cfg, "rank", rank))
            energy_threshold = _cfg_get(
                cfg, "soka_energy", _cfg_get(cfg, "energy_threshold", energy_threshold)
            )
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)

        self.rank = max(1, int(rank))
        self.energy_threshold = energy_threshold
        self.max_error = float(max_error)
        self.target_attention_only = bool(target_attention_only)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.compressed_layers: list[dict[str, Any]] = []
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = [
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
            ]
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name.lower()
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False
        if isinstance(base_mod, LowRankLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _select_rank(self, S: torch.Tensor, in_f: int, out_f: int) -> int:
        """Determines rank dynamically based on singular energy or fixed target, capped for true reduction."""
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = min(len(S), param_cap)

        if self.energy_threshold is None:
            return max(1, min(self.rank, max_r))
        total_energy = torch.sum(S**2)
        if total_energy <= 0:
            return max(1, min(self.rank, max_r))
        cum_energy = torch.cumsum(S**2, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = idx[0].item() if len(idx) > 0 else len(S)
        return max(1, min(r, max_r))

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Replaces target Linear and Conv1D modules with LowRankLinear."""
        self.compressed_layers = []
        self.layer_errors = {}

        linear_layers: list[tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                linear_layers.append((name, module))

        if not linear_layers:
            logger.warning("SoKA: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in linear_layers:
            base_mod = getattr(module, "base_layer", module)
            raw_w = base_mod.weight.data
            cls_name = base_mod.__class__.__name__
            is_conv1d = (
                "conv1d" in cls_name.lower()
                or hasattr(base_mod, "nf")
                or (
                    raw_w.ndim == 2
                    and hasattr(base_mod, "in_features")
                    and base_mod.in_features == raw_w.shape[0]
                    and raw_w.shape[0] < raw_w.shape[1]
                )
            )

            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
            device = raw_w.device
            dtype = raw_w.dtype
            d_out, d_in = W.shape

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = self._select_rank(S, d_in, d_out)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))

                new_layer = LowRankLinear(
                    in_features=d_in,
                    out_features=d_out,
                    rank=r,
                    bias=(base_mod.bias is not None),
                    device=device,
                    dtype=dtype,
                )

                # Assign factorized weights: W ≈ (U_r sqrt(S)) @ (sqrt(S) V_r^T)
                # U: [d_out, r], V: [d_in, r]
                new_layer.U.data = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                new_layer.V.data = (Vh[:r, :].T * sqrt_S.unsqueeze(0)).to(
                    dtype=dtype, device=device
                )

                if base_mod.bias is not None:
                    new_layer.bias.data.copy_(base_mod.bias.data)

                # Frobenius reconstruction error check
                W_rec = new_layer.weight.float()
                norm_orig = torch.norm(W, p="fro")
                rel_err = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()
                self.layer_errors[name] = rel_err

                if self.fallback_to_original and rel_err > self.max_error:
                    logger.warning(
                        "SoKA reconstruction error %.4f on %s exceeded threshold %.4f. Skipping.",
                        rel_err,
                        name,
                        self.max_error,
                    )
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                compressed_layer = new_layer
                # Preserve existing LoRA adapter weights if the source module has them
                if (
                    hasattr(module, "lora_A")
                    and hasattr(module, "lora_B")
                    and module.lora_A is not None
                    and module.lora_B is not None
                ):
                    src_lA = getattr(module.lora_A, "weight", module.lora_A)
                    src_lB = getattr(module.lora_B, "weight", module.lora_B)
                    if hasattr(new_layer, "lora_A") and new_layer.lora_A is not None:
                        if hasattr(src_lA, "data") and src_lA.shape == new_layer.lora_A.shape:
                            new_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(new_layer, "lora_B") and new_layer.lora_B is not None:
                        if hasattr(src_lB, "data") and src_lB.shape == new_layer.lora_B.shape:
                            new_layer.lora_B.data.copy_(src_lB.data)

                if hasattr(module, "base_layer"):
                    module.base_layer = compressed_layer  # put compressed layer inside wrapper
                    # module itself (LoRALinear wrapper) remains the replacement
                    new_module = module
                else:
                    new_module = compressed_layer
                if not hasattr(module, "base_layer"):
                    if name == "":
                        self.model = new_module
                    else:
                        replace_module(self.model, name, new_module)
                self.compressed_layers.append(
                    {
                        "name": name,
                        "original_shape": (d_out, d_in),
                        "rank": r,
                        "error": rel_err,
                    }
                )
            except Exception as e:
                logger.debug("SVD decomposition error in %s: %s", name, e)
                for p in module.parameters():
                    p.requires_grad = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "total_layers": len(self.compressed_layers),
                "method": "soka_low_rank_svd",
                "fixed_rank": self.rank,
                "energy_threshold": self.energy_threshold,
                "layer_errors": self.layer_errors,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "max_error": self.max_error,
            }
        )
        self.stats = stats
        return self.model, stats


# Aliases for framework naming compatibility
SoKACompressor = LowRankCompressor


@COMPRESSOR_REGISTRY.register("soka")
@COMPRESSOR_REGISTRY.register("soka_compressor")
def soka_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for SoKA compressor."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "soka_rank", _cfg_get(comp_config, "rank", 8)))
    energy = kwargs.pop(
        "energy_threshold",
        _cfg_get(comp_config, "soka_energy", _cfg_get(comp_config, "energy_threshold", None)),
    )
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.5))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )

    compressor = LowRankCompressor(
        model,
        rank=rank,
        energy_threshold=energy,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "LowRankLinear",
    "LowRankCompressor",
    "SoKACompressor",
    "soka_factory",
]
