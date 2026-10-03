"""
SLiM: Structured Low-Rank Integrated Matrix Compression (Quantization + 2:4 Sparsity + SVD Residual Correction).
Registered in COMPRESSOR_REGISTRY as 'slim' and 'slim_compressor'.

Mathematical Formulation:
1. Uniform/Non-Uniform b-bit Quantization:
   $$W_q = \\operatorname{Quant}_b(W_0)$$
2. 2:4 Semi-Structured Sparsity:
   $$W_{sp} = \\Pi_{2:4}(W_q, |W_q|)$$
3. Pruning & Quantization Residual:
   $$R = W_0 - W_{sp}$$
4. Low-Rank Singular Value Decomposition Correction:
   $$R \\approx U_r \\Sigma_r V_r^\\top = L_r$$
5. Final Reconstructed Weight:
   $$W_{slim} = W_{sp} + L_r$$

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import apply_24_structured_mask
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("slim")
@COMPRESSOR_REGISTRY.register("slim_compressor")
class SLiMCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    SLiM Compressor: One-shot joint b-bit quantization, 2:4 structured sparsity, and low-rank residual recovery.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        sparsity: float = 0.5,
        rank: int = 8,
        structured_24: bool = True,
        target_attention_only: bool = False,
        max_error: float = 0.5,
        fallback_to_original: bool = True,  # Fix 1: Default changed to True
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="slim", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            bits = _cfg_get(cfg, "slim_bits", _cfg_get(cfg, "bits", bits))
            sparsity = _cfg_get(
                cfg, "slim_sparsity", _cfg_get(cfg, "sparsity", sparsity)
            )
            rank = _cfg_get(cfg, "slim_rank", _cfg_get(cfg, "rank", rank))
            structured_24 = _cfg_get(
                cfg, "slim_structured_24", _cfg_get(cfg, "structured_24", structured_24)
            )
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(
                cfg, "fallback_to_original", fallback_to_original
            )
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.bits = int(bits)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.rank = max(1, int(rank))
        self.structured_24 = structured_24
        self.target_attention_only = bool(target_attention_only)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """
        Fix 4: Unified critical-layer protection delegated to should_compress_linear.
        Eliminates duplicate keyword matching.
        """
        name_l = name.lower()
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
        cls_name = base_mod.__class__.__name__.lower()
        # Fix 2: Reliable Conv1D check using class name and hasattr(nf)
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _quantize(self, W: torch.Tensor) -> torch.Tensor:
        """Quantizes weight to b-bit representation (biases remain unquantized)."""
        if self.bits >= 32:
            return W
        min_val = W.min()
        max_val = W.max()
        if (max_val - min_val).abs() < 1e-6:
            max_val = min_val + 1e-6
        scale = (max_val - min_val) / (2.0**self.bits - 1.0)
        W_q = torch.round((W - min_val) / scale) * scale + min_val
        return W_q

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Executes SLiM compression pipeline across all Linear and Conv1D layers."""
        self.layer_errors = {}
        pruned_layers_count = 0
        total_svd_energy = 0.0

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                base_mod = getattr(module, "base_layer", module)
                raw_w = base_mod.weight.data
                cls_name = base_mod.__class__.__name__.lower()
                # Fix 2: Check for Conv1D via class name and nf attribute
                is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")
                device = raw_w.device
                dtype = raw_w.dtype

                W = (
                    raw_w.t().contiguous().clone().float()
                    if is_conv1d
                    else raw_w.clone().float()
                )

                # 1. Quantize (Weights only; biases are preserved in full precision to avoid drift)
                W_q = self._quantize(W)

                # 2. Apply 2:4 structured sparsity or magnitude sparsity
                if self.structured_24:
                    W_sp, _ = apply_24_structured_mask(
                        W_q, saliency=torch.abs(W_q), dim=1
                    )
                else:
                    flat = W_q.abs().reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    mask = (W_q.abs() >= thr).float()
                    W_sp = W_q * mask

                # 3. Low-Rank Residual Correction
                residual = W - W_sp
                try:
                    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                    r = min(self.rank, len(S))
                    if r > 0:
                        correction = torch.matmul(
                            U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :]
                        )
                        total_svd_energy += torch.norm(correction).item()
                        W_final = W_sp + correction
                    else:
                        W_final = W_sp
                except Exception as e:
                    logger.debug("SVD residual correction exception in %s: %s", name, e)
                    W_final = W_sp

                # Relative Frobenius error check
                norm_orig = torch.norm(W, p="fro")
                rel_err = (torch.norm(W - W_final, p="fro") / (norm_orig + 1e-8)).item()
                self.layer_errors[name] = rel_err

                # Fix 3: Always log a warning when reconstruction error exceeds max_error
                if rel_err > self.max_error:
                    logger.warning(
                        "SLiM reconstruction error %.4f on %s exceeded threshold %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        (
                            ". Skipping (fallback to original)."
                            if self.fallback_to_original
                            else ". Proceeding with compressed weight."
                        ),
                    )
                    if self.fallback_to_original:
                        logger.warning(
                            "Fallback triggered for %s. Layer is kept original and NOT wrapped in LoRA. It will not be fine-tuned.",
                            name,
                        )
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                final_w = W_final.t().contiguous() if is_conv1d else W_final
                base_mod.weight.data = final_w.to(dtype=dtype, device=device)
                compressed_layer = base_mod
                # Preserve existing LoRA adapter weights if the source module has them
                if (
                    hasattr(module, "lora_A")
                    and hasattr(module, "lora_B")
                    and module.lora_A is not None
                    and module.lora_B is not None
                ):
                    src_lA = getattr(module.lora_A, "weight", module.lora_A)
                    src_lB = getattr(module.lora_B, "weight", module.lora_B)
                    if (
                        hasattr(compressed_layer, "lora_A")
                        and compressed_layer.lora_A is not None
                    ):
                        if (
                            hasattr(src_lA, "data")
                            and src_lA.shape == compressed_layer.lora_A.shape
                        ):
                            compressed_layer.lora_A.data.copy_(src_lA.data)
                    if (
                        hasattr(compressed_layer, "lora_B")
                        and compressed_layer.lora_B is not None
                    ):
                        if (
                            hasattr(src_lB, "data")
                            and src_lB.shape == compressed_layer.lora_B.shape
                        ):
                            compressed_layer.lora_B.data.copy_(src_lB.data)

                # Determine if module is wrapped (LoRALinear) or standalone
                is_wrapped = hasattr(module, "base_layer")
                if is_wrapped:
                    module.base_layer = compressed_layer
                    new_module = module
                else:
                    new_module = compressed_layer
                    replace_module(self.model, name, new_module)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        # Fix 6: Store max_error and fallback_to_original in stats
        stats.update(
            {
                "pruned_layers": pruned_layers_count,
                "quant_bits": self.bits,
                "rank": self.rank,
                "structured_24": self.structured_24,
                "total_svd_energy": total_svd_energy,
                "layer_errors": self.layer_errors,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "target_attention_only": self.target_attention_only,
                "biases_quantized": False,  # Fix 5: Explicitly document bias policy
                "method": "slim_quant_sparse_lowrank",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("slim_factory")
def slim_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for SLiM compressor."""
    bits = kwargs.pop(
        "bits", _cfg_get(comp_config, "slim_bits", _cfg_get(comp_config, "bits", 4))
    )
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "slim_sparsity", _cfg_get(comp_config, "sparsity", 0.5)),
    )
    rank = kwargs.pop(
        "rank", _cfg_get(comp_config, "slim_rank", _cfg_get(comp_config, "rank", 8))
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(
            comp_config,
            "slim_structured_24",
            _cfg_get(comp_config, "structured_24", True),
        ),
    )
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )

    compressor = SLiMCompressor(
        model,
        bits=bits,
        sparsity=sparsity,
        rank=rank,
        structured_24=structured_24,
        target_attention_only=target_attention_only,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


slim_compressor_factory = slim_factory

__all__ = [
    "SLiMCompressor",
    "slim_factory",
    "slim_compressor_factory",
]
