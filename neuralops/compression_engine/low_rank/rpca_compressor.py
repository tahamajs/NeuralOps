"""
Robust PCA: decompose W = L + S, factorizing low-rank component L with structural SVDLinear.
Registered in COMPRESSOR_REGISTRY as 'rpca', 'rpca_factory', and 'rpca_compressor_factory'.
"""

from __future__ import annotations
from neuralops.core.common import _cfg_get

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class RPCALinear(nn.Module):
    """
    Robust PCA factorized linear layer:
        y = (x @ V.T) @ U.T + (x @ S.T) + bias
    where L = U @ V captures low-rank principal components and S captures sparse outliers.
    Leverages triton_fused_lowrank_sparse_gemm for zero intermediate HBM materialization.
    """
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        use_triton: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.use_triton = bool(use_triton)

        self.V = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))
        self.register_buffer("S_sparse", torch.zeros(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def set_factors(self, U: torch.Tensor, V: torch.Tensor, S: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)
            if S is not None:
                self.S_sparse.copy_(S)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.U.float(), self.V.float()) + self.S_sparse.float()
        return w.to(self.U.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        if getattr(self, "use_triton", True) and x.is_cuda and not self.training:
            try:
                from neuralops.core.triton_kernels.low_rank import triton_fused_lowrank_sparse_gemm
                out = triton_fused_lowrank_sparse_gemm(x_flat, self.V, self.U, s=self.S_sparse, bias=self.bias)
                return out.view(*orig_shape[:-1], self.out_features)
            except Exception as e:
                logger.debug("Triton RPCALinear failed (%s); fallback to PyTorch", e)

        y_lr = F.linear(F.linear(x_flat, self.V), self.U)
        y_sp = F.linear(x_flat, self.S_sparse)
        out = y_lr + y_sp
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)


@COMPRESSOR_REGISTRY.register("rpca_compressor")
class RPCACompressor(BaseCompressor):
    """RPCACompressor neural operator.

    Robust PCA: decompose W = L + S, factorizing low-rank component L with structural SVDLinear. Registered in COMPRESSOR_REGISTRY as 'rpca', 'rpca_factory', and 'rpca_compressor_factory'.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """
    is_custom_attention: bool = True
    """
    Robust Principal Component Analysis (RPCA) Low-Rank Compressor.

    Decomposes weight matrices W into low-rank component L and sparse outlier matrix S
    via Principal Component Pursuit / alternating soft-thresholding:
        min ||L||_* + lambda ||S||_1  s.t.  W = L + S
    Factorizes the low-rank component L into symmetric factors (U * sqrt(Sigma), sqrt(Sigma) * V^T)
    and replaces nn.Linear or Conv1D with factorized SVDLinear modules with trainable LoRA fine-tuning adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        mu: float = 1e-3,
        max_iter: int = 5,
        min_rank: int = 8,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_modules: list[str] | None = None,
        max_error: float = 1.0,
        fallback_to_original: bool = False,
        min_numel: int = 16,
        device: str | torch.device | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="rpca", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config")

        self.rank = int(rank or _cfg_get(cfg_obj, "rpca_rank", _cfg_get(cfg_obj, "rank", 64)))
        self.mu = float(mu or _cfg_get(cfg_obj, "rpca_mu", _cfg_get(cfg_obj, "mu", 1e-3)))
        self.max_iter = int(
            max_iter or _cfg_get(cfg_obj, "rpca_max_iter", _cfg_get(cfg_obj, "max_iter", 5))
        )
        self.min_rank = max(1, int(min_rank or _cfg_get(cfg_obj, "min_rank", 8)))
        self.skip_attention_qkv = bool(
            skip_attention_qkv
            if skip_attention_qkv is not None
            else _cfg_get(cfg_obj, "skip_attention_qkv", False)
        )
        self.target_attention_only = bool(
            target_attention_only
            if target_attention_only is not None
            else _cfg_get(cfg_obj, "target_attention_only", False)
        )
        self.use_lora = bool(use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0))
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.use_triton = bool(kwargs.get("use_triton", _cfg_get(cfg_obj, "use_triton", True)))
        self.target_modules = target_modules
        self.max_error = float(max_error or _cfg_get(cfg_obj, "max_error", 1.0))
        self.fallback_to_original = bool(
            fallback_to_original
            if fallback_to_original is not None
            else _cfg_get(cfg_obj, "fallback_to_original", False)
        )
        self.min_numel = max(1, int(min_numel or _cfg_get(cfg_obj, "min_numel", 16)))

        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = (
                    next(model.parameters()).device
                    if list(model.parameters())
                    else torch.device("cpu")
                )
        else:
            self.target_device = (
                next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            )

        self.visited_modules: set[int] = set()
        self.compressed_layers: list[str] = []
        self.protected_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.failed_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.replaced_layers: list[dict[str, Any]] = []

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer belongs to an attention projection in a GQA architecture."""
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        is_gqa = num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads

        name_l = name.lower()
        is_attn_qkv = any(
            p in name_l
            for p in [
                "q_proj",
                "k_proj",
                "v_proj",
                "c_attn",
                "query",
                "key",
                "value",
                "q_lin",
                "k_lin",
                "v_lin",
                "self_attn.q",
                "self_attn.k",
                "self_attn.v",
            ]
        )
        return (is_gqa or self.skip_attention_qkv) and is_attn_qkv

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(
            k in name_l
            for k in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "embedding",
                "cls_head",
                "head",
                "logits_processor",
            ]
        ):
            self.protected_layers.append(name)
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

        if self._is_gqa_asymmetric(name, module):
            self.skipped_layers.append(name)
            return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        raw_w = getattr(base_mod, "weight", None)
        if raw_w is None or not isinstance(raw_w, torch.Tensor):
            return False

        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
        is_linear_like = isinstance(base_mod, nn.Linear) or is_conv1d
        if not is_linear_like or isinstance(base_mod, SVDLinear):
            return False

        if raw_w.numel() < self.min_numel:
            self.skipped_layers.append(name)
            return False

        if not should_compress_linear(self.model, name):
            self.protected_layers.append(name)
            return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
    ) -> SVDLinear | None:
        """Decompose a single linear layer via RPCA and return factorized SVDLinear with LoRA."""
        base_layer = getattr(layer, "base_layer", layer)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_layer, "nf")
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        if is_conv1d:
            W = raw_w.t().contiguous().to(device=device, dtype=torch.float32)
            out_f, in_f = W.shape
        else:
            W = raw_w.contiguous().to(device=device, dtype=torch.float32)
            out_f = getattr(base_layer, "out_features", W.shape[0])
            in_f = getattr(base_layer, "in_features", W.shape[1])

        max_r = min(out_f, in_f)
        if self.rank is not None:
            r = min(int(self.rank), max_r)
        else:
            param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
            r = min(rank, max_r - 1, param_cap)
            r = max(min(self.min_rank, max_r - 1), r)
            if r <= 0 or r >= max_r:
                logger.debug(
                    "RPCA: rank %d >= max_r %d; skipping layer for no parameter reduction",
                    r,
                    max_r,
                )
                return None

        L = W.clone()
        S_sparse = torch.zeros_like(W)

        try:
            for _ in range(self.max_iter):
                U, S_vals, Vt = torch.linalg.svd(L, full_matrices=False)
                sqrt_S = torch.sqrt(torch.clamp(S_vals[:r], min=1e-12))
                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
                L_recon = U_r @ V_r

                resid = W - L_recon
                S_sparse = torch.sign(resid) * torch.clamp(torch.abs(resid) - self.mu, min=0.0)

                L = W - S_sparse

            U, S_vals, Vt = torch.linalg.svd(L, full_matrices=False)
        except Exception as e_svd:
            logger.warning(
                "RPCA decomposition failed: %s; falling back to standard SVD on W.",
                e_svd,
            )
            try:
                U, S_vals, Vt = torch.linalg.svd(W, full_matrices=False)
            except Exception as e_fb:
                logger.warning("RPCA standard SVD fallback failed: %s", e_fb)
                return None

        sqrt_S = torch.sqrt(torch.clamp(S_vals[:r], min=1e-12))
        U_final = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_final = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        has_existing_lora = hasattr(layer, "lora_A") and hasattr(layer, "lora_B")
        effective_lora_r = self.lora_r
        if has_existing_lora and getattr(layer, "r", 0) > 0:
            effective_lora_r = getattr(layer, "r", self.lora_r)

        W_rec = torch.matmul(U_final.float(), V_final.float())
        W_dev = W.to(device=W_rec.device)
        norm_orig = torch.norm(W_dev, p="fro").clamp_min(1e-8)
        rel_err = (torch.norm(W_dev - W_rec, p="fro") / norm_orig).item()

        is_invalid = torch.isnan(W_rec).any() or torch.isinf(W_rec).any()
        if is_invalid:
            logger.warning("RPCA: reconstruction contains NaN/Inf, falling back.")
            return None
        if rel_err > self.max_error:
            logger.warning(
                "RPCA: reconstruction error %.4f exceeded threshold %.4f. Compressing anyway.",
                rel_err,
                self.max_error,
            )
            if self.fallback_to_original:
                return None

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=base_layer.bias is not None,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r if (self.use_lora or has_existing_lora) else 0,
            lora_alpha=self.lora_alpha,
        )
        svd_layer.set_low_rank_factors(U_final, V_final)
        if base_layer.bias is not None:
            svd_layer.bias.data.copy_(base_layer.bias.data)

        if has_existing_lora and svd_layer.use_lora:
            try:
                if hasattr(layer.lora_A, "weight"):
                    svd_layer.lora_A.weight.data.copy_(layer.lora_A.weight.data)
                if hasattr(layer.lora_B, "weight"):
                    svd_layer.lora_B.weight.data.copy_(layer.lora_B.weight.data)
            except Exception as e_lora:
                logger.warning("Could not copy existing LoRA adapters: %s", e_lora)

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress all linear layers in the model using RPCA decomposition."""
        self.layer_errors = {}
        self.replaced_layers = []
        self.visited_modules.clear()
        self.compressed_layers.clear()
        self.protected_layers.clear()
        self.skipped_layers.clear()
        self.fallback_layers.clear()
        self.failed_layers.clear()

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("RPCA: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            mod_id = id(module)
            if mod_id in self.visited_modules:
                logger.debug("RPCA: Skipping tied module %s", name)
                self.skipped_layers.append(name)
                continue
            self.visited_modules.add(mod_id)

            try:
                svd_layer = self.compress_linear(module, rank=self.rank)
                if svd_layer is not None:
                    err = getattr(svd_layer, "reconstruction_error", 0.0)
                    self.layer_errors[name] = err
                    self.replaced_layers.append(
                        {"name": name, "rank": svd_layer.rank, "error": err}
                    )
                    replace_module(self.model, name, svd_layer)
                    self.compressed_layers.append(name)
                else:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
            except Exception as e_comp:
                logger.warning("Failed to compress %s: %s", name, e_comp)
                self.failed_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        forward_check_success = True
        try:
            device = next(self.model.parameters()).device
            with torch.no_grad():
                cfg = getattr(self.model, "config", None)
                if cfg is not None and getattr(cfg, "vocab_size", None) is not None:
                    dummy_input = torch.randint(0, min(cfg.vocab_size, 100), (1, 8), device=device)
                    self.model(dummy_input)
                else:
                    first_p = next(self.model.parameters())
                    dim = first_p.shape[-1] if first_p.dim() >= 2 else 64
                    dummy_input = torch.randn(1, dim, device=device, dtype=first_p.dtype)
                    self.model(dummy_input)
        except Exception as ex_sanity:
            logger.debug("RPCA post-compression forward sanity check: %s", ex_sanity)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        )
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "protected_layers": self.protected_layers,
                "protected_layer_count": len(self.protected_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "failed_layers": self.failed_layers,
                "failed_layer_count": len(self.failed_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "forward_check_success": forward_check_success,
                "rank": self.rank,
                "min_rank": self.min_rank,
                "mu": self.mu,
                "max_iter": self.max_iter,
                "skip_attention_qkv": self.skip_attention_qkv,
                "target_attention_only": self.target_attention_only,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "method": "rpca",
            }
        )
        self.stats = stats
        return self.model, stats


def rpca_compressor_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """RPCA factory function returning (compressed_model, stats) when requested."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rpca_rank", _cfg_get(comp_config, "rank", 64)))
    mu = kwargs.pop("mu", _cfg_get(comp_config, "rpca_mu", _cfg_get(comp_config, "mu", 1e-3)))
    max_iter = kwargs.pop(
        "max_iter",
        _cfg_get(comp_config, "rpca_max_iter", _cfg_get(comp_config, "max_iter", 5)),
    )
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "min_rank", 8))
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", False)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 1.0))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", False)
    )

    compressor = RPCACompressor(
        model,
        rank=rank,
        mu=mu,
        max_iter=max_iter,
        min_rank=min_rank,
        skip_attention_qkv=skip_attention_qkv,
        target_attention_only=target_attention_only,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


rpca_factory = rpca_compressor_factory

__all__ = [
    "RPCACompressor",
    "rpca_compressor_factory",
    "rpca_factory",
]
