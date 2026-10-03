"""
ROAST: Hardware-Aware Weight Sharing with Tiling.
Reference: 'ROAST: A Hardware-Aware Weight Sharing Scheme for Efficient Neural Networks'
Registered in COMPRESSOR_REGISTRY as 'roast', 'roast_compressor', and 'roast_factory'.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Helper to safely retrieve a configuration value from dict, object, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ROASTLinear(nn.Module):
    """
    Linear layer with weight sharing based on tiles (blocks) rather than
    individual scalar weights. Each tile of size (tile_size, tile_size) shares
    a single weight from a small table, with optional trainable LoRA adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_buckets: int = 1024,
        tile_size: int = 32,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = max(1, int(tile_size))
        self.n_buckets = max(1, int(n_buckets))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # Number of tiles along each dimension (padded)
        self.tiles_out = (out_features + self.tile_size - 1) // self.tile_size
        self.tiles_in = (in_features + self.tile_size - 1) // self.tile_size

        # Weight table: frozen if LoRA is active
        requires_grad = not (self.lora_r > 0)
        self.weight_table = nn.Parameter(
            torch.randn(self.n_buckets, device=device, dtype=dtype),
            requires_grad=requires_grad,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype),
                requires_grad=requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        # Precompute tile indices with high-entropy bitwise mixing (B11, J1)
        tile_idx_i = (
            torch.arange(self.tiles_out, device=device)
            .unsqueeze(1)
            .expand(-1, self.tiles_in)
        )
        tile_idx_j = (
            torch.arange(self.tiles_in, device=device)
            .unsqueeze(0)
            .expand(self.tiles_out, -1)
        )
        h = (tile_idx_i.long() * 0x9E3779B1) ^ (tile_idx_j.long() * 0x85EBCA77)
        h = h ^ (h >> 16)
        h = h * 0x7FEB352D
        h = (h ^ (h >> 15)) & 0x7FFFFFFF
        self.register_buffer(
            "tile_hash_idx",
            (h % self.n_buckets).long(),
            persistent=False,
        )

        self._cached_weight: torch.Tensor | None = None
        self._cache_stale: bool = True

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

    def _invalidate_cache(self) -> None:
        self._cached_weight = None
        self._cache_stale = True

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features) with evaluation caching."""
        if (
            self._cached_weight is not None
            and not self._cache_stale
            and not self.training
        ):
            return self._cached_weight
        W_tiles = self.weight_table[self.tile_hash_idx]  # (tiles_out, tiles_in)
        W_full = torch.repeat_interleave(
            torch.repeat_interleave(W_tiles, self.tile_size, dim=0),
            self.tile_size,
            dim=1,
        )
        W_res = W_full[: self.out_features, : self.in_features]
        if not self.training:
            self._cached_weight = W_res
            self._cache_stale = False
        return W_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized forward pass without per-element python loops."""
        W = self.weight.to(device=x.device, dtype=x.dtype)
        out = F.linear(
            x, W, self.bias.to(dtype=x.dtype) if self.bias is not None else None
        )
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (
                F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
            )
            out = out + lora_out.to(out.dtype)
        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        n_buckets: int = 1024,
        tile_size: int = 32,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> ROASTLinear:
        """Initialize from a pretrained nn.Linear, Conv1D, or LoRALinear base_layer."""
        base_mod = getattr(linear, "base_layer", linear)
        raw_w = base_mod.weight.data
        cls_name = base_mod.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            out_f, in_f = W.shape
        else:
            W = raw_w.contiguous().float()
            out_f, in_f = W.shape

        bias = getattr(base_mod, "bias", None) is not None
        roast = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            tile_size=tile_size,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=raw_w.device,
            dtype=raw_w.dtype,
        )
        if bias and base_mod.bias is not None:
            roast.bias.data.copy_(base_mod.bias.data)

        # Preserve existing source LoRA weights if present
        if (
            hasattr(linear, "lora_A")
            and hasattr(linear, "lora_B")
            and linear.lora_A is not None
            and linear.lora_B is not None
        ):
            src_A = getattr(linear.lora_A, "weight", linear.lora_A)
            src_B = getattr(linear.lora_B, "weight", linear.lora_B)
            if roast.lora_A is not None and roast.lora_B is not None:
                if src_A.shape == roast.lora_A.shape:
                    roast.lora_A.data.copy_(src_A.data)
                if src_B.shape == roast.lora_B.shape:
                    roast.lora_B.data.copy_(src_B.data)

        ts = max(1, tile_size)
        tiles_out = (out_f + ts - 1) // ts
        tiles_in = (in_f + ts - 1) // ts

        # Vectorized computation of per-tile means and scatter into weight_table
        with torch.no_grad():
            pad_out = tiles_out * ts - out_f
            pad_in = tiles_in * ts - in_f
            if pad_out > 0 or pad_in > 0:
                W_padded = F.pad(W, (0, pad_in, 0, pad_out), mode="constant", value=0.0)
                mask = F.pad(
                    torch.ones_like(W),
                    (0, pad_in, 0, pad_out),
                    mode="constant",
                    value=0.0,
                )
                W_blocks = (
                    W_padded.view(tiles_out, ts, tiles_in, ts)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(tiles_out, tiles_in, -1)
                )
                mask_blocks = (
                    mask.view(tiles_out, ts, tiles_in, ts)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(tiles_out, tiles_in, -1)
                )
                tile_means = W_blocks.sum(dim=-1) / mask_blocks.sum(dim=-1).clamp(
                    min=1.0
                )
            else:
                tile_means = (
                    W.view(tiles_out, ts, tiles_in, ts)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(tiles_out, tiles_in, -1)
                    .mean(dim=-1)
                )

            flat_idx = roast.tile_hash_idx.view(-1).to(device=roast.weight_table.device)
            flat_means = tile_means.view(-1).to(
                device=roast.weight_table.device, dtype=roast.weight_table.dtype
            )
            roast.weight_table.data.scatter_(0, flat_idx, flat_means)

        return roast


@COMPRESSOR_REGISTRY.register("roast")
@COMPRESSOR_REGISTRY.register("roast_compressor")
@COMPRESSOR_REGISTRY.register("roast_hashed_linear")
class ROASTCompressor(BaseCompressor):
    """
    Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weight sharing).
    """

    def __init__(
        self,
        model: nn.Module,
        n_buckets: int = 1024,
        tile_size: int = 32,
        skip_attention_qkv: bool = True,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_modules: list[str] | None = None,
        target_attention_only: bool = False,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        min_numel: int = 16,
        device: str | torch.device | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="roast", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config")

        self.n_buckets = max(
            1,
            int(
                n_buckets
                if n_buckets is not None
                else _cfg_get(cfg_obj, "n_buckets", 1024)
            ),
        )
        self.tile_size = max(
            1,
            int(
                tile_size
                if tile_size is not None
                else _cfg_get(cfg_obj, "tile_size", 32)
            ),
        )
        self.target_attention_only = bool(
            target_attention_only
            or _cfg_get(
                cfg_obj,
                "target_attention_only",
                kwargs.get("target_attention_only", False),
            )
        )
        self.skip_attention_qkv = (
            bool(
                skip_attention_qkv
                if skip_attention_qkv is not None
                else _cfg_get(cfg_obj, "skip_attention_qkv", True)
            )
            and not self.target_attention_only
        )
        self.use_lora = bool(
            use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0)
        )
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.target_modules = target_modules
        self.max_error = float(
            max_error if max_error is not None else _cfg_get(cfg_obj, "max_error", 0.50)
        )
        self.fallback_to_original = bool(
            fallback_to_original
            if fallback_to_original is not None
            else _cfg_get(cfg_obj, "fallback_to_original", True)
        )
        self.min_numel = max(
            1,
            int(
                min_numel
                if min_numel is not None
                else _cfg_get(cfg_obj, "min_numel", 16)
            ),
        )

        # Target device derivation
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
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )

        # Diagnostics & Tracking
        self.visited_modules: set[int] = set()
        self.compressed_layers: list[str] = []
        self.protected_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.failed_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer belongs to an attention projection in a GQA architecture."""
        if self.target_attention_only:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        is_gqa = (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        )

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
                "dense",
            ]
            if not any(k in name_l for k in attn_keywords):
                self.skipped_layers.append(name)
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
        if not is_linear_like or isinstance(module, ROASTLinear):
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

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress the target modules using ROAST tiling with LoRA support."""
        self.compressed_layers.clear()
        self.visited_modules.clear()
        self.protected_layers.clear()
        self.skipped_layers.clear()
        self.fallback_layers.clear()
        self.failed_layers.clear()
        self.layer_errors.clear()

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                mod_id = id(module)
                if mod_id in self.visited_modules:
                    logger.debug("ROAST: Skipping tied module %s", name)
                    self.skipped_layers.append(name)
                    continue
                self.visited_modules.add(mod_id)

                try:
                    base_mod = getattr(module, "base_layer", module)
                    raw_w = base_mod.weight.data
                    cls_name = base_mod.__class__.__name__
                    is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
                    W_orig = (
                        raw_w.t().contiguous().float()
                        if is_conv1d
                        else raw_w.contiguous().float()
                    )

                    new_layer = ROASTLinear.from_linear(
                        module,
                        n_buckets=self.n_buckets,
                        tile_size=self.tile_size,
                        lora_r=self.lora_r if self.use_lora else 0,
                        lora_alpha=self.lora_alpha,
                    )
                    W_rec = new_layer.weight.float()
                    norm_orig = torch.norm(W_orig, p="fro").clamp_min(1e-8)
                    err = (torch.norm(W_orig - W_rec, p="fro") / norm_orig).item()
                    self.layer_errors[name] = float(err)

                    is_invalid = torch.isnan(W_rec).any() or torch.isinf(W_rec).any()
                    if err > self.max_error:
                        logger.warning(
                            "ROAST: reconstruction error %.4f on %s exceeded threshold %.4f.",
                            err,
                            name,
                            self.max_error,
                        )
                        is_invalid = True

                    if is_invalid and self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                    replace_module(self.model, name, new_layer)
                    self.compressed_layers.append(name)

                except Exception as e_roast:
                    logger.warning("ROAST: Failed to process %s: %s", name, e_roast)
                    self.failed_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy_input)
        except Exception as ex_sanity:
            logger.warning(
                "ROAST post-compression forward sanity check warning: %s", ex_sanity
            )
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            (sum(self.layer_errors.values()) / len(self.layer_errors))
            if self.layer_errors
            else 0.0
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
                "target_attention_only": self.target_attention_only,
                "n_buckets": self.n_buckets,
                "tile_size": self.tile_size,
                "skip_attention_qkv": self.skip_attention_qkv,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "method": "roast",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("roast_factory")
@COMPRESSOR_REGISTRY.register("roast_compressor_factory")
@COMPRESSOR_REGISTRY.register("roast_hashed_linear_factory")
def roast_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """ROAST compressor factory function returning (compressed_model, stats) when requested."""
    n_buckets = kwargs.pop("n_buckets", _cfg_get(comp_config, "n_buckets", 1024))
    tile_size = kwargs.pop("tile_size", _cfg_get(comp_config, "tile_size", 32))
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", True)
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )

    compressor = ROASTCompressor(
        model,
        n_buckets=n_buckets,
        tile_size=tile_size,
        target_attention_only=target_attention_only,
        skip_attention_qkv=skip_attention_qkv,
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


__all__ = [
    "ROASTLinear",
    "ROASTCompressor",
    "roast_factory",
]
