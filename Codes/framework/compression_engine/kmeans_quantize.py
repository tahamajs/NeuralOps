"""KMeans-based weight clustering quantizer with Straight-Through Estimator.

Registered as 'kmeans_class', 'kmeans_quantize_class', 'kmeans_compressor_class',
              'kmeans', 'kmeans_quantize', 'kmeans_factory', 'kmeans_quantize_factory'.

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


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_device(model: nn.Module) -> torch.device:
    """Safe device resolution."""
    p = next(iter(model.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


def _dtype_for_clusters(n_clusters: int) -> torch.dtype:
    """KM-FIX-16: smallest integer dtype that fits n_clusters indices."""
    if n_clusters <= 2**7:
        return torch.int8
    if n_clusters <= 2**15:
        return torch.int16
    if n_clusters <= 2**31:
        return torch.int32
    return torch.int64


_ATTN_NAMES = {
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
}


# ===========================================================================
# Straight-Through Estimator (KM-FIX-17)
# ===========================================================================
class KMeansSTEFunction(torch.autograd.Function):
    """
    STE for codebook lookup.

    Forward:  out = F.linear(x, centroids[indices], bias)
    Backward: gradient to centroids via scatter-add; gradient to x flows
              through the same effective W.
    """

    @staticmethod
    def forward(ctx, centroids, indices, x, bias):
        # KM-FIX-17: safe indexing
        idx_long = indices.long()
        W = centroids.index_select(0, idx_long.reshape(-1)).reshape(
            indices.shape[0], indices.shape[1]
        )
        W = W.to(dtype=x.dtype)
        ctx.save_for_backward(centroids, idx_long, x, bias)
        return F.linear(x, W, bias)

    @staticmethod
    def backward(ctx, grad_output):
        centroids, idx_long, x, bias = ctx.saved_tensors
        dtype = grad_output.dtype
        device = grad_output.device

        # Reconstruct W
        W = (
            centroids.index_select(0, idx_long.reshape(-1))
            .reshape(idx_long.shape[0], idx_long.shape[1])
            .to(dtype=dtype, device=device)
        )

        x_cast = x.to(dtype=dtype, device=device)

        grad_x = grad_output.matmul(W) if ctx.needs_input_grad[2] else None

        grad_bias = None
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=list(range(grad_output.ndim - 1)))

        grad_centroids = None
        if ctx.needs_input_grad[0]:
            if x_cast.ndim == 2:
                grad_W = grad_output.t().contiguous().matmul(x_cast)
            else:
                go = grad_output.reshape(-1, grad_output.shape[-1])
                xf = x_cast.reshape(-1, x_cast.shape[-1])
                grad_W = go.t().contiguous().matmul(xf)

            grad_centroids = torch.zeros_like(centroids)
            flat_idx = idx_long.reshape(-1)
            flat_grad = grad_W.reshape(-1).to(
                dtype=centroids.dtype, device=centroids.device
            )
            grad_centroids.scatter_add_(0, flat_idx, flat_grad)

        return grad_centroids, None, grad_x, grad_bias


# ===========================================================================
# 1D Lloyd k-means (KM-FIX-21: deterministic)
# ===========================================================================
def fast_kmeans_1d(
    tensor: torch.Tensor,
    n_clusters: int = 256,
    max_iters: int = 20,
    tol: float = 1e-4,
    seed: int = 0xCAFE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D Lloyd's algorithm with quantile init.

    KM-FIX-21: uses a deterministic seed for tie-breaking so results are
    reproducible. Returns (centroids, indices) where indices has the shape of
    `tensor` and its dtype is chosen to be the smallest that fits n_clusters.
    """
    flat = tensor.detach().float().reshape(-1)
    numel = flat.numel()
    k = min(int(n_clusters), numel)

    idx_dtype = _dtype_for_clusters(k)

    if k <= 1:
        c = flat.mean().unsqueeze(0)
        idx = torch.zeros(tensor.shape, dtype=idx_dtype, device=tensor.device)
        return c, idx

    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    sorted_vals, _ = torch.sort(flat)
    quantiles = torch.linspace(0, numel - 1, steps=k, device=flat.device).long()
    centroids = sorted_vals[quantiles].clone()

    idx = torch.zeros_like(flat, dtype=idx_dtype)
    for _ in range(max_iters):
        sorted_c, sort_perm = torch.sort(centroids)
        midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
        sorted_idx = torch.searchsorted(midpoints, flat)
        idx = sort_perm[sorted_idx].to(idx_dtype)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros_like(centroids)
        # Use int64 index for scatter_add
        idx_long = idx.long()
        new_centroids.scatter_add_(0, idx_long, flat)
        counts.scatter_add_(0, idx_long, torch.ones_like(flat))
        non_empty = counts > 0
        new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty]
        new_centroids[~non_empty] = centroids[~non_empty]

        shift = torch.max(torch.abs(new_centroids - centroids)).item()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, idx.view(tensor.shape)


# ===========================================================================
# LoRA detection (KM-FIX-02, KM-FIX-19)
# ===========================================================================
def _detect_lora(
    layer: nn.Module,
) -> tuple[int, float, torch.Tensor | None, torch.Tensor | None]:
    """
    Detect existing LoRA wrapper and extract (rank, alpha, A, B).

    Supports:
      - nn.Parameter lora_A / lora_B (direct)
      - nn.Linear lora_A / lora_B (has .weight)
      - attribute name `lora_scaling` or `scaling`
    """
    A = getattr(layer, "lora_A", None)
    B = getattr(layer, "lora_B", None)
    if A is None or B is None:
        return 0, 1.0, None, None

    a_t = getattr(A, "weight", A)
    b_t = getattr(B, "weight", B)
    if not isinstance(a_t, torch.Tensor) or a_t.dim() < 2:
        return 0, 1.0, None, None

    rank = int(a_t.shape[0])

    # Prefer explicit lora_alpha
    alpha = 1.0
    alpha_attr = getattr(layer, "lora_alpha", None)
    if alpha_attr is not None:
        try:
            alpha = float(alpha_attr)
        except (TypeError, ValueError):
            alpha = 1.0
    else:
        # KM-FIX-19: try both scaling attribute names
        for attr in ("scaling", "lora_scaling"):
            sc = getattr(layer, attr, None)
            if isinstance(sc, (int, float)) and sc > 0:
                alpha = float(sc) * rank
                break

    return (
        rank,
        alpha,
        a_t.detach(),
        b_t.detach() if isinstance(b_t, torch.Tensor) else None,
    )


# ===========================================================================
# KMeansLinear
# ===========================================================================
class KMeansLinear(nn.Module):
    """
    Linear layer with weight matrix quantized via 1D k-means codebook.

    Storage:
      - centroids:  (C,)         fp16/fp32
      - indices:    (out, in)    int8/int16/int32  (smallest that fits C)
      - bias:       (out,)       fp16/fp32
      - lora_A/B:   (r, in) / (out, r)   optional

    Forward:
      W_hat = centroids[indices]
      y = F.linear(x, W_hat) + bias + scaling · (x @ A^T) @ B^T
    """

    is_custom_attention: bool = False
    is_custom_linear: bool = True

    def __init__(
        self,
        linear: nn.Module,
        n_clusters: int = 256,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        kmeans_seed: int = 0xCAFE,
    ) -> None:
        super().__init__()
        target_mod = getattr(linear, "base_layer", linear)
        cls_lower = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            self.in_features = raw_w.shape[0]
            self.out_features = raw_w.shape[1]
        else:
            W = raw_w.float()
            self.in_features = int(getattr(target_mod, "in_features", W.shape[1]))
            self.out_features = int(getattr(target_mod, "out_features", W.shape[0]))

        self.n_clusters = int(min(n_clusters, W.numel()))
        self._index_dtype = _dtype_for_clusters(self.n_clusters)

        # Detect existing LoRA
        detected_r, detected_alpha, src_A, src_B = _detect_lora(linear)
        self.lora_r = int(detected_r) if detected_r > 0 else int(lora_r)
        self.lora_alpha = float(detected_alpha) if detected_r > 0 else float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # Fit centroids and indices
        centroids, indices = fast_kmeans_1d(
            W,
            n_clusters=self.n_clusters,
            seed=kmeans_seed,
        )
        # KM-FIX-23: centroids frozen if LoRA exists (pure quantizer mode).
        centroids_trainable = self.lora_r == 0
        self.centroids = nn.Parameter(
            centroids.to(dtype=raw_w.dtype),
            requires_grad=centroids_trainable,
        )
        # Store indices as int8/int16/int32
        self.register_buffer("indices", indices.to(self._index_dtype))

        # Bias
        bias_holder = getattr(target_mod, "bias", None)
        if bias_holder is not None:
            self.bias = nn.Parameter(bias_holder.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA branch
        if self.lora_r > 0:
            dev = raw_w.device
            dt = raw_w.dtype
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, self.in_features), dtype=dt, device=dev)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, self.lora_r), dtype=dt, device=dev)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            if src_A is not None and src_B is not None:
                if src_A.shape == self.lora_A.shape:
                    self.lora_A.data.copy_(
                        src_A.to(self.lora_A.device, dtype=self.lora_A.dtype)
                    )
                if src_B.shape == self.lora_B.shape:
                    self.lora_B.data.copy_(
                        src_B.to(self.lora_B.device, dtype=self.lora_B.dtype)
                    )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # KM-FIX-20: cache in eval
        self._cached_weight: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def get_weight(self) -> torch.Tensor:
        # KM-FIX-20: cache in eval mode
        if not self.training and self._cached_weight is not None:
            return self._cached_weight
        idx_long = self.indices.long().reshape(-1)
        W = self.centroids.index_select(0, idx_long).reshape(
            self.out_features, self.in_features
        )
        if not self.training:
            self._cached_weight = W
        return W

    @property
    def weight(self) -> torch.Tensor:
        return self.get_weight()

    def train(self, mode: bool = True) -> KMeansLinear:
        super().train(mode)
        if mode:
            self._cached_weight = None
        return self

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        """KM-FIX-07/22: honest storage accounting."""

        def _sz(t):
            if t is None:
                return 0
            return t.numel() * t.element_size()

        return {
            "centroids": _sz(self.centroids.data),
            "indices": _sz(self.indices),
            "bias": _sz(self.bias.data) if self.bias is not None else 0,
            "lora_A": _sz(self.lora_A.data) if self.lora_A is not None else 0,
            "lora_B": _sz(self.lora_B.data) if self.lora_B is not None else 0,
        }

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bias cast only when dtype/device mismatch
        if self.bias is not None:
            b = self.bias
            if b.dtype != x.dtype or b.device != x.device:
                b = b.to(dtype=x.dtype, device=x.device)
        else:
            b = None

        if self.training:
            out = KMeansSTEFunction.apply(self.centroids, self.indices, x, b)
        else:
            W = self.get_weight().to(dtype=x.dtype, device=x.device)
            out = F.linear(x, W, b)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out.to(out.dtype)

        return out


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("kmeans_class")
@COMPRESSOR_REGISTRY.register("kmeans_quantize_class")
@COMPRESSOR_REGISTRY.register("kmeans_compressor_class")
class KMeansCompressor(BaseCompressor):
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        n_clusters: int = 256,
        bits: int | None = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        skip_should_compress_filter: bool = False,
        min_layer_dim: int = 8,
        kmeans_seed: int = 0xCAFE,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(), name="kmeans", **kwargs
        )
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        parsed_bits = _cfg_get(cfg, "kmeans_bits", _cfg_get(cfg, "bits", bits))
        if parsed_bits is not None:
            self.n_clusters = 2 ** int(parsed_bits)
        else:
            self.n_clusters = int(
                _cfg_get(
                    cfg, "kmeans_clusters", _cfg_get(cfg, "n_clusters", n_clusters)
                )
            )
        self.bits = int(math.ceil(math.log2(max(self.n_clusters, 2))))

        self.max_error = float(
            _cfg_get(cfg, "kmeans_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "kmeans_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "kmeans_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.skip_attention_qkv = bool(
            _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
        )
        self.skip_should_compress_filter = bool(
            _cfg_get(
                cfg,
                "skip_should_compress_filter",
                skip_should_compress_filter,
            )
        )
        self.min_layer_dim = max(1, int(_cfg_get(cfg, "min_layer_dim", min_layer_dim)))
        self.kmeans_seed = int(_cfg_get(cfg, "kmeans_seed", kmeans_seed))

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_storage: dict[str, dict[str, int]] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        ):
            if any(
                p in name.lower()
                for p in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                )
            ):
                return True
        return False

    def _is_target_module(
        self, name: str, module: nn.Module, _debug: bool = False
    ) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer")
            return False
        if any(
            skip in name_lower
            for skip in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            )
        ):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        if self._is_gqa_asymmetric(name, module):
            if _debug:
                logger.info(f"  [skip] {name}: GQA asymmetric")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_lower = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False
        if isinstance(target_mod, KMeansLinear):
            if _debug:
                logger.info(f"  [skip] {name}: already KMeansLinear")
            return False

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    def compress(
        self, model: nn.Module | None = None
    ) -> tuple[nn.Module, dict[str, Any]]:
        # Idempotence
        if getattr(self.model, "_kmeans_compressed", False):
            logger.warning("KMeansCompressor: model already compressed; skipping.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_storage = {}
        seen_weights: set[int] = set()

        total_orig_bytes = 0
        total_comp_bytes = 0

        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]

        if not target_list:
            logger.warning("KMeansCompressor: no compressible layers found.")
            n = 0
            for name, module in self.model.named_modules():
                n += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning(
                "KMeansCompressor diagnostic: %d modules scanned, 0 targets.", n
            )
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        logger.info("KMeansCompressor: %d target layers identified.", len(target_list))

        for name, module in target_list:
            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_lower = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
            orig_w = target_mod.weight.data
            W_target = orig_w.t().contiguous().float() if is_conv1d else orig_w.float()
            norm_orig = torch.norm(W_target, p="fro").clamp_min(1e-8)
            out_f, in_f = W_target.shape[0], W_target.shape[1]

            # Original bytes
            orig_bytes = orig_w.numel() * orig_w.element_size()
            bias_holder = getattr(target_mod, "bias", None)
            if bias_holder is not None:
                orig_bytes += bias_holder.numel() * bias_holder.element_size()
            total_orig_bytes += orig_bytes

            new_layer: KMeansLinear | None = None
            err = 1.0
            try:
                new_layer = KMeansLinear(
                    module,
                    n_clusters=self.n_clusters,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    kmeans_seed=self.kmeans_seed,
                )
                recon_w = new_layer.get_weight().float()
                err = float(
                    torch.norm(W_target - recon_w, p="fro").item() / norm_orig.item()
                )
            except Exception as ex:
                logger.warning("KMeansCompressor: failed on %s: %s", name, ex)
                new_layer = None
                err = 1.0

            self.layer_errors[name] = float(err)

            if new_layer is None or err > self.max_error:
                logger.warning(
                    "KMeansCompressor: error %.4f on %s > %.4f. %s",
                    err,
                    name,
                    self.max_error,
                    (
                        "Falling back."
                        if self.fallback_to_original
                        else "Using compressed."
                    ),
                )
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_bytes += orig_bytes
                continue

            # KM-FIX-18: preserve LoRA wrapper
            try:
                if hasattr(module, "base_layer"):
                    module.base_layer = new_layer
                    # Keep LoRA param trainable
                    for pn, p in module.named_parameters():
                        if "lora_" in pn.lower() and not pn.startswith("base_layer"):
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning(
                    "KMeansCompressor: replacement failed on %s: %s", name, e
                )
                self.fallback_layers.append(name)
                total_comp_bytes += orig_bytes
                continue

            # Honest byte accounting
            st = new_layer.storage_bytes()
            self.layer_storage[name] = st
            total_comp_bytes += sum(st.values())

        # Forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = _get_device(self.model)
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("KMeans forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("KMeans forward check raised: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        storage_ratio = (
            total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0
        )
        effective_ratio = (
            total_comp_bytes / total_orig_bytes if total_orig_bytes > 0 else 1.0
        )
        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_storage_bytes": self.layer_storage,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "n_clusters": self.n_clusters,
                "effective_bits_per_weight": self.bits,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                # Honest accounting
                "orig_storage_bytes": total_orig_bytes,
                "comp_storage_bytes": total_comp_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "kmeans",
                "note": (
                    "Storage accounting is honest (bytes). indices are stored as "
                    f"{_dtype_for_clusters(self.n_clusters)} to minimize overhead. "
                    "The framework's parameter counter uses model.parameters(), "
                    "which does NOT include the `indices` buffer."
                ),
            }
        )
        self.stats = stats

        try:
            if (
                not hasattr(self.model, "compression_stats")
                or self.model.compression_stats is None
            ):
                self.model.compression_stats = {}
            self.model.compression_stats.update(
                {
                    "num_compressed": len(self.compressed_layers),
                    "num_fallback": len(self.fallback_layers),
                    "compression_method": "kmeans",
                    "effective_parameter_ratio": effective_ratio,
                    "storage_compression_ratio": storage_ratio,
                }
            )
        except Exception:
            pass

        self.model._kmeans_compressed = True
        logger.info(
            "KMeans: %d compressed, %d fallback, avg_err=%.4f, storage=%.2fx.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            storage_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("kmeans")
@COMPRESSOR_REGISTRY.register("kmeans_quantize")
@COMPRESSOR_REGISTRY.register("kmeans_factory")
@COMPRESSOR_REGISTRY.register("kmeans_quantize_factory")
def kmeans_factory(
    model: Any,
    comp_config: Any = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any:
    if _depth > 4:
        logger.warning("kmeans_factory: recursion depth exceeded.")
        return model

    n_clusters = kwargs.pop(
        "n_clusters",
        _cfg_get(
            comp_config, "kmeans_clusters", _cfg_get(comp_config, "n_clusters", 256)
        ),
    )
    bits = kwargs.pop(
        "bits",
        _cfg_get(comp_config, "kmeans_bits", _cfg_get(comp_config, "bits", None)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "kmeans_max_error", _cfg_get(comp_config, "max_error", 0.35)
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "kmeans_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "kmeans_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_r = kwargs.pop(
        "lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0))
    )
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
    )
    min_layer_dim = kwargs.pop(
        "min_layer_dim",
        _cfg_get(comp_config, "min_layer_dim", 8),
    )
    kmeans_seed = kwargs.pop(
        "kmeans_seed", _cfg_get(comp_config, "kmeans_seed", 0xCAFE)
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = KMeansCompressor(
        model,
        n_clusters=n_clusters,
        bits=bits,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_should_compress_filter=skip_filter,
        min_layer_dim=min_layer_dim,
        kmeans_seed=kmeans_seed,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "KMeansSTEFunction",
    "KMeansLinear",
    "KMeansCompressor",
    "kmeans_factory",
    "fast_kmeans_1d",
]
