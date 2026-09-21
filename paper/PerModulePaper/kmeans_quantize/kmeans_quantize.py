"""KMeans-based weight clustering quantizer with STE. Registered as 'kmeans' and 'kmeans_quantize'."""
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


class KMeansSTEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for codebook lookup.
    Forward projects centroid values by index; backward passes gradients directly to centroids.
    """

    @staticmethod
    def forward(
        ctx: Any,
        centroids: torch.Tensor,
        indices: torch.Tensor,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(centroids, indices, x, bias)
        W = centroids[indices].to(dtype=x.dtype)
        return F.linear(x, W, bias)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], Optional[torch.Tensor]]:
        centroids, indices, x, bias = ctx.saved_tensors
        dtype = grad_output.dtype
        device = grad_output.device
        W = centroids[indices].to(dtype=dtype, device=device)
        x_cast = x.to(dtype=dtype, device=device)

        # Gradient w.r.t input x
        grad_x = grad_output.matmul(W) if ctx.needs_input_grad[2] else None

        # Gradient w.r.t bias
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=list(range(grad_output.ndim - 1)))

        # Gradient w.r.t centroids
        grad_centroids = None
        if ctx.needs_input_grad[0]:
            if x_cast.ndim == 2:
                grad_W = grad_output.t().contiguous().matmul(x_cast)
            else:
                grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
                x_flat = x_cast.reshape(-1, x_cast.shape[-1])
                grad_W = grad_output_flat.t().contiguous().matmul(x_flat)

            grad_centroids = torch.zeros_like(centroids)
            flat_indices = indices.view(-1)
            flat_grad_W = grad_W.view(-1).to(dtype=centroids.dtype, device=centroids.device)
            grad_centroids.scatter_add_(0, flat_indices, flat_grad_W)

        return grad_centroids, None, grad_x, grad_bias


def fast_kmeans_1d(
    tensor: torch.Tensor,
    n_clusters: int = 256,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast 1D Lloyd's algorithm on PyTorch tensor with linear quantile initialization.
    Returns (centroids, indices).
    """
    flat = tensor.detach().float().view(-1)
    numel = flat.numel()
    k = min(n_clusters, numel)

    if k <= 1:
        c = flat.mean().unsqueeze(0)
        idx = torch.zeros(tensor.shape, dtype=torch.long, device=tensor.device)
        return c, idx

    # Quantile-based centroid initialization
    sorted_vals, _ = torch.sort(flat)
    quantiles = torch.linspace(0, numel - 1, steps=k, device=flat.device).long()
    centroids = sorted_vals[quantiles].clone()
    idx = torch.zeros(flat.shape, dtype=torch.long, device=flat.device)

    for _ in range(max_iters):
        # Midpoints between sorted centroids for fast searchsorted lookup
        sorted_c, sort_perm = torch.sort(centroids)
        midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
        sorted_idx = torch.searchsorted(midpoints, flat)
        idx = sort_perm[sorted_idx]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros_like(centroids)
        new_centroids.scatter_add_(0, idx, flat)
        counts.scatter_add_(0, idx, torch.ones_like(flat))

        non_empty = counts > 0
        new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty]
        new_centroids[~non_empty] = centroids[~non_empty]

        shift = torch.max(torch.abs(new_centroids - centroids)).item()
        centroids = new_centroids
        if shift < tol:
            break

    final_idx = idx.view(tensor.shape)
    return centroids, final_idx


class KMeansLinear(nn.Module):
    """Linear layer using K-Means vector quantization and Straight-Through Estimator with optional LoRA."""

    def __init__(
        self,
        linear: nn.Module,
        n_clusters: int = 256,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        target_mod = getattr(linear, "base_layer", linear)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            W = raw_w.t().contiguous().float()  # [out_features, in_features]
            self.in_features = raw_w.shape[0]
            self.out_features = raw_w.shape[1]
        else:
            W = raw_w.float()      # [out_features, in_features]
            self.in_features = getattr(target_mod, "in_features", W.shape[1])
            self.out_features = getattr(target_mod, "out_features", W.shape[0])

        self.n_clusters = n_clusters
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        centroids, indices = fast_kmeans_1d(W, n_clusters=n_clusters)
        self.centroids = nn.Parameter(centroids.to(dtype=raw_w.dtype), requires_grad=(lora_r == 0))
        self.register_buffer("indices", indices)

        if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
            self.bias = nn.Parameter(target_mod.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            self.lora_r = existing_lora_A.weight.shape[0]

        if self.lora_r > 0:
            dev = raw_w.device
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, self.in_features), dtype=torch.float32, device=dev))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.lora_r), dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            if existing_lora_A is not None and existing_lora_B is not None:
                w_A = getattr(existing_lora_A, "weight", existing_lora_A)
                w_B = getattr(existing_lora_B, "weight", existing_lora_B)
                if hasattr(w_A, "data") and hasattr(w_B, "data"):
                    self.lora_A.data.copy_(w_A.data)
                    self.lora_B.data.copy_(w_B.data)
                    self.lora_A.requires_grad = True
                    self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        if hasattr(self, "centroids") and hasattr(self, "indices"):
            return self.centroids[self.indices]
        raise AttributeError("'KMeansLinear' object has no centroids/indices initialized")

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            centroids, indices = fast_kmeans_1d(value.data, n_clusters=self.n_clusters)
            prev_requires_grad = getattr(self.centroids, "requires_grad", True)
            self.centroids = nn.Parameter(centroids.to(dtype=value.dtype, device=value.device), requires_grad=prev_requires_grad)
            self.register_buffer("indices", indices.to(device=value.device))

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return self.centroids[self.indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = KMeansSTEFunction.apply(
                self.centroids,
                self.indices,
                x,
                self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None,
            )
        else:
            W = self.centroids[self.indices].to(dtype=x.dtype, device=x.device)
            out = F.linear(
                x,
                W,
                self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None,
            )

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out


@COMPRESSOR_REGISTRY.register("kmeans")
@COMPRESSOR_REGISTRY.register("kmeans_quantize")
@COMPRESSOR_REGISTRY.register("kmeans_compressor")
class KMeansCompressor(BaseCompressor):
    """
    KMeansCompressor: Vector quantization via scalar k-means clustering of weight parameters.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_clusters: int = 256,
        bits: Optional[int] = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="kmeans", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.cfg = cfg

        parsed_bits = _cfg_get(cfg, "kmeans_bits", _cfg_get(cfg, "bits", bits))
        if parsed_bits is not None:
            self.n_clusters = 2 ** int(parsed_bits)
        else:
            self.n_clusters = int(_cfg_get(cfg, "kmeans_clusters", _cfg_get(cfg, "n_clusters", n_clusters)))
        self.bits = math.ceil(math.log2(max(self.n_clusters, 2)))

        self.max_error = float(_cfg_get(cfg, "kmeans_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "kmeans_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "kmeans_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d) or isinstance(target_mod, KMeansLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace eligible nn.Linear and Conv1D modules with KMeansLinear layers."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze skipped non-target layers if target_attention_only is enabled
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            orig_w = target_mod.weight.data.t().contiguous().float() if is_conv1d else target_mod.weight.data.float()
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)
            out_f, in_f = orig_w.shape[0], orig_w.shape[1]
            orig_layer_params = in_f * out_f
            total_orig_params += orig_layer_params

            try:
                new_layer = KMeansLinear(
                    module,
                    n_clusters=self.n_clusters,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                recon_w = new_layer.get_weight().float()
                err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()
            except Exception as ex:
                logger.warning("KMeansCompressor: Error compressing %s: %s", name, ex)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_layer_params
                continue

            if err > self.max_error:
                logger.warning("KMeansCompressor: Layer %s error %.4f exceeded max_error %.4f", name, err, self.max_error)
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            replace_module(self.model, name, new_layer)
            self.compressed_layers.append(name)
            self.layer_errors[name] = float(err)
            comp_layer_params = new_layer.n_clusters + int(orig_layer_params * (self.bits / 16.0))
            if new_layer.lora_r > 0:
                comp_layer_params += new_layer.lora_r * (in_f + out_f)
            total_comp_params += comp_layer_params

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, KMeansLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("KMeans post-compression forward check warning: %s", e)
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / max(1, len(self.layer_errors))) if self.layer_errors else 0.0
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
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "n_clusters": self.n_clusters,
                "effective_bits_per_weight": self.bits,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "effective_parameter_ratio_of_target_layers": round(effective_ratio, 4),
                "overall_model_compression_ratio": round(self._original_params / max(1, self._compressed_params), 4) if hasattr(self, "_original_params") else 1.0,
                "forward_check_success": forward_check_success,
                "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            }
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kmeans_factory")
@COMPRESSOR_REGISTRY.register("kmeans_quantize_factory")
def kmeans_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """KMeans compressor factory function."""
    n_clusters = kwargs.pop("n_clusters", _cfg_get(comp_config, "kmeans_clusters", _cfg_get(comp_config, "n_clusters", 256)))
    bits = kwargs.pop("bits", _cfg_get(comp_config, "kmeans_bits", _cfg_get(comp_config, "bits", None)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "kmeans_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "kmeans_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "kmeans_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 16)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
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
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["KMeansSTEFunction", "KMeansLinear", "KMeansCompressor", "kmeans_factory", "fast_kmeans_1d"]
