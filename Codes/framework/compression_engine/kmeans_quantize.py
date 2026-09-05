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
                grad_W = grad_output.t().matmul(x_cast)
            else:
                grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
                x_flat = x_cast.reshape(-1, x_cast.shape[-1])
                grad_W = grad_output_flat.t().matmul(x_flat)

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

    for _ in range(max_iters):
        # Compute distances: |x_i - c_j|
        # For 1D, use searchsorted with midpoints between sorted centroids
        sorted_c, sort_perm = torch.sort(centroids)
        midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
        sorted_idx = torch.searchsorted(midpoints, flat)
        idx = sort_perm[sorted_idx]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=flat.device)
        counts.scatter_add_(0, idx, torch.ones_like(flat))
        new_centroids.scatter_add_(0, idx, flat)

        non_empty = counts > 0
        new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty]
        new_centroids[~non_empty] = centroids[~non_empty]

        shift = torch.max(torch.abs(new_centroids - centroids)).item()
        centroids = new_centroids
        if shift < tol:
            break

    # Final assignment
    sorted_c, sort_perm = torch.sort(centroids)
    midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
    sorted_idx = torch.searchsorted(midpoints, flat)
    indices = sort_perm[sorted_idx].reshape(tensor.shape)

    return centroids, indices


class KMeansLinear(nn.Module):
    """Linear layer with weights represented by k-means centroids and discrete index table with optional LoRA adapter."""

    def __init__(
        self,
        linear: Union[nn.Linear, Any],
        n_clusters: int = 256,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        raw_w = linear.weight.data
        cls_name = linear.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(linear, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous()
            self.in_features = W.shape[1]
            self.out_features = W.shape[0]
        else:
            W = raw_w.contiguous()
            self.in_features = getattr(linear, "in_features", W.shape[1])
            self.out_features = getattr(linear, "out_features", W.shape[0])

        self.n_clusters = n_clusters
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        centroids, indices = fast_kmeans_1d(W, n_clusters=n_clusters)
        self.centroids = nn.Parameter(centroids.to(dtype=W.dtype), requires_grad=(lora_r == 0))
        self.register_buffer("indices", indices)

        if getattr(linear, "bias", None) is not None and linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            dev = raw_w.device
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, self.in_features), dtype=torch.float32, device=dev))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.lora_r), dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
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
            self.centroids = nn.Parameter(centroids.to(dtype=value.dtype, device=value.device))
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

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_clusters={self.n_clusters}, lora_r={self.lora_r}"


@COMPRESSOR_REGISTRY.register("kmeans")
@COMPRESSOR_REGISTRY.register("kmeans_quantize")
@COMPRESSOR_REGISTRY.register("kmeans_compressor")
class KMeansCompressor(BaseCompressor):
    """
    KMeansCompressor: Vector quantization via scalar k-means clustering of weight parameters.

    Args:
        model: PyTorch model.
        n_clusters: Number of centroid codebook entries (default 256 -> 8 bits, 16 -> 4 bits).
        bits: Optional bitwidth constraint (e.g. 4 or 8) which sets n_clusters = 2**bits.
        max_error: Maximum allowed relative Frobenius error (default 0.20).
        fallback_to_original: If True, keep original layer on error excess (default True).
        lora_r: Optional LoRA rank.
        lora_alpha: LoRA scaling alpha.
        skip_attention_qkv: Skip asymmetric GQA projections.
        cfg: Optional configuration object.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_clusters: int = 256,
        bits: Optional[int] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="kmeans", **kwargs)
        if bits is not None:
            self.n_clusters = 2 ** int(bits)
        else:
            self.n_clusters = int(n_clusters or getattr(cfg, "kmeans_clusters", 256))
        self.bits = math.ceil(math.log2(max(self.n_clusters, 2)))
        self.max_error = float(max_error if max_error is not None else getattr(cfg, "max_error", 0.20))
        self.fallback_to_original = bool(fallback_to_original if fallback_to_original is not None else getattr(cfg, "fallback_to_original", True))
        self.lora_r = int(getattr(cfg, "lora_r", getattr(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(getattr(cfg, "lora_alpha", lora_alpha))
        self.skip_attention_qkv = bool(getattr(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.cfg = cfg
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

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, KMeansLinear):
                continue

            if not should_compress_linear(self.model, name):
                continue

            if self._is_gqa_asymmetric(name, module):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")
            orig_w = module.weight.data.t().float() if is_conv1d else module.weight.data.float()
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)

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
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue
                raise

            if err > self.max_error:
                logger.warning("KMeansCompressor: Layer %s error %.4f exceeded max_error %.4f", name, err, self.max_error)
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

            replace_module(self.model, name, new_layer)
            self.compressed_layers.append(name)
            self.layer_errors[name] = float(err)

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

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "n_clusters": self.n_clusters,
                "effective_bits_per_weight": self.bits,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            }
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kmeans_factory")
@COMPRESSOR_REGISTRY.register("kmeans_quantize_factory")
def kmeans_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """KMeans compressor factory function."""
    n_clusters = kwargs.pop("n_clusters", getattr(comp_config, "kmeans_clusters", 256) if comp_config is not None else 256)
    bits = kwargs.pop("bits", getattr(comp_config, "kmeans_bits", None) if comp_config is not None else None)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20) if comp_config is not None else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config is not None else True)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config is not None else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config is not None else 1.0)
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = KMeansCompressor(
        model,
        n_clusters=n_clusters,
        bits=bits,
        max_error=max_error,
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
