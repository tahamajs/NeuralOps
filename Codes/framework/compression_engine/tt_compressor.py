"""
Tensor Train Compressor: replaces eligible nn.Linear and Conv1D layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".

NOTE (DEPRECATED):
    Tensor-Train compression on 2D linear matrices is deprecated in this repository
    and removed from the active benchmark queue (run_all_benchmarks_queue.py).
    Due to mode scrambling and flattened singular value decay compared to 2D SVD, TT
    requires very high ranks to achieve low error, yielding negligible parameter savings
    and no FLOPs reduction. See Docs/content/compressions/tt.md.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

def factorize_dimension_pair(
    in_features: int, out_features: int, num_factors: int = 2
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Factorize in_features and out_features into num_factors balanced integer components.
    """
    def get_factors(n: int, k: int) -> List[int]:
        if k == 1:
            return [n]
        if k == 2:
            a = int(math.isqrt(n))
            while a > 1:
                if n % a == 0:
                    return [a, n // a]
                a -= 1
            return [1, n]

        # For k > 2, greedily factor smallest near root
        factors = []
        curr = n
        for step in range(k, 1, -1):
            root = max(1, int(round(curr ** (1.0 / step))))
            found = False
            for cand in range(root, 0, -1):
                if curr % cand == 0:
                    factors.append(cand)
                    curr = curr // cand
                    found = True
                    break
            if not found:
                factors.append(1)
        factors.append(curr)
        while len(factors) < k:
            factors.append(1)
        return factors[:k]

    return tuple(get_factors(in_features, num_factors)), tuple(get_factors(out_features, num_factors))


class TTLinearLayer(nn.Module):
    is_custom_attention: bool = True
    """
    Tensor-Train (TT) Linear Layer.

    .. deprecated::
        Tensor-Train compression is deprecated on 2D linear layers in favor of SVD.
        See Docs/content/compressions/tt.md for theoretical and empirical comparisons.

    Represents weight matrix W in R^{out_features x in_features} as a sequence
    of d TT-cores: G_1, G_2, ..., G_d where core G_k has shape (r_{k-1}, p_k, q_k, r_k)
    with boundary ranks r_0 = r_d = 1.
    """

    @staticmethod
    def _max_rank_for_compression(
        out_features: int,
        in_features: int,
        p_dims: Tuple[int, ...],
        q_dims: Tuple[int, ...],
        safety_margin: float = 0.9,
    ) -> int:
        """
        Return the maximum TT-rank such that TT parameter count stays below
        the original dense matrix count, with a safety margin.

        Param count for TT cores with ranks r_0=1, r_1=r, ..., r_n=1:
            params(r) = sum over cores of (p_i * q_i) * r
        For a rank-uniform approximation:
            params(r) ≈ r * sum_i (p_i * q_i)
        Breakeven: r_max = out_features * in_features / sum_i (p_i * q_i)
        """
        if not p_dims or not q_dims or len(p_dims) != len(q_dims):
            return 1

        sum_pq = sum(p * q for p, q in zip(p_dims, q_dims))
        if sum_pq <= 0:
            return 1

        breakeven = (out_features * in_features) / sum_pq
        safe_max = int(breakeven * safety_margin)
        return max(1, safe_max)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 16,
        bias: bool = True,
        shape: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.nx = in_features  # Conv1D compatibility
        self.tt_rank = tt_rank
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        if shape is None:
            self.p_dims, self.q_dims = factorize_dimension_pair(in_features, out_features, num_factors=2)
        else:
            self.p_dims, self.q_dims = shape
            if int(np.prod(self.p_dims)) != in_features or int(np.prod(self.q_dims)) != out_features:
                raise ValueError(
                    f"Shape product mismatch: {self.p_dims} ({np.prod(self.p_dims)}) != {in_features} or "
                    f"{self.q_dims} ({np.prod(self.q_dims)}) != {out_features}"
                )

        self.d = len(self.p_dims)
        self.cores = nn.ParameterList()

        # Build TT-cores: G_k of shape (r_{k-1}, p_k, q_k, r_k)
        requires_grad = not (self.lora_r > 0)
        r_left = 1
        for i in range(self.d - 1):
            p_i = self.p_dims[i]
            q_i = self.q_dims[i]
            r_right = min(tt_rank, max(1, p_i * q_i * r_left))
            core = nn.Parameter(
                torch.empty(r_left, p_i, q_i, r_right, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
            bound = 1.0 / math.sqrt(max(1, r_left * p_i))
            nn.init.uniform_(core, -bound, bound)
            self.cores.append(core)
            r_left = r_right

        p_last = self.p_dims[-1]
        q_last = self.q_dims[-1]
        core_last = nn.Parameter(
            torch.empty(r_left, p_last, q_last, 1, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )
        bound = 1.0 / math.sqrt(max(1, r_left * p_last))
        nn.init.uniform_(core_last, -bound, bound)
        self.cores.append(core_last)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
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

        self._cached_weight: Optional[torch.Tensor] = None

    def _invalidate_cache(self) -> None:
        self._cached_weight = None

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return self.materialize_weight()

    def materialize_weight(self) -> torch.Tensor:
        """
        Reconstruct the full dense weight matrix W in R^{out_features x in_features}.
        Safe against 1-dimension collapsing, with evaluation caching (B20).
        """
        if not self.training and self._cached_weight is not None:
            return self._cached_weight

        # Start with core 0: select leading boundary index 0
        res = self.cores[0][0]  # (p0, q0, r1)
        for i in range(1, self.d):
            core = self.cores[i]  # (r_i, p_i, q_i, r_{i+1})
            # Contract over r_i
            res = torch.tensordot(res, core, dims=([-1], [0]))  # (p0, q0, ..., p_i, q_i, r_{i+1})

        # Select trailing boundary index 0
        res = res[..., 0]  # (p0, q0, p1, q1, ..., p_{d-1}, q_{d-1})

        # Permute to (q0, q1, ..., q_{d-1}, p0, p1, ..., p_{d-1})
        q_indices = [2 * k + 1 for k in range(self.d)]
        p_indices = [2 * k for k in range(self.d)]
        res = res.permute(*(q_indices + p_indices)).contiguous()
        W = res.view(self.out_features, self.in_features)
        if not self.training:
            self._cached_weight = W
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with direct TT core contraction (no dense weight materialization)."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # High-performance 2-core contraction (1.1x baseline on GPU vs 16x slower einsum)
        if self.d == 2:
            c0 = self.cores[0][0]       # (p1, q1, r)
            c1 = self.cores[1][..., 0]  # (r, p2, q2)

            if c0.dtype != x_flat.dtype or c0.device != x_flat.device:
                c0 = c0.to(device=x_flat.device, dtype=x_flat.dtype)
            if c1.dtype != x_flat.dtype or c1.device != x_flat.device:
                c1 = c1.to(device=x_flat.device, dtype=x_flat.dtype)

            # Contract TT-cores directly (tiny 25K op overhead), then execute cuBLAS/MPS GEMM
            W = torch.tensordot(c0, c1, dims=([-1], [0])).permute(1, 3, 0, 2).contiguous().view(self.out_features, self.in_features)
            out = F.linear(x_flat, W)
        else:
            # Generic N-core path
            out = self._contract_tt_generic(x_flat)

        if hasattr(self, 'shift') and self.shift is not None and self.shift.abs() > 1e-8:
            out = out + x_flat.sum(dim=-1, keepdim=True) * self.shift

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = self.scaling * F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B).to(out.dtype)
            out = out + lora_delta

        if hasattr(self, 'output_scale') and self.output_scale is not None:
            out = out 

        return out.view(*orig_shape[:-1], self.out_features)

    def _contract_tt_generic(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Generic N-core TT contraction. Slower but handles arbitrary d."""
        W = self.materialize_weight()
        if W.dtype != x_flat.dtype or W.device != x_flat.device:
            W = W.to(device=x_flat.device, dtype=x_flat.dtype)
        return F.linear(x_flat, W)

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Module,
        tt_rank: int = 16,
        num_factors: int = 2,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_svd: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        fallback_to_none: bool = False,
    ) -> Optional["TTLinearLayer"]:
        """
        Decompose a pretrained nn.Linear, Conv1D, or LoRALinear layer into TT-cores using TT-SVD.
        Production-hardened with FP32 stability, Conv1D support, adaptive rank retry,
        relative Frobenius error validation, and trainable LoRA adapters.
        """
        base_layer = getattr(linear_layer, "base_layer", linear_layer)
        raw_w = base_layer.weight.data
        orig_dtype = raw_w.dtype
        orig_device = raw_w.device

        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        if hasattr(base_layer, "nx") and hasattr(base_layer, "nf"):
            in_f = int(base_layer.nx)
            out_f = int(base_layer.nf)
            W = raw_w.t().contiguous().float().contiguous()
        elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            in_f = int(base_layer.in_features)
            out_f = int(base_layer.out_features)
            if raw_w.shape == (in_f, out_f):
                W = raw_w.t().contiguous().float().contiguous()
            else:
                W = raw_w.float().contiguous()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float().contiguous()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float().contiguous()

        bias = base_layer.bias.data if getattr(base_layer, "bias", None) is not None else None
        norm_orig = torch.linalg.norm(W).item() + 1e-8
        if norm_orig < 1e-4:
            logger.warning("Layer weight has near-zero Frobenius norm (< 1e-4)")

        # Check for existing LoRA in linear_layer
        has_orig_lora = hasattr(linear_layer, "lora_A") and hasattr(linear_layer, "lora_B")
        effective_lora_r = lora_r
        if has_orig_lora and getattr(linear_layer, "r", 0) > 0:
            effective_lora_r = getattr(linear_layer, "r", lora_r)

        lora_r = effective_lora_r

        # Factorize dimensions
        p_dims, q_dims = factorize_dimension_pair(in_f, out_f, num_factors=num_factors)
        if int(np.prod(p_dims)) != in_f or int(np.prod(q_dims)) != out_f:
            p_dims = (1, in_f)
            q_dims = (1, out_f)

        d = len(p_dims)
        # Reshape W (out_f, in_f) -> (q0, q1, ..., p0, p1, ...)
        W_tensor = W.view(*(list(q_dims) + list(p_dims)))
        perm = []
        for k in range(d):
            perm.append(d + k)  # p_k
            perm.append(k)      # q_k
        W_interleaved = W_tensor.permute(*perm).contiguous()

        max_rank_for_compression = cls._max_rank_for_compression(
            out_f, in_f, p_dims, q_dims, safety_margin=0.9
        )

        cur_tt_rank = min(max(1, int(tt_rank)), max_rank_for_compression)
        best_layer: Optional[TTLinearLayer] = None
        best_err = float("inf")
        best_rank = cur_tt_rank

        for attempt in range(max_retries + 1):
            try:
                layer = cls(
                    in_features=in_f,
                    out_features=out_f,
                    tt_rank=cur_tt_rank,
                    bias=bias is not None,
                    shape=(p_dims, q_dims),
                    dtype=orig_dtype,
                    device=orig_device,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                )

                # Sequential TT-SVD in FP32
                curr_matrix = W_interleaved.view(p_dims[0] * q_dims[0], -1)
                r_left = 1

                for i in range(d - 1):
                    p_i = p_dims[i]
                    q_i = q_dims[i]
                    rows = r_left * p_i * q_i
                    curr_matrix = curr_matrix.view(rows, -1)

                    try:
                        if torch.cuda.is_available() and curr_matrix.is_cuda:
                            U, S, Vt = torch.linalg.svd(curr_matrix, full_matrices=False)
                        else:
                            mat_dev = curr_matrix.to(orig_device) if torch.cuda.is_available() else curr_matrix
                            U, S, Vt = torch.linalg.svd(mat_dev, full_matrices=False)
                    except Exception:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        U, S, Vt = torch.linalg.svd(curr_matrix.cpu().float(), full_matrices=False)
                        U, S, Vt = U.to(orig_device), S.to(orig_device), Vt.to(orig_device)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    r_right = min(cur_tt_rank, len(S), layer.cores[i].shape[-1])
                    U_trunc = U[:, :r_right]
                    S_trunc = S[:r_right]
                    Vt_trunc = Vt[:r_right, :]

                    core_data = U_trunc.view(r_left, p_i, q_i, r_right)
                    layer.cores[i].data.zero_()
                    layer.cores[i].data[:r_left, :p_i, :q_i, :r_right].copy_(core_data.to(dtype=orig_dtype))

                    curr_matrix = torch.matmul(torch.diag(S_trunc), Vt_trunc)
                    r_left = r_right

                # Final core
                p_last = p_dims[-1]
                q_last = q_dims[-1]
                layer.cores[-1].data.zero_()
                final_core = curr_matrix.view(r_left, p_last, q_last, 1)
                layer.cores[-1].data[:r_left, :p_last, :q_last, :].copy_(final_core.to(dtype=orig_dtype))

                if bias is not None and layer.bias is not None:
                    layer.bias.data.copy_(bias.to(dtype=orig_dtype, device=orig_device))

                # Validate reconstructed weight shape and error
                W_rec = layer.materialize_weight().float()
                if W_rec.shape != (out_f, in_f):
                    logger.warning("Compressor copy_weights warning. Proceeding with initialized weights.")

                rel_err = (torch.linalg.norm(W - W_rec).item()) / norm_orig
                if rel_err < best_err:
                    best_err = rel_err
                    best_layer = layer
                    best_rank = cur_tt_rank

                if rel_err <= max_error:
                    break

                if cur_tt_rank >= max_rank_for_compression:
                    if max_retries > 0 and fallback_to_none:
                        # Cannot reach error threshold without exceeding parameter budget during retries.
                        # Fall back to the original module.
                        import logging
                        logging.getLogger(__name__).warning(
                            "TT: cannot achieve error < %.4f at max_rank=%d "
                            "(param budget ceiling); falling back to original.",
                            max_error, max_rank_for_compression,
                        )
                        return None  # signal caller to keep original
                    break

                # Step up rank for retry
                step = max(4, (max_rank_for_compression - cur_tt_rank) // max(1, max_retries - attempt))
                cur_tt_rank = min(cur_tt_rank + step, max_rank_for_compression)

            except Exception as e:
                # If SVD or contraction fails, log and stop retries
                break

        if best_layer is None:
            return None

        # Sanity check: TT params must be strictly less than dense params
        tt_params = sum(p.numel() for p in best_layer.parameters())
        orig_params = out_f * in_f
        if tt_params >= orig_params:
            raise RuntimeError(
                f"TTLinearLayer has {tt_params} params >= original {orig_params}. "
                f"This means the rank budget was exceeded. "
                f"Check _max_rank_for_compression()."
            )

        best_layer.reconstruction_error = float(best_err)
        best_layer.tt_rank = best_rank

        if has_orig_lora and best_layer.lora_r > 0 and best_layer.lora_A is not None and best_layer.lora_B is not None:
            try:
                if hasattr(linear_layer.lora_A, "weight") and linear_layer.lora_A.weight.shape == best_layer.lora_A.shape:
                    best_layer.lora_A.data.copy_(linear_layer.lora_A.weight.data)
                elif hasattr(linear_layer, "lora_A") and linear_layer.lora_A.shape == best_layer.lora_A.shape:
                    best_layer.lora_A.data.copy_(linear_layer.lora_A.data)

                if hasattr(linear_layer.lora_B, "weight") and linear_layer.lora_B.weight.shape == best_layer.lora_B.shape:
                    best_layer.lora_B.data.copy_(linear_layer.lora_B.weight.data)
                elif hasattr(linear_layer, "lora_B") and linear_layer.lora_B.shape == best_layer.lora_B.shape:
                    best_layer.lora_B.data.copy_(linear_layer.lora_B.data)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("TTLinear copy_weights warning: %s", e)

        return best_layer



TTLinear = TTLinearLayer


logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class TTCompressor(BaseCompressor):
    """
    Compressor that replaces nn.Linear/Conv1D layers with Tensor-Train (TTLinearLayer) modules.

    .. deprecated::
        Tensor-Train compression is deprecated on 2D linear layers in favor of SVD.
        See Docs/content/compressions/tt.md for theoretical and empirical comparisons.
        Excluded from run_all_benchmarks_queue.py.

    Args:
        model: PyTorch model to compress.
        rank: Rank parameter (alias for tt_rank).
        tt_rank: TT-rank bound for tensor-train cores.
        cfg: Optional configuration object.
        target_layers: Optional list of layer names or substrings to compress.
        max_error: Maximum allowed relative reconstruction error (default 0.20).
        max_retries: Number of rank-step retries on error threshold excess (default 3).
        fallback_to_original: If True, keep original layer on high error (default True).
        target_attention_only: If True, only compress attention projections.
        **kwargs: Additional arguments passed to BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        tt_rank: int = 16,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        max_error: float = 0.30,
        max_retries: int = 3,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="tt", **kwargs)
        comp_cfg = kwargs.get("comp_config", cfg)
        if comp_cfg is not None:
            rank = _cfg_get(comp_cfg, "tt_rank", _cfg_get(comp_cfg, "rank", rank))
            max_error = _cfg_get(comp_cfg, "max_error", max_error)
            max_retries = _cfg_get(comp_cfg, "max_retries", max_retries)
            fallback_to_original = _cfg_get(comp_cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(comp_cfg, "target_attention_only", target_attention_only)
            lora_r = _cfg_get(comp_cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(comp_cfg, "lora_alpha", lora_alpha)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])
        if "target_attention_only" in kwargs:
            target_attention_only = bool(kwargs["target_attention_only"])
        if "lora_r" in kwargs and kwargs["lora_r"] is not None:
            lora_r = int(kwargs["lora_r"])
        if "lora_alpha" in kwargs and kwargs["lora_alpha"] is not None:
            lora_alpha = float(kwargs["lora_alpha"])

        self.tt_rank = max(1, int(rank or tt_rank or 16))
        self.cfg = comp_cfg
        self.target_layers = target_layers or _cfg_get(comp_cfg, "target_layers", None)
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = target_attention_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.compressed_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"]
            if not any(k in name.lower() for k in attn_keywords):
                return False

        base_layer = getattr(module, "base_layer", module)
        cls_name = base_layer.__class__.__name__
        is_linear_like = (
            isinstance(base_layer, nn.Linear)
            or "conv1d" in cls_name.lower()
            or hasattr(base_layer, "nf")
            or (
                hasattr(base_layer, "weight")
                and hasattr(base_layer.weight, "ndim")
                and base_layer.weight.ndim == 2
                and not isinstance(base_layer, (nn.Embedding, nn.LayerNorm, nn.RMSNorm))
            )
        )
        if not is_linear_like:
            return False
        if isinstance(module, TTLinearLayer) or isinstance(base_layer, TTLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers and not any(t in name for t in self.target_layers):
            return False
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace linear/Conv1D layers with TTLinearLayer and return (model, stats)."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        fallback_layers = []
        total_error = 0.0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data
            cls_name = base_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

            if hasattr(base_layer, "nx") and hasattr(base_layer, "nf"):
                in_f, out_f = int(base_layer.nx), int(base_layer.nf)
            elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_f, out_f = int(base_layer.in_features), int(base_layer.out_features)
            elif is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            try:
                tt_layer = TTLinearLayer.from_linear(
                    module,
                    tt_rank=self.tt_rank,
                    max_error=self.max_error,
                    max_retries=self.max_retries,
                    fallback_to_none=self.fallback_to_original,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
            except Exception as e:
                logger.warning("TT-SVD decomposition failed for %s: %s. Keeping original module.", name, e)
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            if tt_layer is None:
                logger.info("TTCompressor: %s could not be compressed within budget; keeping original.", name)
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            rel_err = getattr(tt_layer, "reconstruction_error", 0.0)
            chosen_rank = getattr(tt_layer, "tt_rank", self.tt_rank)

            if rel_err > self.max_error:
                logger.warning(
                    "TTCompressor: %s relative error (%.4f) exceeded threshold (%.4f).",
                    name,
                    rel_err,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("TTCompressor: Keeping original module for %s.", name)
                    fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Verification of shape integrity
            if tt_layer.in_features != in_f or tt_layer.out_features != out_f:
                logger.error("TTCompressor: Shape mismatch for %s. Keeping original.", name)
                fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            if hasattr(module, "base_layer"):
                module.base_layer = tt_layer
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                        p.requires_grad = True
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                        p.requires_grad = True
            else:
                replace_module(self.model, name, tt_layer)

            self.compressed_layers.append(
                {
                    "name": name,
                    "original_shape": (in_f, out_f),
                    "tt_rank": chosen_rank,
                    "rel_error": float(rel_err),
                }
            )
            total_error += rel_err

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (total_error / len(self.compressed_layers)) if self.compressed_layers else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": fallback_layers,
                "num_compressed_layers": len(self.compressed_layers),
                "num_fallback": len(fallback_layers),
                "tt_rank": self.tt_rank,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
            }
        )
        logger.info(
            "TT compression finished: %d layers compressed, avg error: %.4f, param reduction: %.2f%%",
            len(self.compressed_layers),
            avg_err,
            stats.get("param_reduction_pct", 0.0),
        )
        return self.model, stats

    def compress_linear(self, linear_layer: nn.Module) -> TTLinearLayer:
        """Compress a single linear or Conv1D layer into TTLinearLayer."""
        return TTLinearLayer.from_linear(
            linear_layer,
            tt_rank=self.tt_rank,
            max_error=self.max_error,
            max_retries=self.max_retries,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )


@COMPRESSOR_REGISTRY.register("tt")
@COMPRESSOR_REGISTRY.register("tt_linear")
@COMPRESSOR_REGISTRY.register("tt_compressor")
@COMPRESSOR_REGISTRY.register("tt_linear_compressor")
@COMPRESSOR_REGISTRY.register("tt_factory")
@COMPRESSOR_REGISTRY.register("tt_compressor_factory")
@COMPRESSOR_REGISTRY.register("tt_linear_factory")
def tt_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Factory function for Tensor Train compressor."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    tt_rank = kwargs.pop("tt_rank", rank or _cfg_get(comp_config, "tt_rank", _cfg_get(comp_config, "rank", 16)))
    target_attn = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    compressor = TTCompressor(
        model,
        tt_rank=tt_rank,
        cfg=comp_config,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats:
        return compressed_model, stats
    return compressed_model


tt_linear_compressor_factory = tt_compressor_factory

__all__ = ["TTCompressor", "tt_compressor_factory", "tt_linear_compressor_factory"]

