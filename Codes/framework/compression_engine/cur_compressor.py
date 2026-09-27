"""
CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import adaptive_max_error
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

def compute_leverage_scores(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact statistical leverage scores from the top-k singular vectors of W.
    W: shape (out_features, in_features)
    Returns:
        (col_scores, row_scores) normalized to valid probability distributions summing to 1.0.
    """
    m, n = W.shape
    k = max(1, min(rank, m, n))
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    # Row leverage scores: norm of rows of U_k
    row_scores = (U[:, :k] ** 2).sum(dim=1) / float(k)
    # Column leverage scores: norm of cols of V_k (rows of Vt_k)
    col_scores = (Vt[:k, :] ** 2).sum(dim=0) / float(k)
    # Normalize leverage scores
    row_scores = row_scores / row_scores.sum().clamp_min(1e-12)
    col_scores = col_scores / col_scores.sum().clamp_min(1e-12)
    return col_scores, row_scores


def compute_energy_scores(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Frobenius energy sampling probabilities for columns and rows.
    """
    frob_sq = (W.float() ** 2).sum().clamp_min(1e-12)
    col_scores = (W.float() ** 2).sum(dim=0) / frob_sq
    row_scores = (W.float() ** 2).sum(dim=1) / frob_sq
    return col_scores, row_scores


def select_indices(scores: torch.Tensor, k: int, deterministic: bool = True) -> torch.Tensor:
    """
    Select k indices based on scores (either deterministic top-k or multinomial sampling).
    """
    n = scores.numel()
    k = min(k, n)
    if deterministic:
        _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
        return idx
    else:
        probs = (scores / scores.sum().clamp_min(1e-12)).clamp_min(1e-8)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=k, replacement=False)
        return idx


def tikhonov_pinv(A: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    """
    Computes Tikhonov regularized pseudo-inverse of matrix A: (A^T A + reg * I)^(-1) A^T
    or via SVD regularized inversion for numerical stability.
    """
    try:
        u, s, vt = torch.linalg.svd(A.float(), full_matrices=False)
        eps = reg * s.max().clamp_min(1e-6)
        s_inv = s / (s**2 + eps**2)
        return torch.matmul(vt.T, torch.matmul(torch.diag(s_inv), u.T)).to(dtype=A.dtype)
    except Exception:
        return torch.linalg.pinv(A.float(), rcond=reg).to(dtype=A.dtype)


class CURLinear(nn.Module):
    """
    Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W

    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        method: str = "leverage",
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.method = method
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / max(1, self.lora_r)) if self.lora_r > 0 else 1.0

        # C: (out_features, rank), U: (rank, rank), R: (rank, in_features)
        self.C = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(self.rank, self.rank, dtype=dtype, device=device))
        self.R = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.C, -bound, bound)
        nn.init.eye_(self.U)
        nn.init.uniform_(self.R, -bound, bound)

        if freeze_skeletons:
            self.C.requires_grad = False
            self.R.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_r, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_r, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W_core = torch.matmul(torch.matmul(self.C, self.U), self.R)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            W_core = W_core + (torch.matmul(self.lora_B, self.lora_A) * self.scaling)
        return W_core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Factorized forward pass:
        # x_r = x @ R^T: (K, rank)
        x_r = F.linear(x_flat, self.R)
        # x_ru = x_r @ U^T: (K, rank)
        x_ru = F.linear(x_r, self.U)
        # out = x_ru @ C^T + bias: (K, out_features)
        out = F.linear(x_ru, self.C, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize reconstructed weight matrix W = C @ U @ R."""
        return self.weight

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        rank: int = 64,
        method: str = "leverage",
        deterministic: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
    ) -> "CURLinear":
        """
        Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.
        """
        target_mod = getattr(linear, "base_layer", getattr(linear, "linear", linear))
        raw_w = target_mod.weight.data
        bias = target_mod.bias.data if getattr(target_mod, "bias", None) is not None else None
        cls_name = target_mod.__class__.__name__.lower()
        wrapper_name = linear.__class__.__name__.lower()

        is_conv1d = "conv1d" in cls_name or "conv1d" in wrapper_name or hasattr(target_mod, "nf") or hasattr(linear, "nf")
        if is_conv1d:
            W = raw_w.t().contiguous()
        else:
            W = raw_w.contiguous()

        out_features, in_features = W.shape
        r = max(1, min(rank, out_features, in_features))

        cur_layer = cls(
            in_features=in_features,
            out_features=out_features,
            rank=r,
            bias=bias is not None,
            method=method,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_skeletons=freeze_skeletons,
            dtype=W.dtype,
            device=W.device,
        )

        W_float = W.float()
        if method == "leverage":
            col_scores, row_scores = compute_leverage_scores(W_float, rank=r)
        elif method == "energy":
            col_scores, row_scores = compute_energy_scores(W_float)
        else:
            col_scores = torch.ones(in_features, device=W.device)
            row_scores = torch.ones(out_features, device=W.device)

        col_idx = select_indices(col_scores, k=r, deterministic=deterministic)
        row_idx = select_indices(row_scores, k=r, deterministic=deterministic)

        # C = W[:, col_idx] (out_features, r)
        # R = W[row_idx, :] (r, in_features)
        C = W_float[:, col_idx]
        R = W_float[row_idx, :]

        # Optimal core matrix: U = C^+ @ W @ R^+
        # This is the exact least-squares solution argmin_U ||W - C U R||_F
        try:
            C_pinv = torch.linalg.pinv(C, rcond=1e-4)
            R_pinv = torch.linalg.pinv(R, rcond=1e-4)
            U = torch.matmul(torch.matmul(C_pinv, W_float), R_pinv)
        except Exception:
            W_int = W_float[row_idx][:, col_idx]
            U = torch.linalg.pinv(W_int, rcond=1e-3)

        cur_layer.C.data.copy_(C.to(dtype=W.dtype))
        cur_layer.U.data.copy_(U.to(dtype=W.dtype))
        cur_layer.R.data.copy_(R.to(dtype=W.dtype))

        if bias is not None and cur_layer.bias is not None:
            cur_layer.bias.data.copy_(bias)

        return cur_layer


CURLowRankLinear = CURLinear


logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from dict, object, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("cur_compressor")
class CURCompressor(BaseCompressor):
    """
    Compressor that replaces eligible Linear and Conv1D layers with CURLinear.

    Args:
        model: PyTorch model.
        rank: Target rank for skeleton column and row selection (default 64).
        min_rank: Minimum allowable rank for compression (default 8).
        method: Selection method ('leverage' or 'energy', default 'leverage').
        deterministic: Deterministic top-k selection if True (default True).
        max_error: Maximum relative Frobenius reconstruction error allowed (default 0.20).
        fallback_to_original: If True, retains original layer when reconstruction error exceeds max_error (default True).
        max_retries: Number of adaptive rank retries when error threshold is exceeded (default 2).
        lora_r: LoRA adapter rank if adding fine-tuning adapters (default 0).
        lora_alpha: LoRA scaling factor (default 1.0).
        freeze_skeletons: If True, freeze C and R parameters during post-compression training (default False).
        target_attention_only: If True, restrict compression to attention projections only.
        target_names: Optional list of layer name substrings to target.
        exclude_names: Optional list of layer name substrings to exclude.
        cfg: Optional configuration object.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        min_rank: int = 8,
        method: str = "leverage",
        deterministic: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
        target_attention_only: bool = False,
        target_names: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur", **kwargs)
        c_rank = _cfg_get(cfg, "cur_rank", _cfg_get(cfg, "rank", rank))
        c_min_rank = _cfg_get(cfg, "cur_min_rank", _cfg_get(cfg, "min_rank", min_rank))
        c_method = _cfg_get(cfg, "cur_method", _cfg_get(cfg, "method", method))
        c_det = _cfg_get(cfg, "cur_deterministic", _cfg_get(cfg, "deterministic", deterministic))
        c_max_error = _cfg_get(cfg, "cur_max_error", _cfg_get(cfg, "max_error", max_error))
        c_fallback = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
        c_lora_r = _cfg_get(cfg, "cur_lora_r", _cfg_get(cfg, "lora_r", lora_r))
        c_lora_alpha = _cfg_get(cfg, "cur_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha))
        c_freeze = _cfg_get(cfg, "freeze_skeletons", freeze_skeletons)
        c_target_attn = _cfg_get(cfg, "cur_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))

        if c_max_error is None or c_max_error == 0.20:
            c_max_error = adaptive_max_error(c_rank, 768, 768, "distilbert")

        self.rank = int(c_rank) if c_rank is not None else 64
        self.min_rank = int(c_min_rank) if c_min_rank is not None else 8
        self.method = str(c_method)
        self.deterministic = bool(c_det)
        self.max_error = float(c_max_error)
        self.fallback_to_original = bool(c_fallback)
        self.max_retries = max(0, int(max_retries))
        self.lora_r = int(c_lora_r or 0)
        self.lora_alpha = float(c_lora_alpha or 1.0)
        self.freeze_skeletons = bool(c_freeze)
        self.target_attention_only = bool(c_target_attn)
        self.cfg = cfg
        self.target_names = target_names or _cfg_get(cfg, "cur_target_names", _cfg_get(cfg, "target_layers", None))
        self.exclude_names = exclude_names or _cfg_get(cfg, "cur_exclude_names", [
            "lm_head", "classifier", "head", "score", "pooler", "embedding", "router", "gate"
        ])

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Identify candidate linear/Conv1D/LoRALinear layers."""
        cls_name = module.__class__.__name__.lower()
        has_base = hasattr(module, "base_layer")
        base_cls = module.base_layer.__class__.__name__.lower() if has_base else ""

        is_linear = isinstance(module, nn.Linear) or "linear" in cls_name
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_lora = has_base and (
            isinstance(module.base_layer, nn.Linear)
            or "conv1d" in base_cls
            or hasattr(module.base_layer, "nf")
        )

        if not (is_linear or is_conv1d or is_lora):
            return False

        if isinstance(module, CURLinear) or (has_base and isinstance(module.base_layer, CURLinear)):
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in self.exclude_names):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False

        if self.target_names and not any(t in name for t in self.target_names):
            return False

        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace eligible Linear and Conv1D layers with CURLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        visited_weight_ids: Set[int] = set()

        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                if self.target_attention_only:
                    is_lin_like = isinstance(module, nn.Linear) or "conv1d" in module.__class__.__name__.lower() or hasattr(module, "base_layer")
                    if is_lin_like:
                        for p in module.parameters():
                            p.requires_grad = False
                continue

            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            # Check tied weights
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                for p in module.parameters():
                    p.requires_grad = False
                continue
            weight_id = id(target_mod.weight)
            if weight_id in visited_weight_ids:
                logger.info("CURCompressor: Skipping tied weight layer %s", name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            raw_w = target_mod.weight.data
            orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = orig_w.shape
            max_rank = min(out_f, in_f)

            if max_rank < self.min_rank:
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # Parameter reduction check: r*(in_f + out_f + r) < in_f * out_f
            target_r = min(self.rank, max_rank)
            while target_r >= self.min_rank and (target_r * (in_f + out_f + target_r)) >= 0.85 * (in_f * out_f):
                target_r = int(target_r * 0.8)

            if target_r < self.min_rank:
                logger.debug("CURCompressor: Skipping %s as target rank %d violates parameter reduction constraint", name, target_r)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            cur_r = target_r
            best_cur_layer = None
            best_err = float("inf")
            best_r = cur_r
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)
            effective_lora_r = 0 if has_lora else self.lora_r

            for _ in range(self.max_retries + 1):
                try:
                    candidate = CURLinear.from_linear(
                        target_mod,
                        rank=cur_r,
                        method=self.method,
                        deterministic=self.deterministic,
                        lora_r=effective_lora_r,
                        lora_alpha=self.lora_alpha,
                        freeze_skeletons=self.freeze_skeletons,
                    )
                    recon_w = candidate.get_weight().float()
                    err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()

                    if err < best_err:
                        best_err = err
                        best_cur_layer = candidate
                        best_r = cur_r

                    if err <= self.max_error:
                        break

                    next_r = min(int(cur_r * 1.5), max_rank - 1)
                    if next_r <= cur_r or (next_r * (in_f + out_f + next_r)) >= 0.85 * (in_f * out_f):
                        break
                    cur_r = next_r
                except Exception as ex:
                    logger.warning("CURCompressor: Failed CUR decomposition on %s at rank %d: %s", name, cur_r, ex)
                    break

            if best_err > self.max_error:
                logger.warning(
                    "CURCompressor: Layer %s error %.4f exceeded max_error %.4f",
                    name, best_err, self.max_error
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    # Freeze fallback layers
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if best_cur_layer is not None:
                new_layer = best_cur_layer
                compressed_layer = new_layer
                # Preserve existing LoRA adapter weights if the source module has them
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                    if hasattr(new_layer, 'lora_A') and new_layer.lora_A is not None:
                        if hasattr(src_lA, 'data') and src_lA.shape == new_layer.lora_A.shape:
                            new_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(new_layer, 'lora_B') and new_layer.lora_B is not None:
                        if hasattr(src_lB, 'data') and src_lB.shape == new_layer.lora_B.shape:
                            new_layer.lora_B.data.copy_(src_lB.data)

                if hasattr(module, 'base_layer'):
                    module.base_layer = compressed_layer  # put compressed layer inside wrapper
                    # module itself (LoRALinear wrapper) remains the replacement
                    new_module = module
                else:
                    new_module = compressed_layer

                if hasattr(module, 'base_layer'):
                    for p_name, p in module.named_parameters():
                        if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                            p.requires_grad = True
                else:
                    if name == "":
                        self.model = new_module
                    else:
                        replace_module(self.model, name, new_module)

                self.compressed_layers.append(name)
                self.layer_errors[name] = float(best_err)
                self.layer_ranks[name] = int(best_r)
                visited_weight_ids.add(weight_id)

                total_orig_decomp_params += (in_f * out_f)
                total_comp_decomp_params += (best_r * (in_f + out_f + best_r))

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        effective_param_ratio = (
            total_comp_decomp_params / total_orig_decomp_params if total_orig_decomp_params > 0 else 1.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "cur_method": self.method,
                "cur_rank": self.rank,
                "min_rank": self.min_rank,
                "deterministic": self.deterministic,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_attention_only": self.target_attention_only,
                "effective_parameter_ratio": round(effective_param_ratio, 4),
            }
        )
        logger.info(
            "Compressor cur finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats

    def compress_linear(self, linear: Union[nn.Linear, Any]) -> CURLinear:
        """Compress a single linear layer into CURLinear."""
        return CURLinear.from_linear(
            linear,
            rank=self.rank,
            method=self.method,
            deterministic=self.deterministic,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            freeze_skeletons=self.freeze_skeletons,
        )


@COMPRESSOR_REGISTRY.register("cur")
@COMPRESSOR_REGISTRY.register("cur_linear")
@COMPRESSOR_REGISTRY.register("cure")
@COMPRESSOR_REGISTRY.register("cur_factory")
@COMPRESSOR_REGISTRY.register("cur_linear_factory")
@COMPRESSOR_REGISTRY.register("cure_factory")
@COMPRESSOR_REGISTRY.register("cur_compressor_factory")
def cur_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR compression."""
    effective_rank = kwargs.pop("rank", rank or _cfg_get(comp_config, "cur_rank", _cfg_get(comp_config, "rank", 64)))
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "min_rank", 8))
    method = kwargs.pop("method", _cfg_get(comp_config, "cur_method", _cfg_get(comp_config, "method", "leverage")))
    deterministic = kwargs.pop("deterministic", _cfg_get(comp_config, "deterministic", True))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "cur_max_error", _cfg_get(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "cur_lora_r", _cfg_get(comp_config, "lora_r", 0)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "cur_lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "cur_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)))
    target_names = kwargs.pop("target_names", _cfg_get(comp_config, "cur_target_names", None))
    exclude_names = kwargs.pop("exclude_names", _cfg_get(comp_config, "cur_exclude_names", None))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = CURCompressor(
        model,
        rank=effective_rank,
        min_rank=min_rank,
        method=method,
        deterministic=deterministic,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        target_names=target_names,
        exclude_names=exclude_names,
        cfg=comp_config,
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
