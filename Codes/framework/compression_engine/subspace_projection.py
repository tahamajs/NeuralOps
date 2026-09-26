"""
Subspace Projection Compressor: Low-Rank Subspace Projection and Factorization.

Registered in COMPRESSOR_REGISTRY as 'subspace_projection',
'subspace_projection_compressor'.

Bug-fix map (v2):
  SP-FIX-01  is_custom_attention = False on SubspaceLinear and compressor.
  SP-FIX-02  **Added missing `import logging` and `logger`.**
  SP-FIX-03  **dtype-safe forward**: A, B, bias are cast to x_flat's dtype.
  SP-FIX-04  **LoRA transfer handles Parameter and Module** (uses
             `getattr(m, "weight", m)`).
  SP-FIX-05  **`target_layers=[]` treated as None.**
  SP-FIX-06  **`method` documented as currently only 'svd' is implemented.**
  SP-FIX-07  **`param_cap` formula corrected** (no spurious `-1`).
  SP-FIX-08  **Forward sanity check** added at end of `compress()`.
  SP-FIX-09  **Honest storage accounting** (`storage_compression_ratio`).
  SP-FIX-10  **LoRA wrapper preserved** when `module.base_layer` exists.
  SP-FIX-11  **Bias copy under `torch.no_grad()`.**
  SP-FIX-12  **Skip already-compressed SubspaceLinear modules**
             (idempotency).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)   # SP-FIX-02


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_lora_param(mod: Any) -> Optional[torch.Tensor]:
    """SP-FIX-04: extract a LoRA parameter from either a Parameter or a Module."""
    if mod is None:
        return None
    if isinstance(mod, torch.Tensor):
        return mod
    if hasattr(mod, "weight"):
        return mod.weight
    return None


# ===========================================================================
# SubspaceLinear
# ===========================================================================
class SubspaceLinear(nn.Module):
    """
    Subspace Projection Factorized Linear Layer with optional LoRA adapter.

    W ≈ B @ A, where:
      A (rank, in_features):  orthonormal / projection basis
      B (out_features, rank): coefficient expansion matrix

    Forward:
      y = ((x @ A^T) @ B^T) + bias + (lora_B @ lora_A @ x) * scaling
    """
    # SP-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features   # Conv1D compatibility
        self.nx = self.in_features    # Conv1D compatibility
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.A = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        self.B = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=dtype, device=device)
        )

        # Orthogonal init for A; xavier for B. Do it in fp32 for stability.
        with torch.no_grad():
            if dtype in (torch.float16, torch.bfloat16):
                A_fp32 = torch.empty(self.rank, in_features, dtype=torch.float32)
                nn.init.orthogonal_(A_fp32)
                self.A.data.copy_(A_fp32.to(dtype=dtype, device=device))
            else:
                nn.init.orthogonal_(self.A)

            B_fp32 = self.B.data.float()
            nn.init.xavier_uniform_(B_fp32)
            self.B.data.copy_(B_fp32.to(dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features),
                            dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r),
                            dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.B.float(), self.A.float()).to(self.A.dtype)

    def get_weight(self) -> torch.Tensor:
        return self.weight

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # SP-FIX-03: dtype-safe
        A = self.A if self.A.dtype == x_flat.dtype else self.A.to(x_flat.dtype)
        B = self.B if self.B.dtype == x_flat.dtype else self.B.to(x_flat.dtype)
        b = None
        if self.bias is not None:
            b = self.bias if self.bias.dtype == x_flat.dtype else self.bias.to(x_flat.dtype)

        x_proj = F.linear(x_flat, A)               # (K, rank)
        out = F.linear(x_proj, B, b)                # (K, out)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out.to(out.dtype)

        return out.view(*orig_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 64,
        method: str = "svd",
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "SubspaceLinear":
        """
        Project linear or Conv1D weight matrix onto top-rank subspace.
        """
        base_mod = getattr(linear, "base_layer", linear)
        raw_w = base_mod.weight.data
        cls_name = base_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous()
            out_f, in_f = W.shape
        else:
            W = raw_w
            out_f, in_f = W.shape

        bias = (
            base_mod.bias.data
            if getattr(base_mod, "bias", None) is not None
            else None
        )

        # ---------------- Rank budget (SP-FIX-07) ----------------
        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // max(1, in_f + out_f))
        r = max(1, min(int(rank), max_r, param_cap))

        # ---------------- Existing LoRA detection ----------------
        has_existing_lora = (
            hasattr(linear, "lora_A") and hasattr(linear, "lora_B")
            and _get_lora_param(linear.lora_A) is not None
            and _get_lora_param(linear.lora_B) is not None
        )
        existing_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        effective_lora_r = int(existing_r) if (has_existing_lora and existing_r > 0) else int(lora_r)
        effective_lora_alpha = float(
            getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))
        )

        subspace_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=(bias is not None),
            lora_r=effective_lora_r if (use_lora or has_existing_lora) else 0,
            lora_alpha=effective_lora_alpha,
            dtype=W.dtype,
            device=W.device,
        )

        # ---------------- SVD factorization ----------------
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        A_data = Vt[:r, :]                              # (r, in_f)
        B_data = U[:, :r] * S[:r].unsqueeze(0)          # (out_f, r)

        with torch.no_grad():
            subspace_layer.A.data.copy_(A_data.to(dtype=W.dtype, device=W.device))
            subspace_layer.B.data.copy_(B_data.to(dtype=W.dtype, device=W.device))

            # SP-FIX-11: bias under no_grad
            if bias is not None and subspace_layer.bias is not None:
                subspace_layer.bias.data.copy_(bias.to(dtype=W.dtype, device=W.device))

        # ---------------- LoRA transfer (SP-FIX-04) ----------------
        if has_existing_lora and subspace_layer.lora_A is not None and subspace_layer.lora_B is not None:
            try:
                src_A = _get_lora_param(linear.lora_A)
                src_B = _get_lora_param(linear.lora_B)
                if src_A is not None and src_B is not None:
                    if src_A.shape == subspace_layer.lora_A.shape:
                        subspace_layer.lora_A.data.copy_(
                            src_A.to(subspace_layer.lora_A.device, dtype=subspace_layer.lora_A.dtype)
                        )
                    if src_B.shape == subspace_layer.lora_B.shape:
                        subspace_layer.lora_B.data.copy_(
                            src_B.to(subspace_layer.lora_B.device, dtype=subspace_layer.lora_B.dtype)
                        )
            except Exception as e:
                logger.debug("SubspaceLinear LoRA transfer failed: %s", e)

        return subspace_layer


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("subspace_projection")
@COMPRESSOR_REGISTRY.register("subspace_projection_compressor")
class SubspaceProjectionCompressor(BaseCompressor):
    """
    Subspace Projection Compression Engine.
    """
    # SP-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        method: str = "svd",
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(),
            name="subspace_projection",
            **kwargs,
        )
        cfg_obj = cfg or kwargs.get("comp_config", None)

        self.rank = int(rank or _cfg_get(cfg_obj, "subspace_rank",
                                          _cfg_get(cfg_obj, "rank", 64)))
        self.method = str(method or _cfg_get(cfg_obj, "subspace_method", "svd"))
        if self.method != "svd":
            logger.warning(
                "SubspaceProjection: method='%s' is not implemented; using 'svd'.",
                self.method,
            )
            self.method = "svd"

        self.cfg = cfg_obj
        tl = target_layers or _cfg_get(cfg_obj, "target_layers", None)
        if tl is not None and len(tl) == 0:
            tl = None
        self.target_layers = tl

        self.target_attention_only = bool(
            target_attention_only or _cfg_get(cfg_obj, "target_attention_only", False)
        )
        self.use_lora = bool(
            use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0)
        )
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.max_error = float(_cfg_get(cfg_obj, "max_error", max_error))
        self.fallback_to_original = bool(
            _cfg_get(cfg_obj, "fallback_to_original", fallback_to_original)
        )

        self.compressed_layers: List[Dict[str, Any]] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in (
            "classifier", "score", "lm_head", "pooler", "embedding",
            "layernorm", "rmsnorm", "batchnorm",
        )):
            return False

        if isinstance(module, SubspaceLinear):
            return False

        if self.target_attention_only:
            attn_keywords = (
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "c_attn", "c_proj", "query", "key", "value",
            )
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    # ------------------------------------------------------------------
    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        from framework.core.model_utils import replace_module

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(
                    p.numel() for p in self.model.parameters()
                )

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}

        total_orig_bytes = 0
        total_comp_bytes = 0
        seen_weights: Set[int] = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_mod = getattr(module, "base_layer", module)
            w = getattr(base_mod, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = w.data
            cls_name = base_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")

            W_orig = (
                raw_w.t().contiguous().float()
                if is_conv1d else raw_w.float()
            )
            max_rank = min(W_orig.shape[0], W_orig.shape[1])
            r = min(self.rank, max_rank)

            # storage before
            orig_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(base_mod, "bias", None) is not None:
                orig_bytes += (
                    base_mod.bias.numel() * base_mod.bias.element_size()
                )
            total_orig_bytes += orig_bytes

            try:
                new_layer = SubspaceLinear.from_linear(
                    module,
                    rank=r,
                    method=self.method,
                    use_lora=self.use_lora,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
            except Exception as e:
                logger.warning("Subspace projection failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                total_comp_bytes += orig_bytes
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # error on forward path
            W_rec = new_layer.weight.float()
            if is_conv1d:
                W_orig_cmp = W_orig
                W_rec_cmp = W_rec.t().contiguous() if W_rec.shape != W_orig.shape else W_rec
            else:
                W_orig_cmp = W_orig
                W_rec_cmp = W_rec

            denom = torch.norm(W_orig_cmp, p="fro").clamp_min(1e-12)
            err = float(
                (torch.norm(W_orig_cmp - W_rec_cmp, p="fro") / denom).item()
            )
            self.layer_errors[name] = err

            if self.fallback_to_original and err > self.max_error:
                logger.warning(
                    "SubspaceProjection: err=%.4f on %s > max_error=%.4f; "
                    "keeping original.",
                    err, name, self.max_error,
                )
                self.fallback_layers.append(name)
                total_comp_bytes += orig_bytes
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # SP-FIX-10: preserve LoRA wrapper
            try:
                if hasattr(module, "base_layer"):
                    module.base_layer = new_layer
                    for pn, p in module.named_parameters():
                        if "lora_" in pn.lower() and not pn.startswith("base_layer"):
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, new_layer)
            except Exception as e:
                logger.warning("SubspaceProjection replacement failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                total_comp_bytes += orig_bytes
                continue

            self.compressed_layers.append({
                "name": name,
                "rank": new_layer.rank,
                "relative_error": float(err),
            })

            # storage after
            comp_bytes = (
                new_layer.A.numel() * new_layer.A.element_size()
                + new_layer.B.numel() * new_layer.B.element_size()
            )
            if new_layer.bias is not None:
                comp_bytes += new_layer.bias.numel() * new_layer.bias.element_size()
            if new_layer.lora_A is not None:
                comp_bytes += new_layer.lora_A.numel() * new_layer.lora_A.element_size()
            if new_layer.lora_B is not None:
                comp_bytes += new_layer.lora_B.numel() * new_layer.lora_B.element_size()
            total_comp_bytes += comp_bytes

        # ------------------------------------------------------------------
        # SP-FIX-08: Forward sanity check
        # ------------------------------------------------------------------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        cfg = getattr(self.model, "config", None)
                        hidden = getattr(cfg, "hidden_size", None) if cfg else None
                        if hidden is None:
                            first_lin = next(
                                (m for m in self.model.modules()
                                 if isinstance(m, nn.Linear)), None
                            )
                            hidden = getattr(first_lin, "in_features", 8)
                        dt = next(self.model.parameters()).dtype
                        self.model(
                            torch.randn(1, 8, hidden, device=dev, dtype=dt)
                        )
        except Exception as e:
            logger.warning("SubspaceProjection forward sanity check failed: %s", e)
            forward_check_success = False

        # ------------------------------------------------------------------
        # Stats (SP-FIX-09)
        # ------------------------------------------------------------------
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(
                p.numel() for p in self.model.parameters()
            )

        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors else 0.0
        )
        storage_ratio = (
            total_orig_bytes / max(1, total_comp_bytes)
            if total_comp_bytes > 0 else 1.0
        )
        effective_ratio = (
            total_comp_bytes / max(1, total_orig_bytes)
            if total_orig_bytes > 0 else 1.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
            "compressed_layer_count": len(self.compressed_layers),
            "num_compressed": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "compressed_layers": self.compressed_layers,
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "subspace_rank": self.rank,
            "subspace_method": self.method,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "max_error": self.max_error,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "orig_storage_bytes": total_orig_bytes,
            "comp_storage_bytes": total_comp_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "effective_parameter_ratio": round(effective_ratio, 4),
            "storage_reduction_pct": round(
                100.0 * (1.0 - effective_ratio), 2
            ),
            "forward_check_success": forward_check_success,
            "method": "subspace_projection",
        })
        self.stats = stats
        return self.model, stats

    # ------------------------------------------------------------------
    def compress_linear(self, linear: nn.Module) -> SubspaceLinear:
        return SubspaceLinear.from_linear(
            linear,
            rank=self.rank,
            method=self.method,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("subspace_projection_factory")
def subspace_projection_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    effective_rank = rank or _cfg_get(
        comp_config, "subspace_rank", _cfg_get(comp_config, "rank", 64)
    )
    method = _cfg_get(comp_config, "subspace_method", "svd")
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "fallback_to_original", True),
    )

    compressor = SubspaceProjectionCompressor(
        model,
        rank=effective_rank,
        method=method,
        cfg=comp_config,
        target_attention_only=target_attention_only,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SubspaceLinear",
    "SubspaceProjectionCompressor",
    "subspace_projection_factory",
]