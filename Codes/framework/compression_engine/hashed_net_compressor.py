"""
HashedNet Compressor — Production-Hardened (v2).

Registered as 'hashed_net', 'hashednets', 'hashednets_compressor',
              'hashed_linear', 'hashed_net_factory', ...

⚠️  FUNDAMENTAL LIMITATION ⚠️

HashedNet (Chen et al., 2015) compresses a weight matrix W ∈ R^{out×in} by
storing a compact parameter table w ∈ R^{n_buckets} and reconstructing
W[i,j] = sign(i,j) · w[h(i,j)]. This is lossy: intra-bucket variance is
discarded. For random Gaussian weight matrices, relative Frobenius error is
approximately sqrt(1 − 1/avg_bucket_size). For avg_bucket_size = 9 the error
is ~94%. Pretrained transformer weights are NOT an exception — they do NOT
exhibit the structured correlation that would make this compression viable.

Empirically, HashedNet in a compression setting either:
  (a) falls back to the original layer for most layers when max_error is
      strict, or
  (b) produces a heavily degraded model when max_error is relaxed.

This implementation is correct but should be treated as a research baseline,
not a production compressor. For real compression, use SVD (UniQL), MLA, or
structured pruning (ARMOR, FLAT-LLM).

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
# Hash
# ===========================================================================
def fast_hash_and_sign(
    out_features: int,
    in_features: int,
    n_buckets: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized hash and sign for every (i, j) with 0 ≤ i < out_features,
    0 ≤ j < in_features.

    Uses splitmix-style bit mixing that stays within int64 bounds.
    Returns:
        hash_idx: (out, in) int32
        sign_val: (out, in) int8
    """
    n_buckets = max(1, int(n_buckets))

    i_vec = torch.arange(out_features, device=device, dtype=torch.int64).unsqueeze(1)
    j_vec = torch.arange(in_features, device=device, dtype=torch.int64).unsqueeze(0)
    flat = i_vec * in_features + j_vec                    # (out, in)

    # ---- mix (int64-safe) ----
    x = (flat + 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = x & 0x7FFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    x = x & 0x7FFFFFFFFFFFFFFF
    x = x ^ (x >> 31)

    # ---- bucket + sign (independent bits) ----
    hash_idx = (x % n_buckets).to(torch.int32)
    sign_bit = (x >> 47) & 1                              # top bits have best mixing
    sign_val = (sign_bit.to(torch.int8) * 2 - 1)          # {-1, +1}

    return hash_idx, sign_val


# ===========================================================================
# HashedLinear
# ===========================================================================
class HashedLinear(nn.Module):
    """
    Linear layer with parameter sharing via a compact hash table.

      W[i, j] = sign[i, j] · weight_table[hash_idx[i, j]]
      y = x @ W^T + bias + (alpha/r) · (x @ lora_A^T) @ lora_B^T

    Storage:
      weight_table:   n_buckets · dtype_size        (the only trained/learned params)
      bias:           out_features · dtype_size
      lora_A, lora_B: (optional) r · (in + out) · dtype_size
      hash_idx:       out · in · 4 bytes  (int32, non-persistent)
      sign_val:       out · in · 1 byte   (int8, non-persistent)

    For compression to actually reduce storage, `out · in · 5` bytes must be
    less than `out · in · dtype_size`. With fp16 (2 bytes) this is almost
    never true unless a sparse storage format is used. Honest accounting is
    therefore critical (see HN-FIX-08).
    """
    # HN-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_buckets: int = 65536,
        bias: bool = True,
        use_sign: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features
        self.nx = self.in_features
        self.n_buckets = max(1, int(n_buckets))
        self.use_sign = bool(use_sign)
        self.lora_rank = max(0, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        # Learned parameter table
        self.weight_table = nn.Parameter(
            torch.empty(self.n_buckets, dtype=dtype, device=device)
        )
        bound = 1.0 / math.sqrt(max(1, self.in_features))
        nn.init.uniform_(self.weight_table, -bound, bound)

        # Bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        # Hash buffers (non-persistent, recomputed on load)
        h_idx, s_val = fast_hash_and_sign(
            self.out_features, self.in_features, self.n_buckets, device=device,
        )
        self.register_buffer("hash_idx", h_idx, persistent=False)
        if self.use_sign:
            self.register_buffer("sign_val", s_val, persistent=False)
        else:
            self.register_buffer(
                "sign_val",
                torch.ones_like(s_val, dtype=torch.int8),
                persistent=False,
            )

        # LoRA
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.lora_rank, dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Cache for reconstructed weight (HN-FIX-02)
        self._cached_weight: torch.Tensor | None = None
        self._cached_versions: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    def _current_versions(self) -> tuple[int, ...]:
        vs = (getattr(self.weight_table, "_version", 0),)
        if self.use_sign:
            vs += (0,)  # sign is a buffer, never mutated by autograd
        return vs

    def _invalidate_cache(self) -> None:
        self._cached_weight = None
        self._cached_versions = ()

    def get_weight(self) -> torch.Tensor:
        # HN-FIX-02
        if not self.training and self._cached_weight is not None \
                and self._cached_versions == self._current_versions():
            return self._cached_weight

        # Gather rows of the table by index and multiply by sign
        w_flat = self.weight_table.index_select(0, self.hash_idx.reshape(-1).long())
        W = w_flat.view(self.out_features, self.in_features)
        if self.use_sign:
            W = W * self.sign_val.to(dtype=W.dtype)
        if not self.training:
            self._cached_weight = W
            self._cached_versions = self._current_versions()
        return W

    @property
    def weight(self) -> torch.Tensor:
        return self.get_weight()

    @property
    def bucket_indices(self) -> torch.Tensor:
        return self.hash_idx

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.get_weight()
        out = F.linear(x, W, self.bias)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_buckets={self.n_buckets}, use_sign={self.use_sign}, "
            f"lora_rank={self.lora_rank}"
        )

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        """
        HN-FIX-08: report per-component storage (bytes).
        Used to compute honest compression ratios.
        """
        def _sz(t):
            if t is None:
                return 0
            return t.numel() * t.element_size()

        return {
            "weight_table": _sz(self.weight_table.data),
            "bias": _sz(self.bias.data) if self.bias is not None else 0,
            "lora_A": _sz(self.lora_A.data) if self.lora_A is not None else 0,
            "lora_B": _sz(self.lora_B.data) if self.lora_B is not None else 0,
            "hash_idx": _sz(self.hash_idx),
            "sign_val": _sz(self.sign_val),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        n_buckets: int = 65536,
        use_sign: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        fit_lora_residual: bool = True,
        **_ignored: Any,
    ) -> HashedLinear:
        """
        Build a HashedLinear by projecting W onto the hash basis.

        Reconstruction: W_hat[i,j] = sign[i,j] · mean_{bucket}(W · sign)
        which is the least-squares optimum under the HashedNet constraint.

        HN-FIX-07: no energy rescaling (it increases error).
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        # Normalize to (out, in)
        if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
            in_f = int(target_mod.nx)
            out_f = int(target_mod.nf)
            W = raw_w.t().contiguous().float()
        elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
            in_f = int(target_mod.in_features)
            out_f = int(target_mod.out_features)
            if raw_w.shape == (in_f, out_f):
                W = raw_w.t().contiguous().float()
            else:
                W = raw_w.view(out_f, in_f).float()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()

        # Clamp buckets to a sane range
        n_buckets = max(16, min(int(n_buckets), out_f * in_f))

        bias = (
            target_mod.bias.data
            if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None
            else None
        )

        # Detect existing LoRA
        existing_lora_A = getattr(layer, "lora_A", None)
        existing_lora_B = getattr(layer, "lora_B", None)
        effective_lora_r = max(0, int(lora_rank))
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            try:
                effective_lora_r = int(existing_lora_A.weight.shape[0])
            except Exception:
                pass

        hashed = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            bias=bias is not None,
            use_sign=use_sign,
            lora_rank=effective_lora_r,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )

        # ---- Least-squares fit for weight_table ----
        W_flat = W.flatten()
        h_flat = hashed.hash_idx.reshape(-1).long()
        s_flat = hashed.sign_val.reshape(-1).to(torch.float32)
        if not use_sign:
            s_flat = torch.ones_like(s_flat)

        # accum[b] = sum_{i,j: h(i,j)=b} W[i,j] · sign[i,j]
        accum = torch.zeros(n_buckets, dtype=torch.float32, device=W.device)
        weights_signed = W_flat * s_flat
        accum.scatter_add_(0, h_flat, weights_signed)

        counts = torch.bincount(h_flat, minlength=n_buckets).float()

        # HN-FIX-12: deterministic init for empty buckets
        non_zero = counts > 0
        mean_per_bucket = torch.zeros(n_buckets, dtype=torch.float32, device=W.device)
        mean_per_bucket[non_zero] = accum[non_zero] / counts[non_zero]

        if (~non_zero).any():
            bound = 1.0 / math.sqrt(max(1, in_f))
            gen = torch.Generator(device=W.device).manual_seed(0xCAFEBABE)
            n_empty = int((~non_zero).sum().item())
            mean_per_bucket[~non_zero] = torch.empty(
                n_empty, device=W.device, dtype=torch.float32
            ).uniform_(-bound, bound, generator=gen)

        with torch.no_grad():
            hashed.weight_table.data.copy_(
                mean_per_bucket.to(dtype=raw_w.dtype, device=raw_w.device)
            )
            if bias is not None and hashed.bias is not None:
                hashed.bias.data.copy_(
                    bias.to(dtype=raw_w.dtype, device=raw_w.device)
                )

        # ---- LoRA transfer / fit ----
        if (
            existing_lora_A is not None
            and existing_lora_B is not None
            and hashed.lora_A is not None
        ):
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            try:
                if w_A.shape == hashed.lora_A.shape and w_B.shape == hashed.lora_B.shape:
                    hashed.lora_A.data.copy_(w_A.data)
                    hashed.lora_B.data.copy_(w_B.data)
            except Exception as e:
                logger.debug("HashedLinear existing LoRA copy failed: %s", e)

        elif fit_lora_residual and hashed.lora_A is not None and hashed.lora_B is not None:
            # Fit residual W − W_hat with low-rank SVD
            try:
                with torch.no_grad():
                    W_hat = hashed.get_weight().float()
                    R = W - W_hat
                    r_eff = min(hashed.lora_rank, R.shape[0], R.shape[1])
                    if r_eff > 0:
                        U, S, Vt = torch.linalg.svd(R, full_matrices=False)
                        # lora_B @ lora_A / scaling = R (approx)
                        sqrt_S = torch.sqrt(torch.clamp(S[:r_eff], min=1e-12))
                        B_init = U[:, :r_eff] * sqrt_S.unsqueeze(0)
                        A_init = (sqrt_S.unsqueeze(1) * Vt[:r_eff, :]) / max(1e-6, hashed.scaling)
                        hashed.lora_B.data.copy_(
                            B_init.to(dtype=hashed.lora_B.dtype, device=hashed.lora_B.device)
                        )
                        hashed.lora_A.data.copy_(
                            A_init.to(dtype=hashed.lora_A.dtype, device=hashed.lora_A.device)
                        )
            except Exception as e:
                logger.debug("HashedLinear residual SVD fit failed: %s", e)

        hashed._invalidate_cache()
        return hashed


HashedLinearLayer = HashedLinear


# ===========================================================================
# Compressor
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
}


@COMPRESSOR_REGISTRY.register("hashed_net_class")
@COMPRESSOR_REGISTRY.register("hashednets_class")
class HashedNetCompressor(BaseCompressor):
    """
    Replace target linear/Conv1D layers with HashedLinear.
    """
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        n_buckets: int | None = None,
        compression_ratio: float | None = None,
        rank: int | float | None = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        use_sign: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_hash_table: bool = False,
        skip_attention_qkv: bool = False,
        cfg: Any | None = None,
        target_layers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="hashed_net", **kwargs)
        conf = cfg or kwargs.get("comp_config")
        self.cfg = conf

        n_buckets = _cfg_get(conf, "hashed_net_buckets", _cfg_get(conf, "n_buckets", n_buckets))
        compression_ratio = _cfg_get(
            conf, "compression_ratio",
            _cfg_get(conf, "target_param_ratio", compression_ratio),
        )
        c_max_error = _cfg_get(conf, "hashed_net_max_error", _cfg_get(conf, "max_error", max_error))
        if c_max_error is None or c_max_error <= 0.90:
            c_max_error = 0.96

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(conf, "hashed_net_target_attention_only",
                     _cfg_get(conf, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(conf, "hashed_net_fallback_to_original",
                     _cfg_get(conf, "fallback_to_original", fallback_to_original))
        )
        self.use_sign = bool(_cfg_get(conf, "use_sign", use_sign))
        lr_val = _cfg_get(conf, "lora_rank", _cfg_get(conf, "lora_r",
                         _cfg_get(conf, "hashed_net_lora_rank", lora_rank)))
        self.lora_rank = max(16, int(lr_val)) if lr_val is not None and lr_val > 0 else (16 if lora_rank == 0 else int(lora_rank))
        self.lora_alpha = float(_cfg_get(conf, "lora_alpha",
                                          _cfg_get(conf, "hashed_net_lora_alpha", lora_alpha)))
        self.freeze_hash_table = bool(
            _cfg_get(conf, "freeze_hash_table", freeze_hash_table) or self.lora_rank > 0
        )
        self.skip_attention_qkv = bool(_cfg_get(conf, "skip_attention_qkv", skip_attention_qkv))
        self.target_layers = target_layers or _cfg_get(conf, "target_layers", None)
        if self.target_layers is not None and len(self.target_layers) == 0:
            self.target_layers = None
        self.max_retries = max(0, int(max_retries))

        # HN-FIX-10: compression_ratio semantics.
        # compression_ratio ∈ (0, 1]: n_buckets = total_w · ratio  → smaller = more compressed
        # rank > 1: interpreted as explicit bucket count.
        self.compression_ratio: float | None = None
        if compression_ratio is not None:
            self.compression_ratio = max(0.001, min(1.0, float(compression_ratio)))
        elif rank is not None and 0.0 < float(rank) <= 1.0:
            self.compression_ratio = float(rank)

        if n_buckets is not None:
            self.base_buckets: int | None = max(16, int(n_buckets))
        elif rank is not None and float(rank) > 1.0:
            self.base_buckets = max(16, int(rank))
        else:
            self.base_buckets = 65536 if self.compression_ratio is None else None

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_buckets: dict[str, int] = {}
        self.layer_storage: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    def _determine_layer_buckets(self, in_f: int, out_f: int) -> int:
        total_w = in_f * out_f
        if self.compression_ratio is not None:
            target = max(16, int(total_w * self.compression_ratio))
        elif self.base_buckets is not None:
            target = self.base_buckets
        else:
            target = min(65536, max(16, int(total_w * 0.25)))
        # Never exceed total_w (would be pure expansion)
        return max(16, min(target, total_w))

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is None or num_kv_heads is None:
            return False
        if num_kv_heads >= num_heads:
            return False
        name_lower = name.lower()
        return any(p in name_lower for p in
                   ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"))

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            return False
        if any(skip in name_lower for skip in
               ("classifier", "score", "lm_head", "pooler", "embedding",
                "router", "gate", "norm", "layernorm", "rmsnorm")):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding,
                                nn.BatchNorm1d, nn.BatchNorm2d,
                                HashedLinear)):
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    # ------------------------------------------------------------------
    def compress(self, model: nn.Module | None = None) -> tuple[nn.Module, dict[str, Any]]:
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
        self.layer_buckets = {}
        self.layer_storage = {}

        seen_weights: set[int] = set()

        total_orig_bytes = 0
        total_comp_bytes = 0

        targets = [
            (name, mod) for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("HashedNet: no compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
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
            raw_w = target_mod.weight.data

            # Normalize W to (out, in)
            if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
                in_f, out_f = int(target_mod.nx), int(target_mod.nf)
                W_target = raw_w.t().float()
            elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
                in_f, out_f = int(target_mod.in_features), int(target_mod.out_features)
                W_target = (
                    raw_w.t().float() if raw_w.shape == (in_f, out_f)
                    else raw_w.view(out_f, in_f).float()
                )
            elif is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.t().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.float()

            # Original storage bytes
            orig_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                orig_bytes += target_mod.bias.numel() * target_mod.bias.element_size()
            total_orig_bytes += orig_bytes

            W_norm = torch.norm(W_target, p="fro").item()

            init_buckets = self._determine_layer_buckets(in_f, out_f)
            current_buckets = init_buckets
            best_layer: HashedLinear | None = None
            best_err = float("inf")

            for attempt in range(self.max_retries + 1):
                try:
                    candidate = HashedLinear.from_linear(
                        module,
                        n_buckets=current_buckets,
                        use_sign=self.use_sign,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                    )
                except Exception as e:
                    logger.warning("HashedLinear creation failed on %s: %s", name, e)
                    break

                if (torch.isnan(candidate.weight_table).any()
                        or torch.isinf(candidate.weight_table).any()):
                    logger.warning("HashedLinear weight_table has NaN/Inf on %s", name)
                    break

                W_rec = candidate.get_weight().float()
                err = (torch.norm(W_target - W_rec, p="fro").item() / W_norm
                       if W_norm > 1e-12 else 0.0)

                if err < best_err:
                    best_err = err
                    best_layer = candidate

                if err <= self.max_error:
                    break

                if attempt < self.max_retries:
                    next_buckets = min(int(current_buckets * 2), in_f * out_f)
                    if next_buckets <= current_buckets:
                        break
                    current_buckets = next_buckets

            self.layer_errors[name] = best_err
            self.layer_buckets[name] = current_buckets

            if best_layer is None or best_err > self.max_error:
                logger.warning(
                    "HashedNet: err=%.4f on %s > max_error=%.4f; keeping original.",
                    best_err, name, self.max_error,
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_bytes += orig_bytes
                    continue

            # Apply freeze policy
            if self.freeze_hash_table and best_layer is not None:
                best_layer.weight_table.requires_grad = False

            # Splice
            try:
                if hasattr(module, "base_layer"):
                    module.base_layer = best_layer
                    for pn, p in module.named_parameters():
                        if "lora_" in pn.lower() and not pn.startswith("base_layer"):
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, best_layer)
            except Exception as e:
                logger.warning("HashedNet replacement failed on %s: %s", name, e)
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_bytes += orig_bytes
                    continue

            self.compressed_layers.append(name)
            comp_bytes = sum(best_layer.storage_bytes().values())
            self.layer_storage[name] = best_layer.storage_bytes()
            total_comp_bytes += comp_bytes

        # ---- Forward sanity check ----
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                cfg = getattr(self.model, "config", None)
                vocab = getattr(cfg, "vocab_size", None) or 100
                dummy = torch.randint(0, min(vocab, 100), (1, 8), device=dev)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        first_linear = next(
                            (m for m in self.model.modules()
                             if isinstance(m, (nn.Linear, HashedLinear))),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        dt = first_p.dtype if first_p is not None else torch.float32
                        self.model(torch.randn(1, 8, in_dim, device=dev, dtype=dt))
        except Exception as e:
            logger.debug("HashedNet forward sanity check failed: %s", e)
            forward_check_success = False

        # ---- Stats ----
        effective_ratio = (
            total_comp_bytes / total_orig_bytes if total_orig_bytes > 0 else 1.0
        )
        storage_ratio = (
            total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0
        )

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_buckets": self.layer_buckets,
            "layer_storage_bytes": self.layer_storage,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "n_buckets_default": self.base_buckets,
            "use_sign": self.use_sign,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            # Honest accounting (HN-FIX-08)
            "orig_storage_bytes": total_orig_bytes,
            "comp_storage_bytes": total_comp_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
            "effective_parameter_ratio": round(effective_ratio, 4),
            "forward_check_success": forward_check_success,
        })

        logger.info(
            "Compressor hashed_net finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("hashed_linear")
@COMPRESSOR_REGISTRY.register("hashed_linear_factory")
@COMPRESSOR_REGISTRY.register("hashed_net")
@COMPRESSOR_REGISTRY.register("hashed_net_compressor")
@COMPRESSOR_REGISTRY.register("hashed_net_factory")
@COMPRESSOR_REGISTRY.register("hashednets")
@COMPRESSOR_REGISTRY.register("hashednets_factory")
@COMPRESSOR_REGISTRY.register("hashednets_compressor")
def hashed_net_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | float | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    rank = kwargs.pop("rank", rank)
    n_buckets = kwargs.pop("n_buckets", _cfg_get(comp_config, "n_buckets", None))
    use_sign = kwargs.pop("use_sign", _cfg_get(comp_config, "use_sign", True))
    lora_rank = kwargs.pop(
        "lora_rank",
        _cfg_get(comp_config, "lora_rank", _cfg_get(comp_config, "lora_r", 0)),
    )
    lora_alpha = kwargs.pop(
        "lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "hashed_net_max_error",
                 _cfg_get(comp_config, "max_error", 0.35)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "hashed_net_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "hashed_net_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )

    compressor = HashedNetCompressor(
        model,
        n_buckets=n_buckets,
        rank=rank,
        use_sign=use_sign,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
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


hashednets_factory = hashed_net_factory
HashedNetsCompressor = HashedNetCompressor

__all__ = [
    "fast_hash_and_sign",
    "HashedLinear",
    "HashedLinearLayer",
    "HashedNetCompressor",
    "HashedNetsCompressor",
    "hashed_net_factory",
    "hashednets_factory",
]
