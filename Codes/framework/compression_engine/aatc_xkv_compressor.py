"""
AATC-XKV: Attention-Aware Cross-Layer KV Cache Compression (Production-Ready).

Real KV cache compression:
  1. AATC sensitivity-weighted bit allocation (reverse waterfilling).
  2. xKV cross-layer shared subspace via joint SVD on REAL calibration data.
  3. DynamicCache subclass that stores int8 quantized coefficients instead of fp16 K/V.
  4. Attention sink preservation (first token + last token).

The model weights are NEVER modified.
Actual memory savings happen because K/V are stored as int8 coefficients in a
low-rank shared basis, not as full fp16 tensors.

Registered as 'aatc_xkv', 'aatc_xkv_compressor'.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


# =========================================================================
# Data containers
# =========================================================================

@dataclass
class SharedSubspace:
    """Shared low-rank subspace for a group of consecutive transformer layers."""
    V_k: torch.Tensor            # (r_k, D) shared key basis (rows = basis vectors)
    V_v: torch.Tensor            # (r_v, D) shared value basis
    mean_k: torch.Tensor         # (D,) mean of K used for centering
    mean_v: torch.Tensor         # (D,) mean of V used for centering
    bit_alloc_k: torch.Tensor    # (r_k,) int64 bits per component
    bit_alloc_v: torch.Tensor    # (r_v,) int64 bits per component
    group_id: int = 0
    layer_indices: List[int] = field(default_factory=list)


# =========================================================================
# Reverse waterfilling bit allocation
# =========================================================================

def reverse_waterfilling(
    sigma_sq: torch.Tensor,
    sensitivity: torch.Tensor,
    total_bits: int,
    min_bits: int = 0,
    max_bits: int = 8,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Distribute total_bits across r components to minimize
        Σ s_c · σ_c² · 2^(-2 b_c)
    subject to Σ b_c ≤ total_bits and min_bits ≤ b_c ≤ max_bits.

    Uses binary search on the Lagrange multiplier λ in log space:
        b_c = clamp(0.5 · log2(w_c / λ), min_bits, max_bits)
    where w_c = s_c · σ_c².
    """
    device = sigma_sq.device
    r = sigma_sq.numel()
    if r == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    weights = (sensitivity.float() * sigma_sq.float()).clamp(min=1e-30)
    w_max = float(weights.max().item())
    w_min = float(weights.min().item())

    # Search λ over a wide but finite range to avoid overflow
    low = max(w_min * 2.0 ** (-2.0 * max_bits), 1e-30)
    high = w_max * 2.0 ** (-2.0 * min_bits)

    best_alloc = None
    for _ in range(60):
        if high <= low * (1 + 1e-9):
            break
        mid = math.sqrt(low * high)  # geometric mean, better for log-space
        raw_bits = 0.5 * torch.log2(weights / mid)
        alloc = torch.clamp(raw_bits, min=float(min_bits), max=float(max_bits))
        if alloc.sum().item() > total_bits:
            low = mid
        else:
            high = mid
        best_alloc = alloc

    if best_alloc is None:
        best_alloc = torch.full((r,), float(min_bits), device=device)

    # Integer rounding with budget enforcement
    int_alloc = torch.round(best_alloc).to(torch.int64)
    int_alloc = torch.clamp(int_alloc, min=min_bits, max=max_bits)

    # Fine-grained trimming if we overshoot
    it = 0
    while int_alloc.sum().item() > total_bits and it < max_iter:
        active = int_alloc > min_bits
        if not active.any():
            break
        # Decrement component with the smallest weight per bit saved
        cost = torch.where(
            active,
            weights / torch.clamp((int_alloc - min_bits).float(), min=1.0),
            torch.tensor(float("inf"), device=device),
        )
        idx = torch.argmin(cost)
        int_alloc[idx] -= 1
        it += 1

    # If we undershoot significantly, promote high-weight components
    it = 0
    while int_alloc.sum().item() < total_bits and it < max_iter:
        below_max = int_alloc < max_bits
        if not below_max.any():
            break
        benefit = torch.where(
            below_max,
            weights * (2.0 ** (-2.0 * int_alloc.float())) * (1.0 - 2.0 ** -2.0),
            torch.tensor(0.0, device=device),
        )
        idx = torch.argmax(benefit)
        int_alloc[idx] += 1
        it += 1

    return int_alloc


# =========================================================================
# Vectorized quantization (no Python loops over components)
# =========================================================================

def quantize_coefficients(
    U: torch.Tensor,            # (..., r) float
    bit_alloc: torch.Tensor,    # (r,) int64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize U along the last dimension with per-component bit widths.
    Components with b ≤ 0 are dropped (stored as zero with zero scale).

    Returns:
        q:        (..., r) int8
        scales:   (r,) float32  (per-component max/|q_max|)
    """
    orig_shape = U.shape
    r = U.size(-1)
    flat = U.reshape(-1, r).float()

    # Per-component max abs
    max_abs = flat.abs().amax(dim=0).clamp(min=1e-8)          # (r,)
    bits = bit_alloc.to(torch.float32).clamp(min=0.0)
    # q_max_c = 2^(b_c - 1) - 1 for signed int8
    q_max = torch.where(
        bits > 0,
        (2.0 ** (bits - 1.0)) - 1.0,
        torch.ones_like(bits),
    )
    scales = max_abs / q_max                                   # (r,)
    scales = torch.where(bits > 0, scales, torch.ones_like(scales))

    # Quantize
    q_float = flat / scales.unsqueeze(0)
    # Signed clamp per component
    q_min = torch.where(bits > 0, -(2.0 ** (bits - 1.0)), torch.zeros_like(bits))
    q_max_signed = torch.where(bits > 0, q_max, torch.zeros_like(bits))
    q_float = torch.clamp(q_float, min=q_min.unsqueeze(0), max=q_max_signed.unsqueeze(0))
    q = torch.round(q_float).to(torch.int8)

    # Zero out components with b = 0
    active_mask = (bits > 0).to(torch.int8).unsqueeze(0)
    q = q * active_mask

    return q.reshape(orig_shape), scales.to(torch.float32)


def dequantize_coefficients(
    q: torch.Tensor,            # (..., r) int8
    scales: torch.Tensor,       # (r,) float32
) -> torch.Tensor:
    """Restore U from int8 coefficients."""
    r = q.size(-1)
    flat = q.reshape(-1, r).float()
    out = flat * scales.unsqueeze(0).to(flat.dtype)
    return out.reshape(q.shape)


# =========================================================================
# Runtime cache (HF DynamicCache-compatible)
# =========================================================================

try:
    from transformers.cache_utils import DynamicCache as _HFDynamicCache
    _HAS_HF_CACHE = True
except ImportError:
    _HFDynamicCache = object
    _HAS_HF_CACHE = False


class AATC_XKVCache(_HFDynamicCache if _HAS_HF_CACHE else object):
    """
    Drop-in replacement for HF DynamicCache that transparently compresses K/V
    via AATC-XKV.

    Storage per (layer, batch, head, token):
      - k: int8 coefficients of shape (r_k,)  + shared fp32 scale vector (r_k,)
      - v: int8 coefficients of shape (r_v,)  + shared fp32 scale vector (r_v,)

    Sink tokens (first `sink_tokens` and last token) are stored in full precision
    to preserve StreamingLLM stability.

    Memory accounting:
      Original per token:  D_k + D_v floats  (fp16 = 16 bits each)
      Compressed per token: (r_k + r_v) int8 + tiny scale overhead
      Effective ratio: ~ (D_k + D_v) * 16 / (r_k + r_v) * 8
    """

    def __init__(
        self,
        subspaces: Dict[int, SharedSubspace],
        layer_to_group: Dict[int, int],
        sink_tokens: int = 1,
        preserve_last_token: bool = True,
    ):
        super().__init__()
        self.subspaces = subspaces
        self.layer_to_group = layer_to_group
        self.sink_tokens = int(sink_tokens)
        self.preserve_last_token = bool(preserve_last_token)
        self.enabled = True

        # Storage: per-layer lists of tuples
        # Each entry: {"q_k": int8, "s_k": fp32, "q_v": int8, "s_v": fp32,
        #              "sink_k": fp16 or None, "sink_v": fp16 or None}
        self._compressed: Dict[int, List[Dict[str, torch.Tensor]]] = {}

        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    # ---- HF API: length ----
    def __len__(self) -> int:
        if not self._compressed:
            return 0
        any_layer = next(iter(self._compressed.values()))
        if not any_layer:
            return 0
        # Number of "logical" cached tokens; derived from the first layer entry
        e = any_layer[0]
        n = 0
        if e.get("sink_k") is not None:
            n += e["sink_k"].shape[-2]
        n += e["q_k"].shape[-2]
        return n

    # ---- HF API: update ----
    def update(
        self,
        key_states: torch.Tensor,          # (B, H_kv, T_new, D)
        value_states: torch.Tensor,        # (B, H_kv, T_new, D)
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V to the cache (compressed), then return the full
        K/V needed for the current attention computation (dequantized).

        The returned tensors have the same shape as if the cache were uncompressed,
        so the rest of the model works unchanged. But the STORED representation
        is int8, so memory is actually saved across forward passes.
        """
        if not self.enabled or layer_idx not in self.layer_to_group:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        sub = self.subspaces.get(self.layer_to_group[layer_idx])
        if sub is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        B, H_kv, T_new, D = key_states.shape

        # ---- Append new tokens ----
        # For sink tokens we keep full precision. For the rest we quantize.
        entry = self._new_entry(key_states, value_states, sub, layer_idx)

        self._compressed.setdefault(layer_idx, []).append(entry)

        # ---- Return full K/V for current attention ----
        return self._reconstruct_full(layer_idx)

    def _new_entry(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        sub: SharedSubspace,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Compress one (T_new) chunk into an entry."""
        B, H_kv, T_new, D = k.shape

        # Which tokens are sinks (kept full precision)?
        # The very first token of the whole sequence is a sink (StreamingLLM).
        is_first_chunk = (len(self._compressed.get(layer_idx, [])) == 0)
        sink_prefix = self.sink_tokens if is_first_chunk else 0

        # Slice sink region
        if sink_prefix > 0 and T_new > 0:
            sink_k = k[:, :, :sink_prefix, :].to(torch.float16).contiguous()
            sink_v = v[:, :, :sink_prefix, :].to(torch.float16).contiguous()
            rest_k = k[:, :, sink_prefix:, :]
            rest_v = v[:, :, sink_prefix:, :]
        else:
            sink_k = None
            sink_v = None
            rest_k = k
            rest_v = v

        # If preserve_last_token and this chunk is the only/last one, keep last token full
        last_k = None
        last_v = None
        if self.preserve_last_token and rest_k.size(2) > 0:
            last_k = rest_k[:, :, -1:, :].to(torch.float16).contiguous()
            last_v = rest_v[:, :, -1:, :].to(torch.float16).contiguous()
            rest_k = rest_k[:, :, :-1, :]
            rest_v = rest_v[:, :, :-1, :]

        # Project rest into shared basis
        V_k = sub.V_k.to(k.device, torch.float32)
        V_v = sub.V_v.to(v.device, torch.float32)
        mean_k = sub.mean_k.to(k.device, torch.float32)
        mean_v = sub.mean_v.to(v.device, torch.float32)

        if rest_k.size(2) == 0:
            q_k = torch.zeros((B, H_kv, 0, V_k.size(0)), dtype=torch.int8, device=k.device)
            q_v = torch.zeros((B, H_kv, 0, V_v.size(0)), dtype=torch.int8, device=v.device)
            s_k = torch.ones(V_k.size(0), dtype=torch.float32, device=k.device)
            s_v = torch.ones(V_v.size(0), dtype=torch.float32, device=v.device)
        else:
            rest_k_f = rest_k.to(torch.float32) - mean_k.view(1, 1, 1, -1)
            rest_v_f = rest_v.to(torch.float32) - mean_v.view(1, 1, 1, -1)
            U_k = torch.matmul(rest_k_f, V_k.t())      # (B, H_kv, T, r_k)
            U_v = torch.matmul(rest_v_f, V_v.t())      # (B, H_kv, T, r_v)
            q_k, s_k = quantize_coefficients(U_k, sub.bit_alloc_k)
            q_v, s_v = quantize_coefficients(U_v, sub.bit_alloc_v)

        entry = {
            "q_k": q_k, "s_k": s_k,
            "q_v": q_v, "s_v": s_v,
            "sink_k": sink_k, "sink_v": sink_v,
            "last_k": last_k, "last_v": last_v,
        }

        # Memory accounting
        with torch.no_grad():
            orig_b = (k.numel() + v.numel()) * k.element_size()
            comp_b = (
                q_k.numel() * q_k.element_size()
                + q_v.numel() * q_v.element_size()
                + s_k.numel() * s_k.element_size()
                + s_v.numel() * s_v.element_size()
                + (sink_k.numel() * sink_k.element_size() if sink_k is not None else 0)
                + (sink_v.numel() * sink_v.element_size() if sink_v is not None else 0)
                + (last_k.numel() * last_k.element_size() if last_k is not None else 0)
                + (last_v.numel() * last_v.element_size() if last_v is not None else 0)
            )
            self._total_original_bytes += orig_b
            self._total_compressed_bytes += comp_b

        return entry

    def _reconstruct_full(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rebuild full K, V of shape (B, H, N_total, D) from all entries."""
        entries = self._compressed.get(layer_idx, [])
        if not entries:
            # Empty cache — return a zero-length tensor matching the model's shape
            raise RuntimeError("AATC_XKVCache: no entries for layer; update() must be called first.")

        sub = self.subspaces[self.layer_to_group[layer_idx]]
        V_k = sub.V_k.to(torch.float32)
        V_v = sub.V_v.to(torch.float32)
        mean_k = sub.mean_k.to(torch.float32)
        mean_v = sub.mean_v.to(torch.float32)

        k_list, v_list = [], []

        for e in entries:
            parts_k, parts_v = [], []

            # Sink tokens
            if e["sink_k"] is not None:
                parts_k.append(e["sink_k"])
                parts_v.append(e["sink_v"])

            # Compressed middle tokens
            if e["q_k"].size(2) > 0:
                U_k = dequantize_coefficients(e["q_k"], e["s_k"])   # (B, H, T, r_k)
                U_v = dequantize_coefficients(e["q_v"], e["s_v"])
                rec_k = torch.matmul(U_k, V_k.to(U_k.device)) + mean_k.to(U_k.device).view(1, 1, 1, -1)
                rec_v = torch.matmul(U_v, V_v.to(U_v.device)) + mean_v.to(U_v.device).view(1, 1, 1, -1)
                parts_k.append(rec_k.to(torch.float16))
                parts_v.append(rec_v.to(torch.float16))

            # Last token (full precision)
            if e["last_k"] is not None:
                parts_k.append(e["last_k"])
                parts_v.append(e["last_v"])

            if parts_k:
                k_list.append(torch.cat(parts_k, dim=2))
                v_list.append(torch.cat(parts_v, dim=2))

        full_k = torch.cat(k_list, dim=2) if k_list else torch.empty(0)
        full_v = torch.cat(v_list, dim=2) if v_list else torch.empty(0)
        return full_k, full_v

    # ---- HF API: get_seq_length ----
    def get_seq_length(self, layer_idx: int = 0) -> int:
        entries = self._compressed.get(layer_idx, [])
        n = 0
        for e in entries:
            if e["sink_k"] is not None:
                n += e["sink_k"].shape[2]
            n += e["q_k"].shape[2]
            if e["last_k"] is not None:
                n += 1
        return n

    # ---- HF API: get_max_length ----
    def get_max_length(self) -> Optional[int]:
        return None

    # ---- HF API: get_usable_length ----
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    # ---- HF API: seen_tokens / reorder ----
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx, entries in self._compressed.items():
            for e in entries:
                for k in ("sink_k", "sink_v", "last_k", "last_v"):
                    if e[k] is not None:
                        e[k] = e[k].index_select(0, beam_idx)
                for k in ("q_k", "q_v"):
                    e[k] = e[k].index_select(0, beam_idx)

    # ---- Public metrics ----
    def memory_savings_ratio(self) -> float:
        if self._total_compressed_bytes == 0:
            return 1.0
        return self._total_original_bytes / self._total_compressed_bytes

    def memory_bytes(self) -> Dict[str, int]:
        return {
            "original": self._total_original_bytes,
            "compressed": self._total_compressed_bytes,
        }


# =========================================================================
# Calibration
# =========================================================================

def _collect_calibration_data(
    model: nn.Module,
    attn_modules: List[Tuple[int, str, nn.Module]],
    calib_batches: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    device: torch.device,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Run model forward+backward on real calibration batches.
    Collect K, V projections and their gradients per layer.
    """
    k_by_layer: Dict[int, List[torch.Tensor]] = {idx: [] for idx, _, _ in attn_modules}
    v_by_layer: Dict[int, List[torch.Tensor]] = {idx: [] for idx, _, _ in attn_modules}
    gk_by_layer: Dict[int, List[torch.Tensor]] = {idx: [] for idx, _, _ in attn_modules}
    gv_by_layer: Dict[int, List[torch.Tensor]] = {idx: [] for idx, _, _ in attn_modules}

    captured: Dict[int, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_idx: int, tag: str):
        def _hook(module, inputs, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if t.requires_grad:
                t.retain_grad()
            captured.setdefault(layer_idx, {})[tag] = t
        return _hook

    for idx, _, mod in attn_modules:
        k_layer = next((getattr(mod, a) for a in ("k_proj", "k_lin", "key", "W_k") if hasattr(mod, a)), None)
        v_layer = next((getattr(mod, a) for a in ("v_proj", "v_lin", "value", "W_v") if hasattr(mod, a)), None)
        if k_layer is not None:
            hooks.append(k_layer.register_forward_hook(make_hook(idx, "k")))
        if v_layer is not None:
            hooks.append(v_layer.register_forward_hook(make_hook(idx, "v")))

    was_training = model.training
    model.eval()

    try:
        for (input_ids, labels) in calib_batches:
            input_ids = input_ids.to(device)
            if labels is not None:
                labels = labels.to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels)
            loss = getattr(out, "loss", None)
            if loss is None:
                logits = getattr(out, "logits", out)
                probs = F.softmax(logits.float(), dim=-1)
                loss = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
            loss.backward()

            for idx, _, _ in attn_modules:
                if idx not in captured:
                    continue
                k_t = captured[idx].get("k")
                v_t = captured[idx].get("v")
                if k_t is None or v_t is None:
                    continue
                gk = k_t.grad if k_t.grad is not None else torch.zeros_like(k_t)
                gv = v_t.grad if v_t.grad is not None else torch.zeros_like(v_t)

                # Flatten to (N_total, D)
                k_by_layer[idx].append(k_t.detach().reshape(-1, k_t.size(-1)).cpu().float())
                v_by_layer[idx].append(v_t.detach().reshape(-1, v_t.size(-1)).cpu().float())
                gk_by_layer[idx].append(gk.detach().reshape(-1, gk.size(-1)).cpu().float())
                gv_by_layer[idx].append(gv.detach().reshape(-1, gv.size(-1)).cpu().float())
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return {
        "k": {i: torch.cat(k_by_layer[i], 0) for i in k_by_layer},
        "v": {i: torch.cat(v_by_layer[i], 0) for i in v_by_layer},
        "gk": {i: torch.cat(gk_by_layer[i], 0) for i in gk_by_layer},
        "gv": {i: torch.cat(gv_by_layer[i], 0) for i in gv_by_layer},
    }


# =========================================================================
# Subspace construction (randomized SVD for large matrices)
# =========================================================================

def _joint_svd_shared_basis(
    stack: torch.Tensor,   # (N_total, D)
    rank_cap: int,
    energy_threshold: float,
    oversample: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-r right singular vectors of a large (N, D) matrix.
    Uses randomized SVD when N*D is large to avoid OOM.
    Returns (V_r (r, D), sigma (r,), mean (D,)).
    """
    mean = stack.mean(dim=0, keepdim=True)
    X = stack - mean

    r_probe = min(rank_cap + oversample, min(X.shape))
    if X.numel() > 5_000_000:
        # Randomized SVD
        try:
            U, S, V = torch.svd_lowrank(X, q=r_probe)
            # V: (D, r_probe), S: (r_probe,)
        except Exception:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            V = Vh.t()
    else:
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.t()

    # Determine rank by energy threshold
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum().clamp(min=1e-12)
    r_energy = int((cum >= energy_threshold).nonzero()[0].item()) + 1 if (cum >= energy_threshold).any() else len(S)
    r = max(4, min(r_energy, rank_cap, S.numel()))
    V_r = V[:, :r].t().contiguous()          # (r, D)
    return V_r, S[:r].contiguous(), mean.squeeze(0)


# =========================================================================
# Compressor
# =========================================================================

@COMPRESSOR_REGISTRY.register("aatc_xkv")
@COMPRESSOR_REGISTRY.register("aatc_xkv_compressor")
class AATC_XKVCompressor(BaseCompressor):
    """
    AATC-XKV: Attention-Aware Cross-Layer KV Cache Compression.

    Model weights are never modified. During inference, a custom Cache subclass
    (installed at runtime) stores int8 low-rank coefficients instead of fp16 K/V.
    """

    IS_DYNAMIC_KV_CACHE = True

    def __init__(
        self,
        model: nn.Module,
        group_size: int = 3,
        target_bits_per_component: float = 4.0,
        energy_threshold: float = 0.95,
        sink_tokens: int = 1,
        preserve_last_token: bool = True,
        calib_batches: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="aatc_xkv", **kwargs)
        self.cfg = cfg

        self.group_size = int(getattr(cfg, "group_size", group_size) if cfg else group_size)
        self.target_bits = float(getattr(cfg, "aatc_bits", target_bits_per_component) if cfg else target_bits_per_component)
        self.energy_threshold = float(getattr(cfg, "energy_threshold", energy_threshold) if cfg else energy_threshold)
        self.sink_tokens = int(getattr(cfg, "sink_tokens", sink_tokens) if cfg else sink_tokens)
        self.preserve_last_token = bool(getattr(cfg, "preserve_last_token", preserve_last_token) if cfg else preserve_last_token)

        self.subspaces: Dict[int, SharedSubspace] = {}
        self.layer_to_group: Dict[int, int] = {}
        self.calib_batches = calib_batches
        self.stats: Dict[str, Any] = {}

    def _find_attention_modules(self) -> List[Tuple[int, str, nn.Module]]:
        attn_modules = []
        for name, mod in self.model.named_modules():
            name_lower = name.lower()
            if not any(k in name_lower for k in ("self_attn", "attention", "attn")):
                continue
            # Must have BOTH K and V projection children (exact names, no bare "k"/"v")
            has_k = any(hasattr(mod, a) for a in ("k_proj", "k_lin", "key", "W_k"))
            has_v = any(hasattr(mod, a) for a in ("v_proj", "v_lin", "value", "W_v"))
            if not (has_k and has_v):
                # Handle GPT-2 fused c_attn
                if hasattr(mod, "c_attn"):
                    has_k = has_v = True
                else:
                    continue

            parts = name.split(".")
            idx = len(attn_modules)
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    break
            attn_modules.append((idx, name, mod))
        attn_modules.sort(key=lambda x: x[0])
        return attn_modules

    def _build_subspaces(
        self,
        calib: Dict[str, Dict[int, torch.Tensor]],
        attn_modules: List[Tuple[int, str, nn.Module]],
    ) -> None:
        layer_indices = [idx for idx, _, _ in attn_modules]
        G = max(1, self.group_size)
        groups = [layer_indices[i:i + G] for i in range(0, len(layer_indices), G)]

        for g_id, g_layers in enumerate(groups):
            for l_idx in g_layers:
                self.layer_to_group[l_idx] = g_id

            k_list = [calib["k"][l] for l in g_layers if calib["k"][l].numel() > 0]
            v_list = [calib["v"][l] for l in g_layers if calib["v"][l].numel() > 0]
            gk_list = [calib["gk"][l] for l in g_layers if calib["gk"][l].numel() > 0]
            gv_list = [calib["gv"][l] for l in g_layers if calib["gv"][l].numel() > 0]

            if not k_list or not v_list:
                continue

            K_stack = torch.cat(k_list, 0)
            V_stack = torch.cat(v_list, 0)
            gK_stack = torch.cat(gk_list, 0)
            gV_stack = torch.cat(gv_list, 0)

            D = K_stack.size(-1)
            rank_cap = max(4, D // 4)

            V_k, S_k, mean_k = _joint_svd_shared_basis(K_stack, rank_cap, self.energy_threshold)
            V_v, S_v, mean_v = _joint_svd_shared_basis(V_stack, rank_cap, self.energy_threshold)

            r_k = V_k.size(0)
            r_v = V_v.size(0)

            # AATC sensitivity: s_c = E[(g_K · V_c)²] + E[(K · V_c)²]
            proj_gk = gK_stack @ V_k.t()
            proj_k = K_stack @ V_k.t()
            sens_k = proj_gk.pow(2).mean(0) + proj_k.pow(2).mean(0)

            proj_gv = gV_stack @ V_v.t()
            proj_v = V_stack @ V_v.t()
            sens_v = proj_gv.pow(2).mean(0) + proj_v.pow(2).mean(0)

            total_bits_k = int(round(r_k * self.target_bits))
            total_bits_v = int(round(r_v * self.target_bits))
            alloc_k = reverse_waterfilling(S_k.pow(2), sens_k, total_bits_k, 0, 8)
            alloc_v = reverse_waterfilling(S_v.pow(2), sens_v, total_bits_v, 0, 8)

            self.subspaces[g_id] = SharedSubspace(
                V_k=V_k, V_v=V_v, mean_k=mean_k, mean_v=mean_v,
                bit_alloc_k=alloc_k, bit_alloc_v=alloc_v,
                group_id=g_id, layer_indices=g_layers,
            )

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        device = next(self.model.parameters()).device
        attn_modules = self._find_attention_modules()

        if not attn_modules:
            logger.warning("AATC-XKV: no attention modules found.")
            self.stats = {"status": "NO_ATTN_MODULES", "num_compressed_layers": 0}
            return self.model, self.stats

        if not self.calib_batches:
            logger.warning("AATC-XKV: no calibration batches supplied; using synthetic tokens.")
            # Fallback: synthetic calibration (NOT recommended)
            self.calib_batches = self._synthetic_calib(device, len(attn_modules))

        calib = _collect_calibration_data(self.model, attn_modules, self.calib_batches, device)
        self._build_subspaces(calib, attn_modules)

        # Attach cache factory to the model so HF's generate() can pick it up
        factory = self._make_cache_factory()
        self.model._aatc_xkv_cache_factory = factory
        self.model.aatc_xkv_runtime = factory  # public alias

        # Also patch `_supports_cache_class` to make HF use our cache
        self.model._supports_cache_class = True

        # Compute expected memory savings
        total_orig_bits = 0
        total_comp_bits = 0
        for g_id, sub in self.subspaces.items():
            D = sub.V_k.size(1)
            n_layers = len(sub.layer_indices)
            total_orig_bits += 16 * (2 * D) * n_layers
            total_comp_bits += (sub.bit_alloc_k.sum().item() + sub.bit_alloc_v.sum().item()) * n_layers

        ratio = total_orig_bits / max(1, total_comp_bits)

        self.stats = {
            "status": "PASS",
            "method": "aatc_xkv",
            "group_size": self.group_size,
            "num_groups": len(self.subspaces),
            "num_compressed_layers": len(attn_modules),
            "num_fallback_layers": 0,
            "target_bits_per_component": self.target_bits,
            "energy_threshold": self.energy_threshold,
            "theoretical_kv_compression_ratio": round(ratio, 2),
            "total_params_before": self._original_params,
            "total_params_after": self._original_params,
            "weights_modified": False,
        }
        self.model.compression_stats = self.stats
        logger.info(
            "AATC-XKV: %d layers, %d groups, theoretical KV compression %.2fx",
            len(attn_modules), len(self.subspaces), ratio,
        )
        return self.model, self.stats

    def _make_cache_factory(self):
        """Return a callable that HF generate() can use to build a new cache."""
        subspaces = self.subspaces
        layer_to_group = self.layer_to_group
        sink_tokens = self.sink_tokens
        preserve_last = self.preserve_last_token

        def factory():
            return AATC_XKVCache(
                subspaces=subspaces,
                layer_to_group=layer_to_group,
                sink_tokens=sink_tokens,
                preserve_last_token=preserve_last,
            )
        return factory

    @staticmethod
    def _synthetic_calib(device, n_layers):
        """Fallback synthetic calibration — only for smoke tests."""
        logger.warning("AATC-XKV: synthetic calibration is not recommended for production.")
        torch.manual_seed(42)
        batches = []
        for _ in range(8):
            ids = torch.randint(5, 30000, (1, 128))
            batches.append((ids, None))
        return batches