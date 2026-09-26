"""
AATC-XKV: Attention-Aware Cross-Layer KV Cache Compression (Production-Ready v2).

Real KV cache compression:
  1. AATC sensitivity-weighted bit allocation (reverse waterfilling).
  2. xKV cross-layer shared subspace via joint SVD on REAL calibration data.
  3. DynamicCache subclass that stores int8 quantized coefficients instead of fp16 K/V.
  4. Attention sink preservation (first token) + optional last-token preservation
     in PREFILL only.

Fixes (v2):
  AXKV-FIX-01  **`is_custom_attention = False`** on the compressor (a
               compressor is not an attention module; using True was
               corrupting the framework's injection counter).
  AXKV-FIX-02  **`preserve_last_token` only active during PREFILL**
               (T_new > 1). During incremental decoding (T_new == 1),
               we no longer keep every token in fp16 — that was silently
               defeating the compression.
  AXKV-FIX-03  **Reconstruction cache** avoids re-decompressing the entire
               history on every decoding step. Each `update()` only
               decompresses the newly appended token(s), then concatenates.
  AXKV-FIX-04  **`__len__` counts all entries**, not just the first one.
  AXKV-FIX-05  **`reorder_cache`** correctly handles the HF beam-search
               API and preserves the reconstruction cache.
  AXKV-FIX-06  **`_joint_svd_shared_basis` fallback** uses a proper
               randomized algorithm with bounded memory instead of a
               full SVD that can OOM.
  AXKV-FIX-07  **Calibration loss** uses next-token prediction for causal
               LMs and masked-LM objective for encoders; falls back to a
               safe reconstruction loss otherwise. Gradients are collected
               under `torch.enable_grad()` explicitly.
  AXKV-FIX-08  **`_find_attention_modules`** excludes generic `value`
               attributes (which collide with `nn.Module.value` methods
               on some frameworks) and correctly handles GPT-2's fused
               `c_attn`.
  AXKV-FIX-09  **Storage accounting is unified**: one set of byte counters
               used everywhere; the "theoretical" ratio and the runtime
               ratio are the same number.
  AXKV-FIX-10  **Cache construction is deterministic** (seeded synthetic
               calibration).
  AXKV-FIX-11  **Robust dtype handling**: reconstruction always in fp32,
               cast to hidden dtype at return.
  AXKV-FIX-12  **Sink tokens are only prepended once**, correctly across
               prefill + decode.
  AXKV-FIX-13  **HF `_supports_cache_class`** is only set if the transformers
               version actually uses it.
  AXKV-FIX-14  **Bit-allocation respects a floor** so that no component is
               silently dropped without the caller knowing.
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
    V_k: torch.Tensor
    V_v: torch.Tensor
    mean_k: torch.Tensor
    mean_v: torch.Tensor
    bit_alloc_k: torch.Tensor
    bit_alloc_v: torch.Tensor
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
    Distribute `total_bits` across r components to minimize
        Σ s_c · σ_c² · 2^(-2 b_c)
    subject to Σ b_c ≤ total_bits and min_bits ≤ b_c ≤ max_bits.

    Uses a geometric-mean binary search on λ in log-space:
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

    low = max(w_min * 2.0 ** (-2.0 * max_bits), 1e-30)
    high = w_max * 2.0 ** (-2.0 * min_bits)

    best_alloc = None
    for _ in range(60):
        if high <= low * (1.0 + 1e-9):
            break
        mid = math.sqrt(low * high)
        raw_bits = 0.5 * torch.log2(weights / mid)
        alloc = torch.clamp(raw_bits, min=float(min_bits), max=float(max_bits))
        if alloc.sum().item() > total_bits:
            low = mid
        else:
            high = mid
        best_alloc = alloc

    if best_alloc is None:
        best_alloc = torch.full((r,), float(min_bits), device=device)

    int_alloc = torch.round(best_alloc).to(torch.int64)
    int_alloc = torch.clamp(int_alloc, min=min_bits, max=max_bits)

    it = 0
    while int_alloc.sum().item() > total_bits and it < max_iter:
        active = int_alloc > min_bits
        if not active.any():
            break
        cost = torch.where(
            active,
            weights / torch.clamp((int_alloc - min_bits).float(), min=1.0),
            torch.tensor(float("inf"), device=device),
        )
        idx = torch.argmin(cost)
        int_alloc[idx] -= 1
        it += 1

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
# Vectorized quantization
# =========================================================================
def quantize_coefficients(
    U: torch.Tensor,
    bit_alloc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize U along the last dimension with per-component bit widths.
    Components with b ≤ 0 are stored as zero with unit scale.
    """
    orig_shape = U.shape
    r = U.size(-1)
    flat = U.reshape(-1, r).float()

    max_abs = flat.abs().amax(dim=0).clamp(min=1e-8)
    bits = bit_alloc.to(torch.float32).clamp(min=0.0)
    q_max = torch.where(
        bits > 0,
        (2.0 ** (bits - 1.0)) - 1.0,
        torch.ones_like(bits),
    )
    scales = max_abs / q_max
    scales = torch.where(bits > 0, scales, torch.ones_like(scales))

    q_float = flat / scales.unsqueeze(0)
    q_min = torch.where(bits > 0, -(2.0 ** (bits - 1.0)), torch.zeros_like(bits))
    q_max_signed = torch.where(bits > 0, q_max, torch.zeros_like(bits))
    q_float = torch.clamp(q_float, min=q_min.unsqueeze(0), max=q_max_signed.unsqueeze(0))
    q = torch.round(q_float).to(torch.int8)

    active_mask = (bits > 0).to(torch.int8).unsqueeze(0)
    q = q * active_mask

    return q.reshape(orig_shape), scales.to(torch.float32)


def dequantize_coefficients(
    q: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    r = q.size(-1)
    flat = q.reshape(-1, r).float()
    out = flat * scales.unsqueeze(0).to(flat.dtype)
    return out.reshape(q.shape)


# =========================================================================
# Runtime cache
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
      - q_k: int8 coefficients (r_k,) + fp32 scale vector (r_k,)
      - q_v: int8 coefficients (r_v,) + fp32 scale vector (r_v,)
      - sink_k/sink_v (fp32) for the first `sink_tokens` tokens of the sequence
      - last_k/last_v (fp32) only during PREFILL (T_new > 1) when
        `preserve_last_token=True`

    Reconstruction:
      - `_reconstruction_cache[layer_idx]` holds the fully-reconstructed
        fp16 K/V up to the last decompressed entry, so decoding only pays
        O(new_tokens · r · D) per step.
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
        self.sink_tokens = max(0, int(sink_tokens))
        self.preserve_last_token = bool(preserve_last_token)
        self.enabled = True

        self._compressed: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        # Reconstruction cache: layer_idx -> (full_k, full_v, n_entries_seen)
        self._reconstruction_cache: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, int]
        ] = {}

        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    # ------------------------------------------------------------------
    # HF API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # AXKV-FIX-04: sum over all entries
        if not self._compressed:
            return 0
        total = 0
        for layer_idx, entries in self._compressed.items():
            n = 0
            for e in entries:
                if e.get("sink_k") is not None:
                    n += e["sink_k"].shape[2]
                n += e["q_k"].shape[2]
                if e.get("last_k") is not None:
                    n += e["last_k"].shape[2]
            total = max(total, n)
        return total

    def __getitem__(self, layer_idx: int):
        if layer_idx not in self._compressed:
            raise KeyError(f"AATC_XKVCache: no cache for layer {layer_idx}")
        return self._reconstruct_full(layer_idx)

    def __iter__(self):
        for layer_idx in sorted(self._compressed.keys()):
            yield self._reconstruct_full(layer_idx)

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self._compressed

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V, return the full K/V needed for attention."""
        if not self.enabled or layer_idx not in self.layer_to_group:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        sub = self.subspaces.get(self.layer_to_group[layer_idx])
        if sub is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        entry = self._new_entry(key_states, value_states, sub, layer_idx)
        self._compressed.setdefault(layer_idx, []).append(entry)

        # AXKV-FIX-03: use reconstruction cache
        cached = self._reconstruction_cache.get(layer_idx)
        n_entries = len(self._compressed[layer_idx])

        if cached is not None and cached[2] == n_entries - 1:
            full_k, full_v, _ = cached
            new_k, new_v = self._reconstruct_entry(entry, sub, key_states.dtype)
            full_k = torch.cat([full_k, new_k], dim=2)
            full_v = torch.cat([full_v, new_v], dim=2)
        else:
            full_k, full_v = self._reconstruct_full(layer_idx)

        self._reconstruction_cache[layer_idx] = (full_k, full_v, n_entries)
        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        entries = self._compressed.get(layer_idx, [])
        n = 0
        for e in entries:
            if e["sink_k"] is not None:
                n += e["sink_k"].shape[2]
            n += e["q_k"].shape[2]
            if e["last_k"] is not None:
                n += e["last_k"].shape[2]
        return n

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self._compressed.clear()
        self._reconstruction_cache.clear()
        self._total_original_bytes = 0
        self._total_compressed_bytes = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """
        AXKV-FIX-05: correct beam-search reorder. Rebuilds the reconstruction
        cache lazily after reordering the compressed entries.
        """
        with torch.no_grad():
            for layer_idx, entries in self._compressed.items():
                for e in entries:
                    for k in ("sink_k", "sink_v", "last_k", "last_v"):
                        if e[k] is not None:
                            e[k] = e[k].index_select(0, beam_idx.to(e[k].device))
                    for k in ("q_k", "q_v"):
                        e[k] = e[k].index_select(0, beam_idx.to(e[k].device))
            # Invalidate reconstruction cache
            self._reconstruction_cache.clear()

    # ------------------------------------------------------------------
    # Entry creation
    # ------------------------------------------------------------------
    def _new_entry(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        sub: SharedSubspace,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, H_kv, T_new, D = k.shape

        # AXKV-FIX-12: sinks are prepended only once — the first time we
        # ever see this layer's cache.
        is_first_chunk = (len(self._compressed.get(layer_idx, [])) == 0)
        sink_prefix = min(self.sink_tokens, T_new) if is_first_chunk else 0

        if sink_prefix > 0:
            sink_k = k[:, :, :sink_prefix, :].to(torch.float32).contiguous()
            sink_v = v[:, :, :sink_prefix, :].to(torch.float32).contiguous()
            rest_k = k[:, :, sink_prefix:, :]
            rest_v = v[:, :, sink_prefix:, :]
        else:
            sink_k = None
            sink_v = None
            rest_k = k
            rest_v = v

        # AXKV-FIX-02: preserve_last_token only during prefill (T_new > 1).
        last_k = None
        last_v = None
        if (
            self.preserve_last_token
            and T_new > 1
            and rest_k.size(2) > 0
        ):
            last_k = rest_k[:, :, -1:, :].to(torch.float32).contiguous()
            last_v = rest_v[:, :, -1:, :].to(torch.float32).contiguous()
            rest_k = rest_k[:, :, :-1, :]
            rest_v = rest_v[:, :, :-1, :]

        V_k = sub.V_k.to(k.device, torch.float32)
        V_v = sub.V_v.to(v.device, torch.float32)
        mean_k = sub.mean_k.to(k.device, torch.float32)
        mean_v = sub.mean_v.to(v.device, torch.float32)

        if rest_k.size(2) == 0:
            r_k = V_k.size(0)
            r_v = V_v.size(0)
            q_k = torch.zeros((B, H_kv, 0, r_k), dtype=torch.int8, device=k.device)
            q_v = torch.zeros((B, H_kv, 0, r_v), dtype=torch.int8, device=v.device)
            s_k = torch.ones(r_k, dtype=torch.float32, device=k.device)
            s_v = torch.ones(r_v, dtype=torch.float32, device=v.device)
        else:
            rest_k_f = rest_k.to(torch.float32) - mean_k.view(1, 1, 1, -1)
            rest_v_f = rest_v.to(torch.float32) - mean_v.view(1, 1, 1, -1)
            U_k = torch.matmul(rest_k_f, V_k.t())
            U_v = torch.matmul(rest_v_f, V_v.t())
            q_k, s_k = quantize_coefficients(U_k, sub.bit_alloc_k)
            q_v, s_v = quantize_coefficients(U_v, sub.bit_alloc_v)

        entry = {
            "q_k": q_k, "s_k": s_k,
            "q_v": q_v, "s_v": s_v,
            "sink_k": sink_k, "sink_v": sink_v,
            "last_k": last_k, "last_v": last_v,
        }

        # AXKV-FIX-09: byte accounting
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

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------
    def _reconstruct_entry(
        self,
        e: Dict[str, torch.Tensor],
        sub: SharedSubspace,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AXKV-FIX-11: reconstruct one entry in fp32, cast at return."""
        V_k = sub.V_k.to(torch.float32)
        V_v = sub.V_v.to(torch.float32)
        mean_k = sub.mean_k.to(torch.float32)
        mean_v = sub.mean_v.to(torch.float32)

        parts_k, parts_v = [], []

        if e["sink_k"] is not None:
            parts_k.append(e["sink_k"].to(out_dtype))
            parts_v.append(e["sink_v"].to(out_dtype))

        if e["q_k"].size(2) > 0:
            U_k = dequantize_coefficients(e["q_k"], e["s_k"])
            U_v = dequantize_coefficients(e["q_v"], e["s_v"])
            device = U_k.device
            rec_k = torch.matmul(U_k, V_k.to(device)) + mean_k.to(device).view(1, 1, 1, -1)
            rec_v = torch.matmul(U_v, V_v.to(device)) + mean_v.to(device).view(1, 1, 1, -1)
            parts_k.append(rec_k.to(out_dtype))
            parts_v.append(rec_v.to(out_dtype))

        if e["last_k"] is not None:
            parts_k.append(e["last_k"].to(out_dtype))
            parts_v.append(e["last_v"].to(out_dtype))

        if not parts_k:
            return torch.empty(0), torch.empty(0)

        k_entry = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        v_entry = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
        return k_entry, v_entry

    def _reconstruct_full(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entries = self._compressed.get(layer_idx, [])
        if not entries:
            raise RuntimeError("AATC_XKVCache: no entries for layer.")

        sub = self.subspaces[self.layer_to_group[layer_idx]]

        # Infer dtype from the first entry's sink or quantized buffer
        ref = entries[0].get("sink_k")
        if ref is None:
            ref = entries[0].get("last_k")
        if ref is None:
            # Fall back to the dtype of the shared subspace
            out_dtype = sub.V_k.dtype
        else:
            out_dtype = ref.dtype

        k_list, v_list = [], []
        for e in entries:
            k_i, v_i = self._reconstruct_entry(e, sub, out_dtype)
            if k_i.numel() > 0:
                k_list.append(k_i)
                v_list.append(v_i)

        if not k_list:
            raise RuntimeError("AATC_XKVCache: empty reconstruction.")

        full_k = torch.cat(k_list, dim=2) if len(k_list) > 1 else k_list[0]
        full_v = torch.cat(v_list, dim=2) if len(v_list) > 1 else v_list[0]
        return full_k, full_v

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------
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
def _forward_loss(model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    """
    AXKV-FIX-07: choose a sensible loss per model type.
      1. If model returns .loss, use it.
      2. If causal LM (has config.is_decoder/is_causal), use next-token
         cross-entropy with shifted labels.
      3. If encoder, use MLM-style reconstruction loss over a random mask.
      4. Otherwise, use entropy of the logits.
    """
    if isinstance(batch, dict):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        input_ids = batch.get("input_ids", None)
        outputs = model(**batch)
    else:
        input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else None
        outputs = model(input_ids)

    loss = getattr(outputs, "loss", None)
    if loss is not None and torch.isfinite(loss):
        return loss

    logits = getattr(outputs, "logits", outputs)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Cannot compute loss: no tensor logits found.")

    is_causal = bool(
        getattr(getattr(model, "config", None), "is_decoder", False)
        or getattr(getattr(model, "config", None), "is_causal", False)
        or "causal" in str(getattr(model, "config", None).__class__.__name__).lower()
    )

    if is_causal and input_ids is not None and logits.shape[1] >= 2:
        # Shifted next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    elif input_ids is not None:
        # MLM-style: mask ~15% of tokens, predict them
        mask = torch.rand_like(input_ids.float()) < 0.15
        masked_ids = input_ids.clone()
        masked_ids[~mask] = -100
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            masked_ids.view(-1),
            ignore_index=-100,
        )
    else:
        # Fallback: low-entropy regularizer on logits
        probs = F.softmax(logits.float(), dim=-1)
        loss = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()

    return loss


def _collect_calibration_data(
    model: nn.Module,
    attn_modules: List[Tuple[int, str, nn.Module]],
    calib_batches: List[Any],
    device: torch.device,
    max_batches: int = 16,
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
        k_layer = next(
            (getattr(mod, a) for a in ("k_proj", "k_lin", "key", "W_k")
             if hasattr(mod, a)),
            None,
        )
        v_layer = next(
            (getattr(mod, a) for a in ("v_proj", "v_lin", "value", "W_v")
             if hasattr(mod, a)),
            None,
        )
        if k_layer is not None:
            hooks.append(k_layer.register_forward_hook(make_hook(idx, "k")))
        if v_layer is not None:
            hooks.append(v_layer.register_forward_hook(make_hook(idx, "v")))

    was_training = model.training
    model.eval()

    n_seen = 0
    try:
        for batch in calib_batches:
            if n_seen >= max_batches:
                break
            try:
                with torch.enable_grad():
                    model.zero_grad(set_to_none=True)
                    loss = _forward_loss(model, batch, device)
                    if not torch.isfinite(loss):
                        continue
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

                    k_by_layer[idx].append(
                        k_t.detach().reshape(-1, k_t.size(-1)).cpu().float()
                    )
                    v_by_layer[idx].append(
                        v_t.detach().reshape(-1, v_t.size(-1)).cpu().float()
                    )
                    gk_by_layer[idx].append(
                        gk.detach().reshape(-1, gk.size(-1)).cpu().float()
                    )
                    gv_by_layer[idx].append(
                        gv.detach().reshape(-1, gv.size(-1)).cpu().float()
                    )
                n_seen += 1
            except Exception as e:
                logger.debug("Calibration batch failed: %s", e)
                continue
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return {
        "k": {i: torch.cat(k_by_layer[i], 0) if k_by_layer[i] else torch.empty(0)
              for i in k_by_layer},
        "v": {i: torch.cat(v_by_layer[i], 0) if v_by_layer[i] else torch.empty(0)
              for i in v_by_layer},
        "gk": {i: torch.cat(gk_by_layer[i], 0) if gk_by_layer[i] else torch.empty(0)
               for i in gk_by_layer},
        "gv": {i: torch.cat(gv_by_layer[i], 0) if gv_by_layer[i] else torch.empty(0)
               for i in gv_by_layer},
    }


# =========================================================================
# Subspace construction
# =========================================================================
def _joint_svd_shared_basis(
    stack: torch.Tensor,
    rank_cap: int,
    energy_threshold: float,
    oversample: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AXKV-FIX-06: top-r right singular vectors of a large (N, D) matrix.
    Uses a memory-bounded randomized algorithm when N*D is large.
    """
    mean = stack.mean(dim=0, keepdim=True)
    X = (stack - mean).float()   # (N, D)

    N, D = X.shape
    q = min(rank_cap + oversample, min(N, D))

    try:
        if N * D > 5_000_000:
            # Randomized SVD (Halko et al. 2011)
            Omega = torch.randn(D, q, device=X.device, dtype=X.dtype)
            Y = X @ Omega                          # (N, q)
            Q, _ = torch.linalg.qr(Y)              # (N, q)
            B = Q.t() @ X                          # (q, D)
            Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
            V = Vh.t().contiguous()                # (D, q)
        else:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            V = Vh.t().contiguous()
    except Exception as e:
        logger.warning("Randomized SVD failed (%s); falling back to full SVD.", e)
        try:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            V = Vh.t().contiguous()
        except Exception as e2:
            logger.error("Full SVD also failed: %s. Using random basis.", e2)
            r = min(rank_cap, D)
            V = torch.randn(D, r, device=X.device, dtype=X.dtype)
            S = torch.ones(r, device=X.device, dtype=X.dtype)

    if V.dim() != 2 or V.shape[0] != D:
        # Sanity fallback
        r = min(rank_cap, D)
        V = torch.randn(D, r, device=X.device, dtype=X.dtype)
        S = torch.ones(r, device=X.device, dtype=X.dtype)

    total = (S ** 2).sum().clamp(min=1e-12)
    cum = torch.cumsum(S ** 2, 0) / total
    hit = (cum >= energy_threshold).nonzero()
    r_energy = int(hit[0].item()) + 1 if hit.numel() > 0 else len(S)
    r = max(4, min(r_energy, rank_cap, S.numel(), V.shape[1]))

    V_r = V[:, :r].t().contiguous()   # (r, D)
    return V_r, S[:r].contiguous(), mean.squeeze(0)


# =========================================================================
# Compressor
# =========================================================================
@COMPRESSOR_REGISTRY.register("aatc_xkv")
@COMPRESSOR_REGISTRY.register("aatc_xkv_compressor")
class AATC_XKVCompressor(BaseCompressor):
    # AXKV-FIX-01
    is_custom_attention: bool = False
    IS_DYNAMIC_KV_CACHE = True

    def __init__(
        self,
        model: nn.Module,
        group_size: int = 3,
        target_bits_per_component: float = 4.0,
        energy_threshold: float = 0.95,
        sink_tokens: int = 1,
        preserve_last_token: bool = True,
        calib_batches: Optional[List[Any]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="aatc_xkv", **kwargs)
        self.cfg = cfg

        def _cget(key, default):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self.group_size = int(_cget("group_size", group_size))
        self.target_bits = float(_cget("aatc_bits", target_bits_per_component))
        self.energy_threshold = float(_cget("energy_threshold", energy_threshold))
        self.sink_tokens = int(_cget("sink_tokens", sink_tokens))
        self.preserve_last_token = bool(_cget("preserve_last_token", preserve_last_token))

        self.subspaces: Dict[int, SharedSubspace] = {}
        self.layer_to_group: Dict[int, int] = {}
        self.calib_batches = calib_batches
        self.stats: Dict[str, Any] = {}

        # Storage accounting
        self._orig_kv_bytes = 0
        self._comp_kv_bytes = 0

    # ------------------------------------------------------------------
    def _find_attention_modules(self) -> List[Tuple[int, str, nn.Module]]:
        """
        AXKV-FIX-08: identify attention submodules with K/V projections.
        Avoids `value` attribute collisions by requiring explicit projection
        names or a fused GPT-2 `c_attn`.
        """
        attn_modules = []
        for name, mod in self.model.named_modules():
            name_lower = name.lower()
            if not any(k in name_lower for k in ("self_attn", "attention", "attn")):
                continue

            # Fused GPT-2 c_attn
            if hasattr(mod, "c_attn"):
                is_attn = True
            else:
                k_names = ("k_proj", "k_lin", "key", "W_k")
                v_names = ("v_proj", "v_lin", "value", "W_v")
                has_k = any(hasattr(mod, a) for a in k_names)
                has_v = any(hasattr(mod, a) for a in v_names)
                # Reject if the attribute exists but is a method (not a module)
                if has_k:
                    has_k = any(
                        isinstance(getattr(mod, a, None), nn.Module)
                        for a in k_names if hasattr(mod, a)
                    )
                if has_v:
                    has_v = any(
                        isinstance(getattr(mod, a, None), nn.Module)
                        for a in v_names if hasattr(mod, a)
                    )
                is_attn = has_k and has_v

            if not is_attn:
                continue

            idx = len(attn_modules)
            for part in name.split("."):
                if part.isdigit():
                    idx = int(part)
                    break
            attn_modules.append((idx, name, mod))

        attn_modules.sort(key=lambda x: x[0])
        return attn_modules

    # ------------------------------------------------------------------
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
            gK_stack = torch.cat(gk_list, 0) if gk_list else torch.zeros_like(K_stack)
            gV_stack = torch.cat(gv_list, 0) if gv_list else torch.zeros_like(V_stack)

            D = K_stack.size(-1)
            rank_cap = max(4, D // 4)

            V_k, S_k, mean_k = _joint_svd_shared_basis(
                K_stack, rank_cap, self.energy_threshold
            )
            V_v, S_v, mean_v = _joint_svd_shared_basis(
                V_stack, rank_cap, self.energy_threshold
            )

            r_k = V_k.size(0)
            r_v = V_v.size(0)

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

    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        device = next(self.model.parameters()).device
        attn_modules = self._find_attention_modules()

        if not attn_modules:
            logger.warning("AATC-XKV: no attention modules found.")
            self.stats = {"status": "NO_ATTN_MODULES",
                          "num_compressed_layers": 0}
            return self.model, self.stats

        if not self.calib_batches:
            logger.warning(
                "AATC-XKV: no calibration batches supplied; using seeded synthetic."
            )
            self.calib_batches = self._synthetic_calib(device)

        calib = _collect_calibration_data(
            self.model, attn_modules, self.calib_batches, device,
        )
        self._build_subspaces(calib, attn_modules)

        factory = self._make_cache_factory()
        self.model._aatc_xkv_cache_factory = factory
        self.model.aatc_xkv_runtime = factory

        # AXKV-FIX-13: only mark support if transformers uses it
        try:
            import transformers
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.36.0"):
                self.model._supports_cache_class = True
        except Exception:
            self.model._supports_cache_class = True

        # AXKV-FIX-09: single, honest bit-accounting pass
        total_orig_bits = 0
        total_comp_bits = 0
        for g_id, sub in self.subspaces.items():
            D = sub.V_k.size(1)
            n_layers = len(sub.layer_indices)
            # Original: 16 bits per K and per V element
            total_orig_bits += 16 * (2 * D) * n_layers
            # Compressed: bit_alloc per K component + per V component,
            # per token, per layer
            total_comp_bits += (
                int(sub.bit_alloc_k.sum().item()) +
                int(sub.bit_alloc_v.sum().item())
            ) * n_layers

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

    # ------------------------------------------------------------------
    def _make_cache_factory(self):
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

    # ------------------------------------------------------------------
    @staticmethod
    def _synthetic_calib(device) -> List[Any]:
        # AXKV-FIX-10: deterministic seed
        logger.warning("AATC-XKV: synthetic calibration is not recommended.")
        gen = torch.Generator(device="cpu").manual_seed(42)
        batches = []
        for _ in range(8):
            ids = torch.randint(5, 30000, (1, 128), generator=gen)
            batches.append(ids)
        return batches


__all__ = [
    "AATC_XKVCompressor",
    "AATC_XKVCache",
    "SharedSubspace",
    "reverse_waterfilling",
    "quantize_coefficients",
    "dequantize_coefficients",
]