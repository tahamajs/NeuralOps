#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recurrent State Cache Infrastructure

O(1) persistent state cache engine supporting autoregressive recurrent and linear attention decoding.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class NeuralOpsStateCache:
    """Unified constant-memory inference state cache across multi-layer models.

    Supports:
      - Fixed-shape recurrent state matrices S in R^{B x H x d_k x d_v}
      - Fixed-shape rolling 1D convolution states in R^{B x D x (K_conv - 1)}
      - Fixed-shape sliding-window / sink KV cache states in R^{B x H x (N_sink + W) x d}
      - In-place `.copy_()` updates ensuring static pointer addresses for CUDAGraph capture
      - Duck-typing for Hugging Face `Cache` interface
    """

    def __init__(self) -> None:
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        self.recurrent_z_states: Dict[int, torch.Tensor] = {}
        self.conv_states: Dict[int, torch.Tensor] = {}
        self.window_kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._window_meta: Dict[int, Tuple[int, int, int, int]] = {}
        self.seen_tokens: int = 0

    def update_recurrent_state(
        self,
        layer_idx: int,
        new_state: torch.Tensor,
        state_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stores or in-place updates recurrent state matrix for layer_idx in O(1)."""
        if layer_idx in self.recurrent_states:
            cached = self.recurrent_states[layer_idx]
            if cached.shape == new_state.shape and cached.dtype == new_state.dtype and cached.device == new_state.device:
                cached.copy_(new_state)
            else:
                self.recurrent_states[layer_idx] = new_state.detach().clone()
        else:
            self.recurrent_states[layer_idx] = new_state.detach().clone()

        if state_z is not None:
            if layer_idx in self.recurrent_z_states:
                cached_z = self.recurrent_z_states[layer_idx]
                if cached_z.shape == state_z.shape and cached_z.dtype == state_z.dtype and cached_z.device == state_z.device:
                    cached_z.copy_(state_z)
                else:
                    self.recurrent_z_states[layer_idx] = state_z.detach().clone()
            else:
                self.recurrent_z_states[layer_idx] = state_z.detach().clone()

        return self.recurrent_states[layer_idx]

    def get_recurrent_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.recurrent_states.get(layer_idx, None)

    def get_recurrent_z_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.recurrent_z_states.get(layer_idx, None)

    def update_window_kv(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        max_window: int = 256,
        sink_tokens: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append tokens into a fixed-capacity sink plus circular-window buffer.

        The backing allocation remains stable after initialization. Returned tensors are
        materialized in chronological order to preserve the historical API contract.
        """
        if max_window <= 0 or sink_tokens < 0:
            raise ValueError("max_window must be positive and sink_tokens non-negative")
        if k_new.shape[:-1] != v_new.shape[:-1] or k_new.shape[-2] != v_new.shape[-2]:
            raise ValueError("new key and value tensors must have matching token dimensions")
        capacity = sink_tokens + max_window
        k_new, v_new = k_new.detach(), v_new.detach()

        if layer_idx not in self.window_kv_states:
            k_buffer = k_new.new_empty((*k_new.shape[:-2], capacity, k_new.shape[-1]))
            v_buffer = v_new.new_empty((*v_new.shape[:-2], capacity, v_new.shape[-1]))
            sink_count = min(sink_tokens, k_new.shape[-2])
            if sink_count:
                k_buffer[..., :sink_count, :].copy_(k_new[..., :sink_count, :])
                v_buffer[..., :sink_count, :].copy_(v_new[..., :sink_count, :])
            recent_k = k_new[..., sink_count:, :]
            recent_v = v_new[..., sink_count:, :]
            if recent_k.shape[-2] > max_window:
                recent_k = recent_k[..., -max_window:, :]
                recent_v = recent_v[..., -max_window:, :]
            recent_count = recent_k.shape[-2]
            if recent_count:
                k_buffer[..., sink_tokens : sink_tokens + recent_count, :].copy_(recent_k)
                v_buffer[..., sink_tokens : sink_tokens + recent_count, :].copy_(recent_v)
            write_pos = recent_count % max_window
            self.window_kv_states[layer_idx] = (k_buffer, v_buffer)
            self._window_meta[layer_idx] = (sink_count, recent_count, write_pos, max_window)
        else:
            k_buffer, v_buffer = self.window_kv_states[layer_idx]
            sink_count, recent_count, write_pos, cached_window = self._window_meta[layer_idx]
            if (
                cached_window != max_window
                or k_buffer.shape[-2] != sink_tokens + max_window
                or k_buffer.shape[:-2] != k_new.shape[:-2]
                or v_buffer.shape[:-2] != v_new.shape[:-2]
                or k_buffer.shape[-1] != k_new.shape[-1]
                or v_buffer.shape[-1] != v_new.shape[-1]
                or k_buffer.device != k_new.device
                or v_buffer.device != v_new.device
                or k_buffer.dtype != k_new.dtype
                or v_buffer.dtype != v_new.dtype
            ):
                raise ValueError("window-cache shape, dtype, device, and capacity must remain stable")
            for token_idx in range(k_new.shape[-2]):
                if sink_count < sink_tokens:
                    k_buffer[..., sink_count, :].copy_(k_new[..., token_idx, :])
                    v_buffer[..., sink_count, :].copy_(v_new[..., token_idx, :])
                    sink_count += 1
                else:
                    slot = sink_tokens + write_pos
                    k_buffer[..., slot, :].copy_(k_new[..., token_idx, :])
                    v_buffer[..., slot, :].copy_(v_new[..., token_idx, :])
                    if recent_count < max_window:
                        recent_count += 1
                    write_pos = (write_pos + 1) % max_window
            self._window_meta[layer_idx] = (sink_count, recent_count, write_pos, max_window)

        return self._ordered_window_kv(layer_idx)

    def _ordered_window_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the populated cache entries in sink-then-chronological order."""
        k_buffer, v_buffer = self.window_kv_states[layer_idx]
        sink_count, recent_count, write_pos, cached_window = self._window_meta[layer_idx]
        sink_capacity = k_buffer.shape[-2] - cached_window
        if recent_count == 0:
            return (
                k_buffer[..., :sink_count, :].clone(),
                v_buffer[..., :sink_count, :].clone(),
            )
        if sink_count == sink_capacity and (recent_count < cached_window or write_pos == 0):
            return (
                k_buffer[..., : sink_count + recent_count, :].clone(),
                v_buffer[..., : sink_count + recent_count, :].clone(),
            )
        recent_start = sink_capacity + write_pos if recent_count == cached_window else sink_capacity
        first_count = min(recent_count, cached_window - (recent_start - sink_capacity))
        k_recent = k_buffer[..., recent_start : recent_start + first_count, :]
        v_recent = v_buffer[..., recent_start : recent_start + first_count, :]
        if first_count < recent_count:
            remaining = recent_count - first_count
            k_recent = torch.cat(
                (k_recent, k_buffer[..., sink_capacity : sink_capacity + remaining, :]), dim=-2
            )
            v_recent = torch.cat(
                (v_recent, v_buffer[..., sink_capacity : sink_capacity + remaining, :]), dim=-2
            )
        return (
            torch.cat((k_buffer[..., :sink_count, :], k_recent), dim=-2),
            torch.cat((v_buffer[..., :sink_count, :], v_recent), dim=-2),
        )

    def memory_bytes(self) -> int:
        """Calculates exact total memory bytes occupied across all stored states."""
        total = 0
        for s in self.recurrent_states.values():
            if isinstance(s, torch.Tensor):
                total += s.numel() * s.element_size()
        for sz in self.recurrent_z_states.values():
            if isinstance(sz, torch.Tensor):
                total += sz.numel() * sz.element_size()
        for c in self.conv_states.values():
            if isinstance(c, torch.Tensor):
                total += c.numel() * c.element_size()
        for k, v in self.window_kv_states.values():
            if isinstance(k, torch.Tensor):
                total += k.numel() * k.element_size()
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
        return total

    def memory_mb(self) -> float:
        """Returns exact total memory occupied in Megabytes."""
        return float(self.memory_bytes() / (1024 * 1024))

    def reset(self) -> None:
        """Clears all states and resets token counters."""
        self.recurrent_states.clear()
        self.recurrent_z_states.clear()
        self.conv_states.clear()
        self.window_kv_states.clear()
        self._window_meta.clear()
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seen_tokens

    def __len__(self) -> int:
        return max(len(self.recurrent_states), len(self.window_kv_states), 1)

    def __getitem__(self, layer_idx: int) -> Any:
        window = (
            self._ordered_window_kv(layer_idx)
            if layer_idx in self.window_kv_states
            else None
        )
        return (self.recurrent_states.get(layer_idx, None), window)


def benchmark_autoregressive_decoding(
    seq_lens: Sequence[int] = (512, 2048, 8192, 32768),
    batch_size: int = 1,
    embed_dim: int = 768,
    num_heads: int = 12,
    dtype: torch.dtype = torch.float16,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Measure CUDA decode-step latency and report corresponding cache footprints."""
    if not torch.cuda.is_available():
        raise NotImplementedError("benchmark() requires CUDA; cannot fabricate latency")
    if device is not None and torch.device(device).type != "cuda":
        raise ValueError("autoregressive latency benchmarking requires a CUDA device")
    device = torch.device(device or "cuda")
    if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
        raise ValueError("embed_dim must be positive and divisible by num_heads")
    if batch_size <= 0 or any(int(length) <= 0 for length in seq_lens):
        raise ValueError("batch_size and all sequence lengths must be positive")

    head_dim = embed_dim // num_heads
    dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    records = []
    recurrent_state_bytes = batch_size * num_heads * head_dim * head_dim * dtype_bytes
    recurrent_mb = float(recurrent_state_bytes / (1024 * 1024))

    for T in seq_lens:
        standard_kv_bytes = 2 * batch_size * num_heads * T * head_dim * dtype_bytes
        standard_kv_mb = float(standard_kv_bytes / (1024 * 1024))

        q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(batch_size, num_heads, T, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        recurrent_state = torch.randn(
            batch_size, num_heads, head_dim, head_dim, device=device, dtype=dtype
        )
        recurrent_key = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
        recurrent_value = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
        recurrent_normalizer = torch.randn(
            batch_size, num_heads, 1, head_dim, device=device, dtype=dtype
        )

        def standard_decode_step() -> torch.Tensor:
            scores = torch.matmul(q, k_cache.transpose(-2, -1)) / (head_dim**0.5)
            return torch.matmul(torch.softmax(scores.float(), dim=-1).to(dtype), v_cache)

        def recurrent_decode_step() -> torch.Tensor:
            updated_state = recurrent_state + torch.matmul(
                recurrent_key.transpose(-2, -1), recurrent_value
            )
            updated_normalizer = recurrent_normalizer + recurrent_key
            numerator = torch.matmul(q, updated_state)
            denominator = torch.matmul(q, updated_normalizer.transpose(-2, -1)).clamp_min(1e-6)
            return numerator / denominator

        def measure_ms(fn: Callable[[], torch.Tensor], repeats: int = 10) -> float:
            for _ in range(3):
                fn()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end) / repeats)

        std_latency_ms = measure_ms(standard_decode_step)
        neuralops_latency_ms = measure_ms(recurrent_decode_step)

        records.append({
            "seq_len": T,
            "standard_kv_cache_mb": round(standard_kv_mb, 3),
            "neuralops_state_cache_mb": round(recurrent_mb, 3),
            "memory_reduction_ratio": round(standard_kv_mb / max(recurrent_mb, 1e-6), 2),
            "standard_latency_ms": round(std_latency_ms, 6),
            "neuralops_latency_ms": round(neuralops_latency_ms, 6),
            "speedup_x": round(std_latency_ms / neuralops_latency_ms, 2),
        })

    return {
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "results": records,
    }
