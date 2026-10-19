#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Small deterministic serving primitives for TTFT and inter-token latency."""

from __future__ import annotations

import time
from typing import Any

import torch


class PagedKVCache:
    """Minimal page allocator used to measure cache writes without a model dependency."""

    def __init__(self, pages: int, page_tokens: int, heads: int, head_dim: int, device: torch.device) -> None:
        self.storage = torch.empty(pages, 2, page_tokens, heads, head_dim, device=device)
        self.position = 0

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        key = key.reshape(-1, self.storage.shape[-2], self.storage.shape[-1])
        value = value.reshape(-1, self.storage.shape[-2], self.storage.shape[-1])
        n = key.shape[0]
        if self.position + n > self.storage.shape[0] * self.storage.shape[2]:
            raise RuntimeError("PagedKVCache capacity exceeded")
        flat = self.storage.view(2, -1, self.storage.shape[-2], self.storage.shape[-1])
        flat[0, self.position:self.position + n].copy_(key)
        flat[1, self.position:self.position + n].copy_(value)
        self.position += n


def run_benchmark(device: str = "cuda", prompt_tokens: int = 128, decode_tokens: int = 32) -> dict[str, Any]:
    """Measure prompt projection latency and cached decode-step latency."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    hidden, heads, head_dim = 512, 8, 64
    projection = torch.nn.Linear(hidden, hidden, device=target)
    prompt = torch.randn(1, prompt_tokens, hidden, device=target)
    cache = PagedKVCache(32, 16, heads, head_dim, target)
    started = time.perf_counter()
    projected = projection(prompt)
    ttft = (time.perf_counter() - started) * 1000.0
    key = projected.view(1, prompt_tokens, heads, head_dim).transpose(0, 1)
    cache.append(key, key)
    decode_start = time.perf_counter()
    for _ in range(decode_tokens):
        token = projection(torch.randn(1, 1, hidden, device=target))
        token_k = token.view(1, 1, heads, head_dim).transpose(0, 1)
        cache.append(token_k, token_k)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0 / decode_tokens
    return {"module": "paged_kv_serving", "category": "serving", "device": str(target), "ttft_ms": ttft, "itl_ms": decode_ms, "requests_per_second": 1000.0 / ttft if ttft else 0.0, "status": "PASS" if torch.isfinite(projected).all() else "FAIL"}
