#!/usr/bin/env python3
"""
High-Performance Token Generation Benchmark for NeuralOps & FLA Models.
Measures Time-to-First-Token (TTFT), inter-token generation latency (ms/token),
and total generation throughput (tokens/second) across prompt lengths and generation budgets.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerationBenchmark:
    """Benchmark autoregressive generation latency and throughput."""

    def __init__(
        self,
        model: Union[str, nn.Module] = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=self.dtype,
            ).to(self.device).eval()
        else:
            self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def benchmark_generate(
        self,
        prompt_len: int = 64,
        gen_len: int = 32,
        batch_size: int = 1,
        warmup_runs: int = 2,
        num_runs: int = 3,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Run generation benchmark and return metrics.
        Returns:
            dict containing ttft_ms, latency_per_token_ms, throughput_tok_per_s
        """
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 1000)
        input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=self.device)

        # Warmup
        for _ in range(warmup_runs):
            _ = self.model.generate(
                input_ids,
                max_new_tokens=min(gen_len, 4),
                do_sample=False,
                use_cache=use_cache,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        ttft_list = []
        token_latency_list = []

        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Measure TTFT (prefill pass)
            out = self.model(input_ids, use_cache=use_cache)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1000.0
            ttft_list.append(ttft)

            # Measure autoregressive decode pass
            t_gen_start = time.perf_counter()
            gen_out = self.model.generate(
                input_ids,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=use_cache,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_gen_time = (time.perf_counter() - t_gen_start) * 1000.0

            actual_new_tokens = gen_out.shape[1] - input_ids.shape[1]
            if actual_new_tokens > 0:
                token_latency_list.append(total_gen_time / (actual_new_tokens * batch_size))

        avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
        avg_tok_lat = sum(token_latency_list) / len(token_latency_list) if token_latency_list else 0.0
        throughput = (1000.0 / avg_tok_lat * batch_size) if avg_tok_lat > 0 else 0.0

        return {
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "batch_size": batch_size,
            "ttft_ms": round(avg_ttft, 3),
            "latency_per_token_ms": round(avg_tok_lat, 3),
            "throughput_tokens_per_s": round(throughput, 2),
        }
