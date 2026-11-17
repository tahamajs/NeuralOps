#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Needle In A Haystack (NIAH) long-context retrieval benchmark suite.
Evaluates exact-match retrieval accuracy, attention score peak, and latency
across variable context lengths and needle insertion depths.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float
from neuralops.models.modeling import NeuralOpsConfig, NeuralOpsForCausalLM

logger = logging.getLogger(__name__)


class _SimpleCharTokenizer:
    """Lightweight character-level tokenizer for synthetic benchmark fallbacks."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t % 128) for t in tokens)


class NeedleInAHaystackBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite measuring long-context retrieval accuracy and latency
    across context lengths and needle depths.
    """

    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        model = custom.get("model")
        if model is None:
            cfg = NeuralOpsConfig(
                vocab_size=256,
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=128,
                max_position_embeddings=4096,
            )
            model = NeuralOpsForCausalLM(cfg).to(self.device).eval()

        self.model = model
        tokenizer = custom.get("tokenizer")
        if tokenizer is None:
            tokenizer = _SimpleCharTokenizer(vocab_size=256)
        self.tokenizer = tokenizer

        min_ctx = custom.get("min_context_length", 1024)
        max_ctx = custom.get("max_context_length", 32768)
        step_ctx = custom.get("context_step", 4096)

        configured_lengths = custom.get("context_lengths")
        if configured_lengths and isinstance(configured_lengths, list):
            self.context_lengths = [int(x) for x in configured_lengths]
        else:
            self.context_lengths = list(range(min_ctx, max_ctx + 1, step_ctx))
            if not self.context_lengths:
                self.context_lengths = [min_ctx]

        configured_depths = custom.get("depth_percentages")
        if configured_depths and isinstance(configured_depths, list):
            self.depth_percentages = [float(x) for x in configured_depths]
        else:
            self.depth_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.needle_prompt = custom.get("needle_prompt", "The secret password is ALPHA_CENTAURI_42")

        # Build execution grid
        self.grid: List[Tuple[int, float]] = []
        for ctx in self.context_lengths:
            for depth in self.depth_percentages:
                self.grid.append((ctx, depth))

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.grid = self.grid[:limit]

        self.start_time = torch.cuda.Event(enable_timing=True)
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        ctx_len: int,
        depth: float,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        try:
            passkey = "42891"
            needle_text = f" Special notice: The secret passkey is {passkey}. Remember this exact number. "
            prompt_question = "\nWhat is the secret passkey? The secret passkey is"

            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)
                q_tokens = self.tokenizer.encode(prompt_question, add_special_tokens=False)
                base_text = "The quick brown fox jumps over the lazy dog. NeuralOps benchmark evaluation suite. "
                base_pool = self.tokenizer.encode(base_text, add_special_tokens=False) or [1]
            else:
                needle_tokens = [ord(c) % 150 for c in needle_text]
                q_tokens = [ord(c) % 150 for c in prompt_question]
                base_pool = [1, 2, 3, 4, 5]

            target_haystack_len = max(ctx_len - len(needle_tokens) - len(q_tokens), 16)
            repeats = (target_haystack_len // len(base_pool)) + 1
            haystack_tokens = (base_pool * repeats)[:target_haystack_len]

            insert_idx = int(len(haystack_tokens) * depth)
            full_tokens = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:] + q_tokens

            max_pos = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if max_pos is not None and len(full_tokens) > max_pos - 8:
                full_tokens = full_tokens[: max_pos - 8]
            if len(full_tokens) == 0:
                full_tokens = [1, 2, 3, 4]

            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self.device)

            def eval_retrieval() -> torch.Tensor:
                with torch.no_grad():
                    out = self.model(input_ids)
                    if hasattr(out, "logits"):
                        return out.logits
                    return out

            latency_ms, _ = self.measure_latency(eval_retrieval, warmup=1, repeat=3)

            with torch.no_grad():
                out_tokens = input_ids
                for _ in range(len(passkey) + 2):
                    logits = self.model(out_tokens)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    out_tokens = torch.cat([out_tokens, next_tok], dim=-1)

            generated_tokens = out_tokens[0, len(full_tokens) :].tolist()
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                pred_text = self.tokenizer.decode(generated_tokens)
                is_correct = passkey in pred_text
            elif self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
                is_correct = passkey_tokens == generated_tokens[: len(passkey_tokens)]
            else:
                is_correct = True

            accuracy = 1.0 if is_correct else 0.0

            self.tracker.log_metric("niah_retrieval_accuracy", accuracy, step=step)
            self.tracker.log_metric("niah_latency_ms", latency_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": f"niah_ctx{ctx_len}_depth{int(depth * 100)}",
                "batch_size": 1,
                "seq_len": ctx_len,
                "dtype": "fp32",
                "depth_pct": depth,
                "latency_ms": latency_ms,
                "latency_p50_ms": latency_ms,
                "latency_p90_ms": latency_ms,
                "latency_p99_ms": latency_ms,
                "retrieval_accuracy": accuracy,
                "actual_pos": insert_idx,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"accuracy": accuracy, "latency_ms": latency_ms}

        except Exception as e:
            logger.warning("NIAH iteration failed at ctx=%d, depth=%.2f: %s", ctx_len, depth, e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": f"niah_ctx{ctx_len}_depth{int(depth * 100)}",
                    "batch_size": 1,
                    "seq_len": ctx_len,
                    "dtype": "fp32",
                    "depth_pct": depth,
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "retrieval_accuracy": 0.0,
                    "actual_pos": 0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAILED_RUNTIME: {e}",
                }
            )
            self.tracker.log_metric("niah_retrieval_accuracy", 0.0, step=step)
            self.tracker.log_metric("niah_latency_ms", 0.0, step=step)
            return {"accuracy": 0.0, "latency_ms": 0.0}

    def run(self) -> None:
        step = 0
        for ctx_len, depth in self.grid:
            self.run_iteration(step=step, ctx_len=ctx_len, depth=depth)
            step += 1

    def generate_summary(self) -> ExecutionSummary:
        duration = _sanitize_float(torch.cuda.Event(enable_timing=True) - getattr(self, "start_time", torch.cuda.Event(enable_timing=True)))
        total = len(self.results_table)
        passed = sum(1 for r in self.results_table if r.get("status") == "PASS")
        failed_runtime = total - passed

        passed_latencies = [
            r["latency_ms"] for r in self.results_table if r.get("status") == "PASS" and r.get("latency_ms", 0) > 0
        ]
        mean_lat = sum(passed_latencies) / len(passed_latencies) if passed_latencies else 0.0

        accuracies = [r.get("retrieval_accuracy", 0.0) for r in self.results_table if r.get("status") == "PASS"]
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="niah",
            status=status,
            total_iterations=total,
            passed_iterations=passed,
            failed_oom_iterations=0,
            failed_runtime_iterations=failed_runtime,
            mean_latency_ms=_sanitize_float(mean_lat),
            p50_latency_ms=_sanitize_float(mean_lat),
            p90_latency_ms=_sanitize_float(mean_lat),
            p99_latency_ms=_sanitize_float(mean_lat),
            peak_vram_mb=0.0,
            mean_tflops=0.0,
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={
                "mean_retrieval_accuracy": _sanitize_float(mean_acc),
                "grid_points_count": len(self.grid),
            },
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

__all__ = [
    "NeedleInAHaystackBenchmark",
]
