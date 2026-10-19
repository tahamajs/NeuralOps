#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""CLI and standardized wrapper for prefill/decode generation benchmarks."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from neuralops.benchmarks.generation import GenerationBenchmark


class TinyCausalLM(nn.Module):
    """Self-contained causal language model used for reproducible smoke runs."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 128) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        hidden, _ = self.rnn(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **_: Any) -> torch.Tensor:
        output = input_ids
        for _ in range(max_new_tokens):
            logits = self(output).logits[:, -1, :]
            output = torch.cat([output, logits.argmax(dim=-1, keepdim=True)], dim=1)
        return output


def run_generation_benchmark(model: str = "gpt2", **kwargs: Any) -> dict[str, Any]:
    """Run a real autoregressive generation measurement on the selected device."""
    device = kwargs.pop("device", None) or "cuda"
    benchmark_model = TinyCausalLM() if model == "tiny" else model
    metrics = GenerationBenchmark(model=benchmark_model, device=device).benchmark_generate(**kwargs)
    return {
        "module": model,
        "category": "generation",
        "status": "PASS" if metrics.get("ttft_ms", 0) > 0 and metrics.get("latency_per_token_ms", 0) > 0 else "FAIL",
        "cuda_latency_ms": metrics.get("ttft_ms", 0.0),
        "pytorch_latency_ms": metrics.get("latency_per_token_ms", 0.0),
        "speedup": 0.0,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure generation prefill and decode latency")
    parser.add_argument("--model", default="tiny", help="Checkpoint name or 'tiny' for a self-contained model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run_generation_benchmark(
        model=args.model, device=args.device,
        prompt_len=args.prompt_len, gen_len=args.gen_len,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
