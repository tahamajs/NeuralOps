#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""CLI wrapper for the existing real perplexity evaluation adapter."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.training_throughput import TinyCausalLM


def run_perplexity_evaluation(**kwargs: Any) -> dict[str, Any]:
    """Evaluate a language model on a real tokenized dataset."""
    adapter = PerplexityAdapter(**kwargs)
    try:
        result = adapter.run()
        return {"category": "perplexity", "status": "PASS", "metrics": result}
    finally:
        adapter.cleanup()


def run_synthetic_perplexity(
    *, device: str = "cuda", batches: int = 3, batch_size: int = 2,
    seq_len: int = 32,
) -> dict[str, Any]:
    """Measure perplexity on a deterministic synthetic token stream."""
    model = TinyCausalLM().to(device).eval()
    total_nll = 0.0
    total_tokens = 0
    started = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    ended = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if started is not None:
        started.record()
    for batch in range(batches):
        start = batch * batch_size * seq_len
        tokens = torch.arange(start, start + batch_size * seq_len, device=device).reshape(batch_size, seq_len) % 256
        with torch.no_grad():
            logits = model(tokens).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1))
        total_nll += float(loss.item()) * (tokens.numel() - batch_size)
        total_tokens += tokens.numel() - batch_size
    if ended is not None:
        ended.record()
        ended.synchronize()
        elapsed_ms = float(started.elapsed_time(ended))
    else:
        elapsed_ms = 0.0
    return {
        "module": "synthetic_tiny_lm",
        "category": "perplexity",
        "status": "PASS" if total_tokens > 0 and math.isfinite(total_nll) else "FAIL",
        "perplexity": math.exp(total_nll / total_tokens),
        "eval_latency_ms": elapsed_ms,
        "tokens": total_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate perplexity and latency")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--module", default="baseline")
    parser.add_argument("--module-type", default="attention", choices=["attention", "compressor"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic tiny-model data without downloads")
    args = parser.parse_args()
    result = run_synthetic_perplexity(device=args.device) if args.synthetic else run_perplexity_evaluation(
        module_name=args.module, module_type=args.module_type,
        model_name=args.model, dataset_name=args.dataset, device=args.device,
    )
    payload = json.dumps(result, indent=2, default=str)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
