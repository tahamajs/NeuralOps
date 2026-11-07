#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical deterministic reproducibility benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats


def _run_once(seed: int, device: torch.device) -> torch.Tensor:
    from neuralops.core.utils.general import set_seed
    from examples._shared.mock_models import TinyClassifier

    set_seed(seed, deterministic=True)
    model = TinyClassifier(dim=32, num_labels=2, seed=seed).to(device)
    model.eval()
    input_ids = torch.randint(0, 128, (4, 16), device=device)
    with torch.no_grad():
        out = model(input_ids)
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]
    return out


def run_reproducibility(config: BenchmarkConfig) -> BenchmarkResult:
    """Run identical seeded workloads twice and record exact reproducibility."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for reproducibility, but CUDA is unavailable.")
    device = torch.device(config.device)
    first = _run_once(config.seed, device)
    second = _run_once(config.seed, device)
    max_diff = float((first - second).abs().max().item())
    exact = bool(torch.equal(first, second))
    status = "success" if exact else "FAIL_REPRODUCIBILITY"
    result = BenchmarkResult(
        module={"name": "seeded_reproducibility_gate", "type": "reproducibility", "family": "core"},
        model={"name": "tiny_classifier_probe"},
        dataset={"name": "synthetic", "task": "reproducibility"},
        config={"device": str(device), "seed": config.seed},
        metrics={"max_diff": max_diff, "exact_match": exact, "reproducible": exact},
        status=status,
        error=None if exact else f"same seed produced max_diff={max_diff}",
    )
    save_all_formats(result, Path(config.output_dir))
    if not exact:
        raise RuntimeError(result.error or "Reproducibility gate failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_reproducibility(BenchmarkConfig.from_file(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
