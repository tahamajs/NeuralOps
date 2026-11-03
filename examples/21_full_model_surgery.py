#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run fail-closed attention surgery across every layer of a tiny model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import neuralops
from examples._shared import TinyClassifier, kv, pass_fail, section, tiny_inputs


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback.")
    return torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--attention", default="cosformer")
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    device = resolve_device(args.device)

    section("NeuralOps Example 21: Full-Model Surgery")
    model = TinyClassifier(n_layers=args.layers).to(device)
    inputs = {key: value.to(device) for key, value in tiny_inputs().items()}
    before = sum(parameter.numel() for parameter in model.parameters())

    surg_model = neuralops.replace_attention(
        model,
        attention=args.attention,
        copy_weights=True,
    )
    after = sum(parameter.numel() for parameter in surg_model.parameters())
    output = surg_model(**inputs)
    logits = output.logits if hasattr(output, "logits") else output
    loss = logits.float().square().mean()
    loss.backward()

    finite_output = bool(torch.isfinite(logits).all().item())
    gradients = [parameter.grad for parameter in surg_model.parameters() if parameter.requires_grad]
    finite_gradients = bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    kv("Device", str(device))
    kv("Attention", args.attention)
    kv("Parameters", f"{before} -> {after}")
    pass_fail("Full-model forward is finite", finite_output)
    pass_fail("All trainable gradients are present and finite", finite_gradients)
    return 0 if finite_output and finite_gradients else 1


if __name__ == "__main__":
    raise SystemExit(main())
