#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 0 — Discover instantiable NeuralOps attention keys for BERT (embed_dim=768, num_heads=12).

Attempts create_attention(key, embed_dim=768, num_heads=12) with a random input for each
registered key. Writes two JSON files:
  - registry_keys.json  : list of keys that instantiated and ran without error
  - skipped_keys.json   : list of {key, error} dicts for keys that failed

Exit code: 0 always (skipped keys are not fatal).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bert_zoo_discover")

EMBED_DIM = 768
NUM_HEADS = 12
SEQ_LEN = 16   # small probe input
BATCH = 2


import subprocess

def probe_attention(key: str, device: torch.device) -> str | None:
    """
    Try to instantiate and forward-pass one attention module with BertAttentionAdapter
    in an isolated subprocess. This guarantees that C++/CUDA/Triton JIT aborts
    (SIGABRT/SIGSEGV) do not crash the discovery daemon.
    Returns None on success, error string on failure.
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"
    probe_code = f"""
import sys, os, torch, gc
sys.path.insert(0, os.path.abspath("."))
from neuralops.core.registry import create_attention
from neuralops.layers.bert_adapter import BertAttentionAdapter

device = torch.device("{device_str}")
raw_mod = create_attention("{key}", embed_dim={EMBED_DIM}, num_heads={NUM_HEADS})
mod = BertAttentionAdapter(raw_mod, embed_dim={EMBED_DIM}, num_heads={NUM_HEADS})
mod = mod.to(device)
mod.eval()
with torch.no_grad():
    x = torch.randn({BATCH}, {SEQ_LEN}, {EMBED_DIM}, device=device)
    out, _ = mod(x)
    assert out.shape[-1] == {EMBED_DIM}, f"bad output shape {{out.shape}}"
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=35,
        )
        if proc.returncode == 0:
            return None
        else:
            err = proc.stderr.strip() or proc.stdout.strip() or f"Process exited with code {proc.returncode}"
            return err
    except subprocess.TimeoutExpired:
        return "Probe timed out after 35 seconds"
    except Exception as e:
        return str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover instantiable NeuralOps attention keys")
    parser.add_argument("--output", default="results/bert_attention_zoo/registry_keys.json",
                        help="Path to write the valid keys JSON list")
    parser.add_argument("--skipped_output", default="results/bert_attention_zoo/skipped_keys.json",
                        help="Path to write the skipped keys JSON list")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available. Aborting.")
        sys.exit(3)

    from neuralops.layers import ATTENTION_REGISTRY
    all_keys = sorted(ATTENTION_REGISTRY.keys())
    logger.info(f"Registry total: {len(all_keys)} keys")
    logger.info(f"Device: {device}")

    valid_keys: list[str] = []
    skipped: list[dict] = []

    t0 = time.time()
    for i, key in enumerate(all_keys):
        logger.info(f"[{i+1:03d}/{len(all_keys):03d}] Probing: {key}")
        err = probe_attention(key, device)
        if err is None:
            valid_keys.append(key)
            logger.info(f"  OK")
        else:
            first_line = err.strip().splitlines()[-1]
            logger.warning(f"  SKIP: {first_line}")
            skipped.append({"key": key, "error": err.strip()})

    elapsed = time.time() - t0
    logger.info(
        f"\nDiscovery complete in {elapsed:.1f}s: "
        f"{len(valid_keys)} valid, {len(skipped)} skipped out of {len(all_keys)} total."
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.skipped_output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(valid_keys, f, indent=2)
    logger.info(f"Valid keys written to: {args.output}")

    with open(args.skipped_output, "w") as f:
        json.dump(skipped, f, indent=2)
    logger.info(f"Skipped keys written to: {args.skipped_output}")


if __name__ == "__main__":
    main()
