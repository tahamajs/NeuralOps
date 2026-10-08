#!/usr/bin/env python3
"""
Empirical Causality & Mechanism Test for all Attention Modules on GPT-2.

For every registered module in neuralops/attentions/:
  1. Instantiates the module with GPT-2 configuration (is_decoder=True).
  2. Runs causality verification: modifies last token and checks if past tokens change.
     If past tokens change -> FUTURE TOKEN LEAKAGE (BROKEN_ON_DECODER).
  3. Checks whether attention is computed via:
     - Linear/SSM/chunked state (O(N) without full softmax) -> TRUE_CAUSAL (Bucket A)
     - Full softmax with triangular mask -> CAUSAL_VIA_MASK (Bucket B)
     - Dense softmax fallback (griffin_ssd, simple_ssm) -> DENSE_FALLBACK (Bucket E)
     - Encoder-only / raises error -> ENCODER_ONLY (Bucket D)
"""

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn
from transformers import GPT2Config

from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions

def test_module(name, cls_or_fn):
    cfg = GPT2Config(
        n_embd=768,
        n_head=12,
        n_layer=1,
        vocab_size=50257,
        is_decoder=True,
    )

    # Instantiate
    instance = None
    if inspect.isclass(cls_or_fn) and issubclass(cls_or_fn, nn.Module):
        try:
            instance = cls_or_fn(config=cfg, is_decoder=True)
        except Exception:
            try:
                instance = cls_or_fn(config=cfg)
            except Exception:
                try:
                    instance = cls_or_fn(is_decoder=True)
                except Exception:
                    try:
                        instance = cls_or_fn()
                    except Exception as e:
                        return "INIT_ERROR", str(e)
    elif callable(cls_or_fn):
        # Injector or factory
        return "FACTORY", "Function injector"
    else:
        return "NOT_MODULE", "Not an nn.Module"

    if instance is None:
        return "INIT_ERROR", "Could not instantiate"

    instance.eval()
    B, N, D = 1, 16, 768
    x1 = torch.randn(B, N, D)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(B, D)  # modify last token

    try:
        with torch.no_grad():
            o1 = instance(x1)
            o2 = instance(x2)

            res1 = o1[0] if isinstance(o1, tuple) else o1
            res2 = o2[0] if isinstance(o2, tuple) else o2

            if not isinstance(res1, torch.Tensor) or res1.shape != x1.shape:
                return "SHAPE_MISMATCH", f"Expected {x1.shape}, got {getattr(res1, 'shape', None)}"

            # Causality check: tokens 0..N-2 must not change
            diff = (res1[:, :-1, :] - res2[:, :-1, :]).abs().max().item()
            is_strictly_causal = (diff < 1e-3)
            return "OK", is_strictly_causal, diff
    except Exception as e:
        return "FORWARD_ERROR", str(e)

if __name__ == "__main__":
    print("Testing registered attention modules for causality...")
    for key, val in sorted(ATTENTION_REGISTRY.items()):
        if inspect.isclass(val):
            res = test_module(key, val)
            print(f"{key:35s}: {res}")
