#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import sys
# Python 3.14 has a standard library module named 'compression'.
# If sys.path[0] is tests/, tests/compression shadows the stdlib compression module.
if sys.path and sys.path[0].endswith("tests"):
    sys.path.pop(0)
if "." not in sys.path:
    sys.path.insert(0, ".")

import copy
import json
import math
import os
import traceback
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import COMPRESSOR_REGISTRY

ASSIGNED_KEYS = [
    "mgie_budget",
    "mlorq_compressor",
    "modegpt_compressor",
    "nmf_compressor",
    "oats_compressor",
    "optimal_subspace_projection",
    "proxsparse",
    "quarot",
    "residual_svd_compressor",
    "roast_hashed_linear",
    "rotational_slicegpt_int4",
    "rpca_compressor",
    "saes_svd_v2",
    "salr",
    "sdrm_compressor",
    "slicegpt",
]

def make_mlp():
    return nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768),
    )

def make_bert():
    cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=1,
        intermediate_size=3072,
        vocab_size=1000,
    )
    return BertModel(cfg)

def apply_compressor(key: str, model: nn.Module, rank: int = 16):
    target = COMPRESSOR_REGISTRY[key]
    kwargs = {
        "rank": rank,
        "target_rank": rank,
        "fallback_to_original": False,
        "max_error": 1.0,
    }
    # Try different invocation signatures
    instance = None
    for fn in [
        lambda: target(model, **kwargs),
        lambda: target(model, rank=rank),
        lambda: target(model),
    ]:
        try:
            instance = fn()
            break
        except Exception as e:
            continue
    
    if instance is None:
        raise RuntimeError(f"Could not instantiate compressor {key}")
    
    stats = {}
    if isinstance(instance, tuple):
        comp_model = instance[0]
        if len(instance) > 1 and isinstance(instance[1], dict):
            stats = instance[1]
    elif hasattr(instance, "compress"):
        try:
            res = instance.compress()
        except TypeError:
            res = instance.compress(model)
        if isinstance(res, tuple):
            comp_model = res[0]
            if len(res) > 1 and isinstance(res[1], dict):
                stats = res[1]
        else:
            comp_model = res
    elif isinstance(instance, nn.Module):
        comp_model = instance
    else:
        comp_model = getattr(instance, "model", model)
        if hasattr(instance, "get_compression_stats"):
            stats = instance.get_compression_stats()
            
    return comp_model, stats

def test_model(key: str, model_type: str):
    torch.manual_seed(42)
    if model_type == "mlp":
        orig_model = make_mlp()
        x = torch.randn(2, 768)
        with torch.no_grad():
            orig_out = orig_model(x)
    else:
        orig_model = make_bert()
        x = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            orig_out = orig_model(x).last_hidden_state
            
    orig_params = sum(p.numel() for p in orig_model.parameters())
    orig_nonzero = sum((p != 0).sum().item() for p in orig_model.parameters())
    
    comp_model, stats = apply_compressor(key, orig_model, rank=16)
    
    new_params = sum(p.numel() for p in comp_model.parameters())
    new_nonzero = sum((p != 0).sum().item() for p in comp_model.parameters())
    
    # Calculate ratios
    param_ratio = orig_params / max(1, new_params)
    sparsity_ratio = orig_nonzero / max(1, new_nonzero)
    sparsity = max(0.0, 1.0 - (new_nonzero / max(1, new_params)))
    
    # Bit reduction
    bits = stats.get("bits") or stats.get("quant_bits") or (4 if "int4" in key or key == "quarot" else 32)
    bit_reduction = 32.0 / float(bits) if bits < 32 else 1.0
    effective_ratio = max(param_ratio, sparsity_ratio, bit_reduction)
    
    # Test forward
    with torch.no_grad():
        if model_type == "mlp":
            comp_out = comp_model(x)
        else:
            res = comp_model(x)
            comp_out = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
            
    finite = torch.isfinite(comp_out).all().item()
    rel_error = (torch.norm(comp_out - orig_out) / torch.norm(orig_out).clamp_min(1e-6)).item()
    passed = effective_ratio > 1.05 and finite
    
    return {
        "status": "PASS" if passed else "FAIL_RATIO",
        "orig_params": orig_params,
        "comp_params": new_params,
        "param_ratio": param_ratio,
        "sparsity": sparsity,
        "sparsity_ratio": sparsity_ratio,
        "bit_reduction": bit_reduction,
        "effective_ratio": effective_ratio,
        "rel_error": rel_error,
        "is_finite": finite,
        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))},
    }

def main():
    results = {}
    print(f"{'Key':<30} | {'MLP Status':<12} | {'MLP Ratio':<10} | {'BERT Status':<12} | {'BERT Ratio':<10}")
    print("-" * 85)
    for key in ASSIGNED_KEYS:
        mlp_res, bert_res = None, None
        mlp_err, bert_err = None, None
        try:
            mlp_res = test_model(key, "mlp")
        except Exception as e:
            mlp_err = f"{type(e).__name__}: {e}"
            
        try:
            bert_res = test_model(key, "bert")
        except Exception as e:
            bert_err = f"{type(e).__name__}: {e}"
            
        mlp_status = mlp_res["status"] if mlp_res else "ERROR"
        bert_status = bert_res["status"] if bert_res else "ERROR"
        
        mlp_ratio_str = f"{mlp_res['effective_ratio']:.2f}x" if mlp_res else "N/A"
        bert_ratio_str = f"{bert_res['effective_ratio']:.2f}x" if bert_res else "N/A"
        
        print(f"{key:<30} | {mlp_status:<12} | {mlp_ratio_str:<10} | {bert_status:<12} | {bert_ratio_str:<10}")
        if mlp_err:
            print(f"  [MLP Error] {mlp_err[:120]}")
        if bert_err:
            print(f"  [BERT Error] {bert_err[:120]}")
            
        results[key] = {
            "mlp": mlp_res or {"error": mlp_err},
            "bert": bert_res or {"error": bert_err},
        }

    os.makedirs("results/compressor_zoo", exist_ok=True)
    with open("results/compressor_zoo/group4_verified.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nVerified results written to results/compressor_zoo/group4_verified.json")

if __name__ == "__main__":
    main()
