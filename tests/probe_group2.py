#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Probe script to test Group 2 compressors (keys 18-34) on MLP and BERT.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import traceback

# Ensure repository root is first in sys.path and tests directory does not shadow stdlib
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if len(sys.path) > 1 and os.path.basename(sys.path[1]) == "tests":
    sys.path.pop(1)
if len(sys.path) > 0 and os.path.basename(sys.path[0]) == "tests":
    sys.path.pop(0)
    sys.path.insert(0, _repo_root)

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import neuralops.compression
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor

GROUP2_KEYS = [
    "coopq_quantizer",
    "cr_compressor",
    "cur_compressor",
    "cur_moe_compressor",
    "delta_kv",
    "direct_svd",
    "dobi_svd",
    "drone_compressor",
    "dynamic_rank",
    "enlighten_compressor",
    "entropy_budget_allocation",
    "erc_svd",
    "faq_quantizer",
    "flap",
    "flat_llm",
    "fp8",
    "frandi_compressor",
]


class TwoLayerMLP(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def get_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_sparsity(model: nn.Module) -> float:
    total_zeros = 0
    total_elements = 0
    for p in model.parameters():
        total_elements += p.numel()
        total_zeros += (p == 0).sum().item()
    if total_elements == 0:
        return 0.0
    return total_zeros / total_elements


def apply_compressor(name: str, model: nn.Module, rank: int = 128) -> Tuple[nn.Module, Any]:
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Compressor '{name}' not found in registry")

    kwargs = {
        "rank": rank,
        "fallback_to_original": False,
        "max_error": 1.0,
        "target_attention_only": False,
    }
    comp_obj = None
    try:
        comp_obj = create_compressor(name, model=model, **kwargs)
    except Exception:
        try:
            comp_obj = cls(model=model, **kwargs)
        except Exception:
            try:
                comp_obj = cls(model, **kwargs)
            except Exception:
                try:
                    comp_obj = cls(model=model)
                except Exception:
                    try:
                        comp_obj = cls(model)
                    except Exception:
                        comp_obj = cls()

    compressed = None
    if hasattr(comp_obj, "compress"):
        res = comp_obj.compress()
        compressed = res[0] if isinstance(res, tuple) else res
    elif callable(comp_obj) and comp_obj is not model:
        try:
            res = comp_obj(model, **kwargs)
        except Exception:
            res = comp_obj(model)
        if isinstance(res, tuple):
            compressed = res[0]
        elif hasattr(res, "compress"):
            c_res = res.compress()
            compressed = c_res[0] if isinstance(c_res, tuple) else c_res
        else:
            compressed = res
    else:
        compressed = model

    return compressed, comp_obj


def test_compressor_on_model(name: str, model: nn.Module, is_bert: bool = False) -> Dict[str, Any]:
    orig_model = copy.deepcopy(model)
    orig_params = get_model_params(orig_model)
    orig_model.eval()

    if is_bert:
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        with torch.no_grad():
            orig_out = orig_model(input_ids).last_hidden_state
    else:
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 768)
        with torch.no_grad():
            orig_out = orig_model(x)

    try:
        compressed_model, comp_obj = apply_compressor(name, model, rank=128)
    except Exception as e:
        return {
            "status": "FAIL_COMPRESS",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }

    if compressed_model is None or not isinstance(compressed_model, nn.Module):
        return {
            "status": "FAIL_INVALID_MODULE",
            "error": f"Result is not nn.Module: {type(compressed_model)}",
        }

    comp_params = get_model_params(compressed_model)
    sparsity = get_model_sparsity(compressed_model)
    sparsity_ratio = 1.0 / (1.0 - sparsity + 1e-9) if sparsity > 0.0 else 1.0

    # Check bit reduction if attribute exists or quantizer
    bit_reduction = 1.0
    for obj in [comp_obj, compressed_model]:
        if hasattr(obj, "effective_bits"):
            bit_reduction = max(bit_reduction, 32.0 / getattr(obj, "effective_bits"))
        if hasattr(obj, "bits"):
            bit_reduction = max(bit_reduction, 32.0 / getattr(obj, "bits"))
        if hasattr(obj, "bit_reduction"):
            bit_reduction = max(bit_reduction, float(getattr(obj, "bit_reduction")))
        if hasattr(obj, "compression_ratio"):
            cr = float(getattr(obj, "compression_ratio"))
            if cr > 1.05:
                bit_reduction = max(bit_reduction, cr)

    param_ratio = orig_params / max(comp_params, 1)
    effective_ratio = max(param_ratio, sparsity_ratio, bit_reduction)

    # Forward pass
    try:
        compressed_model.eval()
        with torch.no_grad():
            if is_bert:
                cand_out = compressed_model(input_ids)
                if hasattr(cand_out, "last_hidden_state"):
                    cand_out = cand_out.last_hidden_state
                elif isinstance(cand_out, tuple):
                    cand_out = cand_out[0]
            else:
                cand_out = compressed_model(x)
                if hasattr(cand_out, "last_hidden_state"):
                    cand_out = cand_out.last_hidden_state
                elif isinstance(cand_out, tuple):
                    cand_out = cand_out[0]
    except Exception as e:
        return {
            "status": "FAIL_FORWARD",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
            "orig_params": orig_params,
            "comp_params": comp_params,
            "param_ratio": param_ratio,
        }

    is_finite = bool(torch.isfinite(cand_out).all().item())
    if not is_finite:
        return {
            "status": "FAIL_NON_FINITE",
            "error": "Outputs contain NaN or Inf",
            "orig_params": orig_params,
            "comp_params": comp_params,
            "param_ratio": param_ratio,
        }

    # Relative reconstruction error
    eps = 1e-12
    ref_f = orig_out.float()
    cand_f = cand_out.float()
    diff = ref_f - cand_f
    rel_error = float((torch.linalg.norm(diff) / (torch.linalg.norm(ref_f) + eps)).item())

    passed = effective_ratio > 1.05 and is_finite

    return {
        "status": "PASS" if passed else "FAIL_RATIO",
        "orig_params": orig_params,
        "comp_params": comp_params,
        "param_ratio": param_ratio,
        "sparsity": sparsity,
        "sparsity_ratio": sparsity_ratio,
        "bit_reduction": bit_reduction,
        "effective_ratio": effective_ratio,
        "rel_error": rel_error,
        "is_finite": is_finite,
    }


def main():
    print("=" * 80)
    print("PROBING GROUP 2 COMPRESSORS (18-34)")
    print("=" * 80)

    bert_cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=1,
        intermediate_size=3072,
        vocab_size=1000,
    )

    results = {}
    for key in GROUP2_KEYS:
        print(f"\n---> Testing {key} ...")
        # Test MLP
        mlp = TwoLayerMLP(hidden_size=768)
        mlp_res = test_compressor_on_model(key, mlp, is_bert=False)
        print(f"  MLP:  status={mlp_res.get('status')} eff_ratio={mlp_res.get('effective_ratio', 1.0):.2f}x err={mlp_res.get('rel_error', float('nan')):.4f}")
        if "error" in mlp_res:
            print(f"        error: {mlp_res['error']}")

        # Test BERT
        bert = BertModel(bert_cfg)
        bert_res = test_compressor_on_model(key, bert, is_bert=True)
        print(f"  BERT: status={bert_res.get('status')} eff_ratio={bert_res.get('effective_ratio', 1.0):.2f}x err={bert_res.get('rel_error', float('nan')):.4f}")
        if "error" in bert_res:
            print(f"        error: {bert_res['error']}")

        results[key] = {
            "mlp": mlp_res,
            "bert": bert_res,
        }

    os.makedirs("results/compressor_zoo", exist_ok=True)
    out_path = "results/compressor_zoo/group2_verified.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary saved to {out_path}")
    print("=" * 80)

    # Print summary table
    print(f"{'Key':<28} | {'MLP Status':<10} | {'MLP Ratio':<10} | {'BERT Status':<11} | {'BERT Ratio':<10}")
    print("-" * 80)
    all_passed = True
    for key, data in results.items():
        mlp_s = data["mlp"].get("status", "UNKNOWN")
        mlp_r = data["mlp"].get("effective_ratio", 1.0)
        bert_s = data["bert"].get("status", "UNKNOWN")
        bert_r = data["bert"].get("effective_ratio", 1.0)
        if mlp_s != "PASS" or bert_s != "PASS":
            all_passed = False
        print(f"{key:<28} | {mlp_s:<10} | {mlp_r:<9.2f}x | {bert_s:<11} | {bert_r:<9.2f}x")
    print("-" * 80)
    print(f"ALL 17 COMPRESSORS PASSED: {all_passed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
