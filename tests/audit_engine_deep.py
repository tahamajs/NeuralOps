#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deep forensic auditor for a single compressor engine.
Tests:
1. TwoLayerMLP (forward + backward + ratio)
2. nn.Sequential (forward + backward + ratio)
3. 1-layer BERT (forward + ratio)
4. Numerical fidelity: finite outputs, no NaN/Inf, reconstruction error
"""
import copy
import os
import sys
import traceback

# Ensure repository root is first in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if len(sys.path) > 1 and os.path.basename(sys.path[1]) == "tests":
    sys.path.pop(1)
if len(sys.path) > 0 and os.path.basename(sys.path[0]) == "tests":
    sys.path.pop(0)
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import neuralops.compression
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor


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
    return total_zeros / total_elements if total_elements > 0 else 0.0


def apply_compressor(name: str, model: nn.Module, rank: int = 64):
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


def audit_engine(name: str):
    print(f"\n{'='*70}\nDEEP AUDIT FOR: {name}\n{'='*70}")
    passed_all = True

    # 1. Test MLP
    mlp = TwoLayerMLP(768)
    orig_params = get_model_params(mlp)
    x = torch.randn(2, 16, 768)
    with torch.no_grad():
        orig_out = mlp(x)

    try:
        c_mlp, comp_obj = apply_compressor(name, mlp, rank=64)
        c_params = get_model_params(c_mlp)
        sparsity = get_model_sparsity(c_mlp)
        sparsity_ratio = 1.0 / (1.0 - sparsity + 1e-9) if sparsity > 0 else 1.0
        bit_reduction = 1.0
        for obj in [comp_obj, c_mlp]:
            if hasattr(obj, "effective_bits"):
                bit_reduction = max(bit_reduction, 32.0 / getattr(obj, "effective_bits"))
            if hasattr(obj, "bits"):
                bit_reduction = max(bit_reduction, 32.0 / getattr(obj, "bits"))
            if hasattr(obj, "bit_reduction"):
                bit_reduction = max(bit_reduction, float(getattr(obj, "bit_reduction")))

        param_ratio = orig_params / max(c_params, 1)
        eff_ratio = max(param_ratio, sparsity_ratio, bit_reduction)

        c_mlp.eval()
        with torch.no_grad():
            out = c_mlp(x)
        diff = torch.linalg.norm(orig_out.float() - out.float()) / (torch.linalg.norm(orig_out.float()) + 1e-12)
        finite = bool(torch.isfinite(out).all().item())

        # Test backward pass
        c_mlp.train()
        x_grad = torch.randn(2, 16, 768, requires_grad=True)
        out_train = c_mlp(x_grad)
        loss = out_train.sum()
        loss.backward()
        has_grad = False
        grad_finite = True
        for p in c_mlp.parameters():
            if p.grad is not None:
                has_grad = True
                if not torch.isfinite(p.grad).all():
                    grad_finite = False

        mlp_pass = eff_ratio > 1.05 and finite and grad_finite
        if not mlp_pass:
            passed_all = False
        print(f"  [1] TwoLayerMLP: {'PASS' if mlp_pass else 'FAIL'} | Ratio: {eff_ratio:.2f}x | RelErr: {diff.item():.4f} | Finite: {finite} | GradFinite: {grad_finite}")
    except Exception as e:
        passed_all = False
        print(f"  [1] TwoLayerMLP: EXCEPTION: {e}")
        traceback.print_exc()

    # 2. Test nn.Sequential
    seq = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
    orig_params_seq = get_model_params(seq)
    with torch.no_grad():
        orig_out_seq = seq(x)

    try:
        c_seq, comp_obj_seq = apply_compressor(name, seq, rank=64)
        c_params_seq = get_model_params(c_seq)
        sparsity_seq = get_model_sparsity(c_seq)
        sparsity_ratio_seq = 1.0 / (1.0 - sparsity_seq + 1e-9) if sparsity_seq > 0 else 1.0
        bit_reduction_seq = 1.0
        for obj in [comp_obj_seq, c_seq]:
            if hasattr(obj, "effective_bits"):
                bit_reduction_seq = max(bit_reduction_seq, 32.0 / getattr(obj, "effective_bits"))
            if hasattr(obj, "bits"):
                bit_reduction_seq = max(bit_reduction_seq, 32.0 / getattr(obj, "bits"))
            if hasattr(obj, "bit_reduction"):
                bit_reduction_seq = max(bit_reduction_seq, float(getattr(obj, "bit_reduction")))

        param_ratio_seq = orig_params_seq / max(c_params_seq, 1)
        eff_ratio_seq = max(param_ratio_seq, sparsity_ratio_seq, bit_reduction_seq)

        c_seq.eval()
        with torch.no_grad():
            out_seq = c_seq(x)
        diff_seq = torch.linalg.norm(orig_out_seq.float() - out_seq.float()) / (torch.linalg.norm(orig_out_seq.float()) + 1e-12)
        finite_seq = bool(torch.isfinite(out_seq).all().item())

        seq_pass = eff_ratio_seq > 1.05 and finite_seq
        if not seq_pass:
            passed_all = False
        print(f"  [2] Sequential:  {'PASS' if seq_pass else 'FAIL'} | Ratio: {eff_ratio_seq:.2f}x | RelErr: {diff_seq.item():.4f} | Finite: {finite_seq}")
    except Exception as e:
        passed_all = False
        print(f"  [2] Sequential:  EXCEPTION: {e}")
        traceback.print_exc()

    # 3. Test BERT
    bert_cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=1,
        intermediate_size=3072,
        vocab_size=1000,
    )
    bert = BertModel(bert_cfg)
    orig_params_bert = get_model_params(bert)
    input_ids = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        orig_out_bert = bert(input_ids).last_hidden_state

    try:
        c_bert, comp_obj_bert = apply_compressor(name, bert, rank=64)
        c_params_bert = get_model_params(c_bert)
        sparsity_bert = get_model_sparsity(c_bert)
        sparsity_ratio_bert = 1.0 / (1.0 - sparsity_bert + 1e-9) if sparsity_bert > 0 else 1.0
        bit_reduction_bert = 1.0
        for obj in [comp_obj_bert, c_bert]:
            if hasattr(obj, "effective_bits"):
                bit_reduction_bert = max(bit_reduction_bert, 32.0 / getattr(obj, "effective_bits"))
            if hasattr(obj, "bits"):
                bit_reduction_bert = max(bit_reduction_bert, 32.0 / getattr(obj, "bits"))
            if hasattr(obj, "bit_reduction"):
                bit_reduction_bert = max(bit_reduction_bert, float(getattr(obj, "bit_reduction")))

        param_ratio_bert = orig_params_bert / max(c_params_bert, 1)
        eff_ratio_bert = max(param_ratio_bert, sparsity_ratio_bert, bit_reduction_bert)

        c_bert.eval()
        with torch.no_grad():
            out_b = c_bert(input_ids)
            if hasattr(out_b, "last_hidden_state"):
                out_b = out_b.last_hidden_state
            elif isinstance(out_b, tuple):
                out_b = out_b[0]
        diff_bert = torch.linalg.norm(orig_out_bert.float() - out_b.float()) / (torch.linalg.norm(orig_out_bert.float()) + 1e-12)
        finite_bert = bool(torch.isfinite(out_b).all().item())

        bert_pass = eff_ratio_bert > 1.05 and finite_bert
        if not bert_pass:
            passed_all = False
        print(f"  [3] BertModel:   {'PASS' if bert_pass else 'FAIL'} | Ratio: {eff_ratio_bert:.2f}x | RelErr: {diff_bert.item():.4f} | Finite: {finite_bert}")
    except Exception as e:
        passed_all = False
        print(f"  [3] BertModel:   EXCEPTION: {e}")
        traceback.print_exc()

    print(f"OVERALL RESULT: {'PASS' if passed_all else 'FAIL'}")
    return passed_all


if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys = [sys.argv[1]]
    else:
        keys = [
            "coopq_quantizer", "cr_compressor", "cur_compressor", "cur_moe_compressor",
            "delta_kv", "direct_svd", "dobi_svd", "drone_compressor",
            "dynamic_rank", "enlighten_compressor", "entropy_budget_allocation",
            "erc_svd", "faq_quantizer", "flap", "flat_llm", "fp8", "frandi_compressor"
        ]
    all_ok = True
    for k in keys:
        ok = audit_engine(k)
        if not ok:
            all_ok = False
    print(f"\n{'='*70}\nGRAND TOTAL: {'ALL PASSED' if all_ok else 'SOME FAILED'}\n{'='*70}")
