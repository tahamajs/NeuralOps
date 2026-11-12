#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Detailed individual diagnostic script for Group 3 compressors (keys 35-51).
"""
import sys
import os
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)
if os.path.abspath(".") not in sys.path:
    sys.path.insert(0, os.path.abspath("."))

import copy
import traceback
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from tests.probe_group3 import GROUP3_KEYS, TwoLayerMLP, apply_compressor, get_model_params, get_model_sparsity

def diagnose_compressor(name: str):
    print("=" * 80)
    print(f"DIAGNOSING: {name}")
    print("=" * 80)
    
    report = {
        "name": name,
        "mlp_compress": False,
        "mlp_forward": False,
        "mlp_backward": False,
        "mlp_ratio": 0.0,
        "mlp_rel_err": float("nan"),
        "bert_compress": False,
        "bert_forward": False,
        "bert_backward": False,
        "bert_ratio": 0.0,
        "bert_rel_err": float("nan"),
        "issues": []
    }
    
    # ----------------- TwoLayerMLP -----------------
    try:
        mlp = TwoLayerMLP(768)
        orig_mlp_params = get_model_params(mlp)
        c_mlp, comp_obj = apply_compressor(name, mlp, rank=128)
        report["mlp_compress"] = True
        
        # Check params and ratio
        comp_mlp_params = get_model_params(c_mlp)
        mlp_sparsity = get_model_sparsity(c_mlp)
        mlp_sparsity_ratio = 1.0 / (1.0 - mlp_sparsity + 1e-9) if mlp_sparsity > 0.0 else 1.0
        
        bit_reduction = 1.0
        if hasattr(comp_obj, "effective_bits"):
            bit_reduction = 32.0 / comp_obj.effective_bits
        elif hasattr(comp_obj, "bits"):
            bit_reduction = 32.0 / comp_obj.bits
        elif hasattr(c_mlp, "effective_bits"):
            bit_reduction = 32.0 / c_mlp.effective_bits
            
        param_ratio = orig_mlp_params / max(comp_mlp_params, 1)
        eff_ratio = max(param_ratio, mlp_sparsity_ratio, bit_reduction)
        report["mlp_ratio"] = eff_ratio
        
        # Forward pass
        x = torch.randn(2, 16, 768, requires_grad=True)
        c_mlp.eval()
        with torch.no_grad():
            out_eval = c_mlp(x)
        if torch.isfinite(out_eval).all():
            report["mlp_forward"] = True
        else:
            report["issues"].append("MLP forward output non-finite")
            
        # Error vs original
        with torch.no_grad():
            orig_out = mlp(x)
        rel_err = (orig_out.float() - out_eval.float()).norm() / (orig_out.float().norm() + 1e-9)
        report["mlp_rel_err"] = float(rel_err.item())
        
        # Backward pass
        c_mlp.train()
        x_train = torch.randn(2, 16, 768, requires_grad=True)
        out_train = c_mlp(x_train)
        loss = out_train.sum()
        loss.backward()
        
        grad_finite = True
        for pname, p in c_mlp.named_parameters():
            if p.requires_grad and p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    grad_finite = False
                    report["issues"].append(f"MLP non-finite grad in parameter: {pname}")
        if grad_finite:
            report["mlp_backward"] = True
            
    except Exception as e:
        report["issues"].append(f"MLP Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

    # ----------------- BERT -----------------
    try:
        bert_cfg = BertConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=1,
            intermediate_size=3072,
            vocab_size=1000,
        )
        bert = BertModel(bert_cfg)
        orig_bert_params = get_model_params(bert)
        c_bert, comp_obj = apply_compressor(name, bert, rank=128)
        report["bert_compress"] = True
        
        comp_bert_params = get_model_params(c_bert)
        bert_sparsity = get_model_sparsity(c_bert)
        bert_sparsity_ratio = 1.0 / (1.0 - bert_sparsity + 1e-9) if bert_sparsity > 0.0 else 1.0
        
        bit_reduction = 1.0
        if hasattr(comp_obj, "effective_bits"):
            bit_reduction = 32.0 / comp_obj.effective_bits
        elif hasattr(comp_obj, "bits"):
            bit_reduction = 32.0 / comp_obj.bits
        elif hasattr(c_bert, "effective_bits"):
            bit_reduction = 32.0 / c_bert.effective_bits
            
        param_ratio = orig_bert_params / max(comp_bert_params, 1)
        eff_ratio = max(param_ratio, bert_sparsity_ratio, bit_reduction)
        report["bert_ratio"] = eff_ratio
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 16))
        c_bert.eval()
        with torch.no_grad():
            cand_out = c_bert(input_ids)
            if hasattr(cand_out, "last_hidden_state"):
                cand_out = cand_out.last_hidden_state
            elif isinstance(cand_out, tuple):
                cand_out = cand_out[0]
                
        if torch.isfinite(cand_out).all():
            report["bert_forward"] = True
        else:
            report["issues"].append("BERT forward output non-finite")
            
        with torch.no_grad():
            orig_out = bert(input_ids).last_hidden_state
        rel_err = (orig_out.float() - cand_out.float()).norm() / (orig_out.float().norm() + 1e-9)
        report["bert_rel_err"] = float(rel_err.item())
        
        # Check backward if trainable params exist
        c_bert.train()
        trainable = any(p.requires_grad for p in c_bert.parameters())
        if trainable:
            cand_train = c_bert(input_ids)
            if hasattr(cand_train, "last_hidden_state"):
                cand_train = cand_train.last_hidden_state
            elif isinstance(cand_train, tuple):
                cand_train = cand_train[0]
            loss = cand_train.sum()
            loss.backward()
            grad_finite = True
            for pname, p in c_bert.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grad_finite = False
                        report["issues"].append(f"BERT non-finite grad in parameter: {pname}")
            if grad_finite:
                report["bert_backward"] = True
        else:
            report["bert_backward"] = True # No trainable parameters (e.g. quantizer codebook)
            
    except Exception as e:
        report["issues"].append(f"BERT Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

    print(f"Report for {name}:")
    print(f"  MLP : compress={report['mlp_compress']}, forward={report['mlp_forward']}, backward={report['mlp_backward']}, ratio={report['mlp_ratio']:.2f}x, rel_err={report['mlp_rel_err']:.4f}")
    print(f"  BERT: compress={report['bert_compress']}, forward={report['bert_forward']}, backward={report['bert_backward']}, ratio={report['bert_ratio']:.2f}x, rel_err={report['bert_rel_err']:.4f}")
    if report["issues"]:
        print(f"  ISSUES: {report['issues']}")
    else:
        print("  NO ISSUES FOUND.")
    return report

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        diagnose_compressor(target)
    else:
        for k in GROUP3_KEYS:
            diagnose_compressor(k)
