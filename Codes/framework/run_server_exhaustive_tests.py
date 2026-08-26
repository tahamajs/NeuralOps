#!/usr/bin/env python3
"""
run_server_exhaustive_tests.py — Runs exhaustive CUDA FP16/AMP stress-testing across all benchmark tracks on the GPU server.
"""
import sys
import io
import os
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Set paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import (
    TRACK_MODULES,
    build_model_instance,
    apply_module_to_model,
    get_dataset_batch,
    _filter_batch,
    _extract_logits_and_loss,
    evaluate_single_combination_opt,
)

print("=" * 70)
print("[START] RUNNING EXHAUSTIVE GPU SERVER VERIFICATION SUITE")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"PyTorch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")
print(f"Tracks in Framework: {len(TRACK_MODULES)}")
print(f"Registered Compressors: {len(COMPRESSOR_REGISTRY)}")
print("-" * 70)

# -------------------------------------------------------------
# Part 1: Exhaustive Attention Track Validation on CUDA (FP16/AMP)
# -------------------------------------------------------------
print("\n[Part 1] Testing ALL Benchmark Track Attention Modules on CUDA (FP16 + AMP)...")
attn_passed = 0
attn_failed = []

for track, cat_dict in TRACK_MODULES.items():
    total_in_track = sum(len(v) for v in cat_dict.values())
    print(f"\n--- Testing {track} ({total_in_track} modules) ---")
    for cat_name, module_names in cat_dict.items():
        mtype = "attention" if "atten" in cat_name else "compressor"
        for name in module_names:
            try:
                model, cfg = build_model_instance("bert", num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=device)
                model, _ = apply_module_to_model(model, name, mtype, rank=16, device=device)
                model.train()
                
                input_ids = torch.randint(0, 1000, (2, 32), device=device)
                attention_mask = torch.ones((2, 32), device=device)
                labels = torch.tensor([0, 1], device=device)
                
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits, loss = _extract_logits_and_loss(out, num_labels=2, labels=labels)
                    if loss is None:
                        loss = logits.mean()
                
                loss.backward()
                attn_passed += 1
                print(f"  [PASS] {track} | {name} ({mtype})")
            except Exception as e:
                attn_failed.append((track, name, str(e)))
                print(f"  [FAIL] {track} | {name} ({mtype}): {e}")

# -------------------------------------------------------------
# Part 2: Exhaustive Compressor Module Validation on CUDA
# -------------------------------------------------------------
print(f"\n[Part 2] Testing ALL Compression Engines on CUDA...")
comp_passed = 0
comp_failed = []

for name in sorted(COMPRESSOR_REGISTRY.keys()):
    try:
        model, cfg = build_model_instance("bert", num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=device)
        model, _ = apply_module_to_model(model, name, "compressor", rank=16, device=device)
        model.train()
        
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        attention_mask = torch.ones((2, 32), device=device)
        labels = torch.tensor([0, 1], device=device)
        
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits, loss = _extract_logits_and_loss(out, num_labels=2, labels=labels)
            if loss is None:
                loss = logits.mean()
        
        loss.backward()
        comp_passed += 1
        print(f"  [COMP PASS] {name}")
    except Exception as e:
        comp_failed.append((name, str(e)))
        print(f"  [COMP FAIL] {name}: {e}")

# -------------------------------------------------------------
# Part 3: Multi-Architecture Compatibility on CUDA
# -------------------------------------------------------------
print(f"\n[Part 3] Multi-Architecture Compatibility on CUDA (BERT, RoBERTa, T5, DistilBERT, GPT-2)...")
arch_passed = 0
arch_failed = []

for family in ["bert", "roberta", "t5", "distilbert", "gpt2"]:
    try:
        model, cfg = build_model_instance(family, num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=device)
        batch = get_dataset_batch("sst2", family, batch_size=2, seq_len=32, device=device, allow_synthetic=True)
        b = _filter_batch(batch, family, model)
        with torch.no_grad():
            out = model(**b)
        logits, loss = _extract_logits_and_loss(out, num_labels=2, labels=batch.get("labels"))
        arch_passed += 1
        print(f"  [ARCH PASS] {family} -> logits shape: {logits.shape if logits is not None else None}")
    except Exception as e:
        arch_failed.append((family, str(e)))
        print(f"  [ARCH FAIL] {family}: {e}")

# -------------------------------------------------------------
# Part 4: End-to-End Pipeline Evaluation Run
# -------------------------------------------------------------
print(f"\n[Part 4] Testing End-to-End Benchmark Combination on GPU...")
try:
    res = evaluate_single_combination_opt(
        module_name="svd_linear_attention",
        module_type="attention",
        model_family="bert",
        dataset="sst2",
        seq_len=64,
        rank=16,
        batch_size=8,
        device=device,
    )
    print(f"  [E2E PASS] svd_linear_attention | bert | sst2 -> Status: {res.status} | Fwd: {res.forward_latency_ms:.2f}ms | Acc: {res.accuracy_pct:.2f}% | F1: {res.f1_score:.4f}")
    e2e_pass = True
except Exception as e:
    print(f"  [E2E FAIL] evaluate_single_combination_opt: {e}")
    e2e_pass = False

print("\n" + "=" * 70)
print("FINAL GPU SERVER VERIFICATION SUMMARY")
print("=" * 70)
total_track_mods = sum(sum(len(mods) for mods in cat.values()) for cat in TRACK_MODULES.values())
print(f"Track Modules Tested:       {total_track_mods} (PASSED: {attn_passed}, FAILED: {len(attn_failed)})")
print(f"Compressor Modules Tested:  {len(COMPRESSOR_REGISTRY)} (PASSED: {comp_passed}, FAILED: {len(comp_failed)})")
print(f"Model Architectures Tested: 5 (PASSED: {arch_passed}, FAILED: {len(arch_failed)})")
print(f"E2E Pipeline Integration:   {'PASSED' if e2e_pass else 'FAILED'}")
print("=" * 70)

if attn_failed or comp_failed or arch_failed or not e2e_pass:
    sys.exit(1)
else:
    print("[SUCCESS] 100% OF MODULES, COMPRESSORS, AND ARCHITECTURES PASSED CUDA GPU TESTING!")
