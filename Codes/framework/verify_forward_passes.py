#!/usr/bin/env python3
"""
Comprehensive forward-pass verification for top 10 novel attention modules.
Run: PYTHONPATH=../../Codes python3 verify_forward_passes.py
"""
import sys, os, time, torch
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_codes_dir = os.path.dirname(_cur_dir)
_project_dir = os.path.dirname(_codes_dir)
for p in [_cur_dir, _codes_dir, _project_dir, os.path.join(_cur_dir, "../..")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.eval_all_models_datasets import build_model_instance, apply_module_to_model

TOP_NOVEL_MODULES = [
    "aegis_v1", "aegis_v2", "aegis_v3", "aegis_hybrid", "aegis_spectron_fp8",
    "nystrom_attention", "linformer_attention", "performer_attention", "mamba", "gqa_attention"
]

passed, failed = [], []
for mn in TOP_NOVEL_MODULES:
    try:
        model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1)
        model, ratio = apply_module_to_model(model, mn, "attention", rank=8)
        batch = {
            "input_ids": torch.randint(100, 30000, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 16, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            out = model(**batch)
        ms = (time.time() - t0) * 1000
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ {mn}: PASS — {ms:.1f}ms, {params:,} params, {ratio:.2f}x compression")
        passed.append(mn)
    except Exception as e:
        print(f"❌ {mn}: FAIL — {e}")
        failed.append(mn)

print(f"\n{'='*60}")
print(f"Results: {len(passed)} PASS, {len(failed)} FAIL")
if failed:
    print(f"Failed modules: {failed}")
    sys.exit(1)
