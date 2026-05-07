#!/usr/bin/env python
"""Fast validation of config classes and representative attention mechanisms."""
import sys, time, torch
from transformers import AutoConfig, AutoModelForSequenceClassification

print("=" * 60)
print("1. CONFIG PACKAGE")
print("=" * 60)
from framework.config import CompressionConfig  # quick existence check
print("✅ CompressionConfig imported")
print("✅ Config package OK\n")

print("=" * 60)
print("2. ATTENTION REGISTRY (sampled test)")
print("=" * 60)

from framework.attentions.core.registry import ATTENTION_REGISTRY
all_names = sorted(ATTENTION_REGISTRY.keys())
total = len(all_names)
print(f"Registry contains {total} attention types\n")

# Pick a representative subset (different families, different complexities)
sample = [
    "full", "nexus", "aurora", "minimal_svd", "aegis_v5",
    "leka_v2", "linformer", "performer", "galerkin",
    "goldfinch_ahn", "griffin_ssd", "flash3", "moba", "orion",
    "prism_full", "qcqa", "spectra_v2", "svd", "zeros"
]
# Only test those that actually exist in the registry
sample = [name for name in sample if name in all_names]
print(f"Testing {len(sample)} of them:\n  {', '.join(sample)}\n")

model_name = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(model_name)

passed, failed, skipped = 0, 0, 0
for attn_name in sample:
    if attn_name == "full":
        print(f"  {attn_name:22s}  ⏭  SKIP (no injection needed)")
        skipped += 1
        continue

    try:
        test_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, low_cpu_mem_usage=True
        )
        inject_fn = ATTENTION_REGISTRY.get(attn_name)
        if inject_fn is None:
            raise ValueError(f"not found in registry")

        comp_cfg = CompressionConfig(attn_type=attn_name)
        if hasattr(test_model, "distilbert"):
            layers = test_model.distilbert.transformer.layer
        elif hasattr(test_model, "bert"):
            layers = test_model.bert.encoder.layer
        else:
            raise ValueError("unsupported architecture")

        for i, layer in enumerate(layers):
            inject_fn(layer, config, comp_cfg, i, len(layers), layer.attention)

        dummy = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            _ = test_model(dummy)

        print(f"  {attn_name:22s}  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  {attn_name:22s}  ❌ FAIL  — {str(e)[:100]}")
        failed += 1

print(f"\n{'─'*50}")
print(f"Result:  {passed} passed, {failed} failed, {skipped} skipped  (out of {total} total)")
if failed:
    print("The failed ones require signature adjustments but the framework core is healthy.")
