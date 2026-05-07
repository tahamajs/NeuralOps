#!/usr/bin/env python3
"""
Full compression engine audit – adaptive injection, single model load, full report.
Patches missing arguments for calibration‑based and advanced compressors.
"""
import json, traceback, torch, inspect, gc
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from framework.compression_engine.core.registry import COMPRESSOR_REGISTRY

print("=" * 60)
print("COMPRESSION ENGINE AUDIT")
print("=" * 60)

# ── Config ───────────────────────────────────────────────────
config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = 2

print("Loading model once ...", end=" ", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', config=config
).eval()
print("OK\n")

# ── Dummy calibration loader (one batch of random tokens) ────
dummy_input = torch.randint(0, 1000, (8, 64))
dummy_labels = torch.randint(0, 2, (8,))
dummy_dataset = TensorDataset(dummy_input, dummy_labels)
dummy_loader = DataLoader(dummy_dataset, batch_size=8)

all_names = sorted(COMPRESSOR_REGISTRY.keys())
total = len(all_names)
print(f"Registry contains {total} compressor types:\n  {', '.join(all_names)}\n")

# ── Arguments pool ───────────────────────────────────────────
BASE_AVAIL = {
    'model': model,
    'base_model': model,
    'teacher_model': model,
    'source_model': model,
    'original_model': model,
    'config': config,
    'model_config': config,
    'rank': 32,
    'compression_rank': 32,
    'energy': 0.90,
    'svd_energy': 0.90,
    'sparsity': 0.3,
    'keep_ratio': 0.7,
    'num_heads': config.num_attention_heads,
    'hidden_size': config.hidden_size,
    'd_model': config.hidden_size,
    'dim': config.hidden_size,
    'n_heads': config.num_attention_heads,
    'num_layers': 6,
    'tokenizer': None,
    'device': 'cpu',
    'output_dir': None,
    'save_path': None,
    'method': 'svd',
    'calibration_loader': dummy_loader,
    'calib_loader': dummy_loader,
    'eval_loader': dummy_loader,
    'train_loader': dummy_loader,
}

passed, failed, skipped = 0, 0, 0
failures = []

for idx, comp_name in enumerate(all_names):
    # Fresh model for each compressor
    try:
        test_model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', config=config
        ).eval()
    except Exception as e:
        print(f"[{idx+1:3d}/{total}] {comp_name:30s} ❌ MODEL_LOAD  — {str(e)[:80]}")
        failed += 1
        continue

    # Update model references in the pool
    BASE_AVAIL['model'] = test_model
    BASE_AVAIL['base_model'] = test_model
    BASE_AVAIL['teacher_model'] = test_model
    BASE_AVAIL['source_model'] = test_model
    BASE_AVAIL['original_model'] = test_model

    try:
        compressor_fn = COMPRESSOR_REGISTRY.get(comp_name)
        if compressor_fn is None:
            raise ValueError('not found in registry')

        sig = inspect.signature(compressor_fn)
        call_args = {k: BASE_AVAIL[k] for k in sig.parameters if k in BASE_AVAIL}

        # Apply compressor
        result = compressor_fn(**call_args)
        if isinstance(result, torch.nn.Module):
            test_model = result

        # Forward pass
        dummy = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            _ = test_model(dummy)

        print(f"[{idx+1:3d}/{total}] {comp_name:30s} ✅ PASS")
        passed += 1

    except Exception as e:
        print(f"[{idx+1:3d}/{total}] {comp_name:30s} ❌ FAIL  — {str(e)[:90]}")
        failures.append({
            'name': comp_name,
            'error': str(e),
            'traceback': traceback.format_exc(),
        })
        failed += 1

    finally:
        del test_model
        gc.collect()

# ── Report ───────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}  (of {total})")

if failures:
    with open('compressor_failures.json', 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"\nDetailed failures saved to compressor_failures.json")
    print("\nFailures summary:")
    for fail in failures:
        print(f"  {fail['name']:30s} {fail['error'][:100]}")
else:
    print("\n🎉 All compressors passed!")
