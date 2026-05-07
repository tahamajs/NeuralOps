#!/usr/bin/env python3
"""
Final attention audit – adaptive injection, single model load, full report.
Handles float64 fallback correctly and patches all missing config attributes.
"""
import json, traceback, torch, inspect, gc, sys
from transformers import AutoConfig, AutoModelForSequenceClassification
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.config import CompressionConfig

# ── 1. Config with every attribute attention modules might need ──
config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = 2
config.attention_probs_dropout_prob = 0.1
config.num_heads = config.num_attention_heads
config.head_dim = config.hidden_size // config.num_attention_heads
config.hidden_act = getattr(config, 'hidden_act', 'gelu')
config.max_position_embeddings = 512
config.window_size = 64        # for Aegis v6
config.local_window = 64       # for Aegis v2

# ── 2. Load model once in float32 ─────────────────────────────
print("Loading model once ...", end=" ", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', config=config
).eval()
print("OK\n")

all_names = sorted(ATTENTION_REGISTRY.keys())
total = len(all_names)
comp_cfg = CompressionConfig()
layer = model.distilbert.transformer.layer[0]
original_attn_float32 = layer.attention   # keep reference for restoration

passed, failed, skipped = 0, 0, 0
failures = []

# ── 3. Arguments pool ─────────────────────────────────────────
BASE_AVAIL = {
    'layer': layer,
    'model_config': config,
    'comp_config': comp_cfg,
    'compression_config': comp_cfg,
    'layer_idx': 0,
    'total_layers': 6,
    'old_attn': original_attn_float32,
    'pretrained_attn': original_attn_float32,
    'config': config,
    'rank': 64,
    'num_heads': config.num_heads,
    'hidden_size': config.hidden_size,
    'd_model': config.hidden_size,
    'dim': config.hidden_size,
    'n_heads': config.num_attention_heads,
    'num_layers': 6,
    'n_layers': 6,
    'bert_mode': False,
    'num_labels': config.num_labels,
    'spectra_cfg': None,
    'ranks': {'q': 64, 'k': 64, 'v': 64, 'o': 64},
    'block_size': 64,
    'window_size': 256,
    'top_k': 8,
    'num_kv_heads': 4,
    'num_clusters': 32,
    'num_landmarks': 32,
    'feature_dim': 128,
    'use_gate': True,
    'gate_bias': 2.0,
    'compression_rank': 64,
    'svd_rank': 64,
}

for idx, attn_name in enumerate(all_names):
    if attn_name == 'full':
        skipped += 1
        continue

    # ── Always start with a clean float32 attention ──────────
    layer.attention = original_attn_float32
    # Ensure model stays in float32
    if model.distilbert.transformer.layer[0].attention.q_lin.weight.dtype != torch.float32:
        model = model.float()
        layer = model.distilbert.transformer.layer[0]
        original_attn_float32 = layer.attention
        BASE_AVAIL['old_attn'] = original_attn_float32
        BASE_AVAIL['pretrained_attn'] = original_attn_float32

    try:
        inject_fn = ATTENTION_REGISTRY.get(attn_name)
        if not inject_fn:
            raise ValueError('not found')

        sig = inspect.signature(inject_fn)
        call_args = {k: BASE_AVAIL[k] for k in sig.parameters if k in BASE_AVAIL}

        # ── First attempt in float32 ─────────────────────────
        inject_fn(**call_args)
        dummy = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            _ = model(dummy)

        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ✅ PASS")
        passed += 1

    except Exception as e:
        err_str = str(e)

        # ── Float64 fallback (only for SVD‑related dtype errors) ──
        if 'float64' in err_str or 'dtype' in err_str.lower():
            try:
                # Convert to float64
                model = model.double()
                layer = model.distilbert.transformer.layer[0]
                # Update references
                original_attn_f64 = layer.attention
                BASE_AVAIL['old_attn'] = original_attn_f64
                BASE_AVAIL['pretrained_attn'] = original_attn_f64

                # Recompute call_args because some params might be tensor dtypes
                call_args = {k: BASE_AVAIL[k] for k in sig.parameters if k in BASE_AVAIL}

                inject_fn(**call_args)
                dummy = torch.randint(0, 1000, (1, 64))
                with torch.no_grad():
                    _ = model(dummy)

                print(f"[{idx+1:3d}/{total}] {attn_name:30s} ✅ PASS (float64)")
                passed += 1

                # ── Restore float32 model ──────────────────────
                model = model.float()
                layer = model.distilbert.transformer.layer[0]
                original_attn_float32 = layer.attention
                BASE_AVAIL['old_attn'] = original_attn_float32
                BASE_AVAIL['pretrained_attn'] = original_attn_float32
                continue

            except Exception as e2:
                err_str = str(e2)

        # ── Permanent failure ─────────────────────────────────
        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ❌ FAIL  — {err_str[:90]}")
        failures.append({
            'name': attn_name,
            'error': err_str,
            'traceback': traceback.format_exc(),
        })
        failed += 1

# ── 4. Report ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}  (of {total})")

if failures:
    with open('attention_failures.json', 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"\nDetailed failures saved to attention_failures.json")
    print("\nFailures summary:")
    for fail in failures:
        print(f"  {fail['name']:30s} {fail['error'][:100]}")
else:
    print("\n🎉 All attention mechanisms passed!")
