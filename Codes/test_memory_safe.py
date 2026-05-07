#!/usr/bin/env python
"""Memory‑safe audit – single model load, attention swap (first layer)."""
import gc, traceback, torch, inspect
from transformers import AutoConfig, AutoModelForSequenceClassification
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.config import CompressionConfig

# Build config and patch missing attributes that some attention modules expect
config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = 2   # so the classification head is created
config.attention_probs_dropout_prob = 0.1
config.num_heads = config.num_attention_heads   # some modules use config.num_heads

print("Loading model once ...", end=" ")
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', config=config
)
print("OK\n")

all_names = sorted(ATTENTION_REGISTRY.keys())
total = len(all_names)
comp_cfg = CompressionConfig()
layer = model.distilbert.transformer.layer[0]   # only first layer for speed
original_attn = layer.attention

passed, failed, skipped = 0, 0, 0
failures = []

for idx, attn_name in enumerate(all_names):
    if attn_name == 'full':
        skipped += 1
        continue

    try:
        inject_fn = ATTENTION_REGISTRY.get(attn_name)
        if not inject_fn:
            raise ValueError('not found')

        # Build call args dynamically from signature
        sig = inspect.signature(inject_fn)
        avail = {
            'layer': layer,
            'model_config': config,
            'comp_config': comp_cfg,
            'layer_idx': 0,
            'total_layers': 6,
            'old_attn': original_attn,
            'pretrained_attn': original_attn,
            'config': config,
            'rank': getattr(comp_cfg, 'compression_rank', 64),
            'num_heads': config.num_heads,
            'hidden_size': config.hidden_size,
            'bert_mode': False,
        }
        call_args = {k: avail[k] for k in sig.parameters if k in avail}
        inject_fn(**call_args)

        # Forward pass
        dummy = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            _ = model(dummy)

        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ✅ PASS")
        passed += 1

    except Exception as e:
        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ❌ FAIL  — {str(e)[:90]}")
        failures.append((attn_name, str(e)[:120]))
        failed += 1

    finally:
        # Always restore the original attention for the next test
        layer.attention = original_attn

print(f"\n{'='*50}")
print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}  (of {total})")
if failures:
    print("Failures:")
    for name, err in failures:
        print(f"  {name:30s} {err}")
