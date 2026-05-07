#!/usr/bin/env python
"""Test every attention module with adaptive injection, dtype handling, and cleanup."""
import sys, time, gc, traceback, inspect, torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.config import CompressionConfig

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

def adapt_injection(inject_fn, layer, config, comp_cfg, layer_idx, total_layers, old_attn):
    """Call inject_fn with only the parameters it actually accepts."""
    sig = inspect.signature(inject_fn)
    params = sig.parameters
    # Available arguments that injection functions might need
    avail = {
        'layer': layer,
        'model_config': config,
        'comp_config': comp_cfg,
        'compression_config': comp_cfg,
        'layer_idx': layer_idx,
        'total_layers': total_layers,
        'old_attn': old_attn,
        'pretrained_attn': old_attn,
        'rank': getattr(comp_cfg, 'compression_rank', 64),
        'num_heads': config.num_attention_heads,
        'hidden_size': config.hidden_size,
        'bert_mode': False,
        'config': config,
        'comp_config_obj': comp_cfg,
        'attention_type': comp_cfg.attn_type,
    }
    # Build argument dict from what the function expects
    call_args = {}
    for name, param in params.items():
        if name in avail:
            call_args[name] = avail[name]
        elif param.default is not inspect.Parameter.empty:
            pass  # use default
        else:
            # Try to match by type (not perfect but helps)
            pass
    return inject_fn(**call_args)

print("=" * 70)
print("FULL ATTENTION AUDIT (adaptive injection)")
print("=" * 70)

all_names = sorted(ATTENTION_REGISTRY.keys())
total = len(all_names)
print(f"Registry: {total} types\n")

model_name = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(model_name)

passed, failed, skipped = 0, 0, 0
failures = []

for idx, attn_name in enumerate(all_names):
    pct = (idx + 1) / total * 100
    test_model = None

    if attn_name == "full":
        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ⏭ SKIP")
        skipped += 1
        continue

    try:
        test_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, low_cpu_mem_usage=True
        )
        # Try to convert to float64 for modules that need it (like aegis_v1)
        try:
            test_model = test_model.double()
        except Exception:
            pass  # not all models support direct .double()

        inject_fn = ATTENTION_REGISTRY.get(attn_name)
        if inject_fn is None:
            raise ValueError(f"not found in registry")

        comp_cfg = CompressionConfig(attn_type=attn_name)
        layers = (test_model.distilbert.transformer.layer
                  if hasattr(test_model, "distilbert")
                  else test_model.bert.encoder.layer)

        # Injection with adaptive signature
        for i, layer in enumerate(layers):
            adapt_injection(inject_fn, layer, config, comp_cfg, i, len(layers), layer.attention)

        # Forward pass – try multiple sequence lengths
        forward_ok = False
        last_forward_err = ""
        for seq_len in (64, 128, 256):
            try:
                dummy = torch.randint(0, 1000, (1, seq_len))
                with torch.no_grad():
                    _ = test_model(dummy)
                forward_ok = True
                break
            except Exception as fwd_err:
                last_forward_err = str(fwd_err)[:120]
                continue

        if forward_ok:
            print(f"[{idx+1:3d}/{total}] {attn_name:30s} ✅ PASS")
            passed += 1
        else:
            msg = f"Forward failed for all seq lengths. Last: {last_forward_err}"
            print(f"[{idx+1:3d}/{total}] {attn_name:30s} ❌ SHAPE  → {msg}")
            failures.append((attn_name, "SHAPE", msg))
            failed += 1

    except Exception as e:
        full_tb = traceback.format_exc()
        tb_lines = full_tb.strip().splitlines()
        last_line = tb_lines[-1] if tb_lines else str(e)
        context = "\n".join(tb_lines[-5:]) if len(tb_lines) >= 5 else full_tb
        print(f"[{idx+1:3d}/{total}] {attn_name:30s} ❌ FAIL")
        print(f"  ┌─ {last_line}")
        for line in tb_lines[-6:-1]:
            print(f"  │ {line.strip()}")
        print(f"  └─ Full traceback saved below\n")
        failures.append((attn_name, "FAIL", context))
        failed += 1

    finally:
        del test_model
        cleanup()

# ── Final Summary ──
print("\n" + "=" * 70)
print("DETAILED FAILURE REPORT")
print("=" * 70)
if failures:
    for name, category, detail in failures:
        print(f"\n{'─'*70}")
        print(f"  {name}  [{category}]")
        print(f"{'─'*70}")
        print(detail)
else:
    print("  No failures! 🎉")

print(f"\n{'='*70}")
print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped  (of {total})")
print("=" * 70)
