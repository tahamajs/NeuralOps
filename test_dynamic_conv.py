"""Test injectors directly — bypass apply_module_to_model."""

import sys
import copy
import torch
from pathlib import Path

ROOT = Path("D:/SVD_Project")
sys.path.insert(0, str(ROOT))

from transformers import AutoModel, AutoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ---- injectors ----
from neuralops.attentions.modern.dynamic_conv_attention import inject_dynamic_conv
from neuralops.attentions.modern.cross_attention import inject_cross_attention
from neuralops.attentions.modern.alibi import inject_alibi
from neuralops.attentions.modern.wgqa_attention import inject_wgqa

INJECTORS = [
    ("dynamic_conv",  inject_dynamic_conv),
    ("alibi",         inject_alibi),
    ("cross",         inject_cross_attention),
    ("wgqa",          inject_wgqa),
]

print("=" * 70)
print("  DIRECT INJECTOR TEST")
print("=" * 70)

for label, injector in INJECTORS:
    print(f"\n--- {label} ---")
    try:
        gpt2 = AutoModel.from_pretrained("gpt2").to(device).eval()
        cfg = gpt2.config

        # baseline
        ids = torch.randint(0, 50257, (1, 16), device=device)
        with torch.no_grad():
            base = gpt2(ids).last_hidden_state
        base_norm = base.abs().mean().item()

        # inject
        result = injector(model=gpt2, model_config=cfg)
        print(f"  injector returned: {type(result).__name__ if result else None}")

        # check layer0 attn class
        cls_name = gpt2.h[0].attn.__class__.__name__
        print(f"  layer0.attn class: {cls_name}")

        # forward
        try:
            with torch.no_grad():
                new = gpt2(ids).last_hidden_state
            diff = (base - new).abs().max().item()
            print(f"  base_norm={base_norm:.4f}, new_norm={new.abs().mean().item():.4f}, diff={diff:.4f}")
            if diff > 1e-3:
                print(f"  ✓ Injection works")
            else:
                print(f"  ✗ No change")
        except Exception as e:
            print(f"  ✗ forward failed: {type(e).__name__}: {e}")

    except Exception as e:
        import traceback
        print(f"  ✗ injector failed: {type(e).__name__}: {e}")
        traceback.print_exc()

print("\n" + "=" * 70)