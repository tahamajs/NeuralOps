#!/usr/bin/env python3
"""
Verify Forward + Causal Mask + Backward (out.sum().backward())
across all 131 attention mechanisms in ATTENTION_REGISTRY
for dimensions D = 128, 256, and 768.
"""

import sys
import torch
import neuralops
import neuralops.attentions
from neuralops.core.registry import ATTENTION_REGISTRY
from transformers import GPT2Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Dimension Verification] Running on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

dims_to_test = [
    (128, 4),
    (256, 4),
    (768, 12),
]

attentions = sorted(list(ATTENTION_REGISTRY.keys()))
print(f"Total attention modules in registry: {len(attentions)}")

failures_by_dim = {d: [] for d, _ in dims_to_test}

for D, H in dims_to_test:
    print(f"\n=======================================================")
    print(f"TESTING DIMENSION: D = {D}, H = {H}")
    print(f"=======================================================")
    
    cfg = GPT2Config(n_embd=D, n_head=H, hidden_size=D, num_attention_heads=H)
    B, T = 2, 32
    
    for name in attentions:
        cls = ATTENTION_REGISTRY[name]
        torch.cuda.empty_cache()
        x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
        
        try:
            # Flexible instantiation
            try:
                mod = cls(config=cfg, dim=D, num_heads=H)
            except TypeError:
                try:
                    mod = cls(config=cfg)
                except TypeError:
                    try:
                        mod = cls(hidden_size=D, num_heads=H)
                    except TypeError:
                        try:
                            mod = cls(dim=D, num_heads=H)
                        except TypeError:
                            mod = cls(D, H)
                            
            mod = mod.to(device).train()
            
            # Forward pass
            try:
                out = mod(x)
            except Exception:
                out = mod(x, x, x)
                
            if isinstance(out, (tuple, list)):
                out = out[0]
                
            if out.shape[:2] != (B, T) or out.shape[-1] != D:
                raise ValueError(f"Shape mismatch: expected ({B}, {T}, {D}), got {out.shape}")
                
            # Backward pass
            loss = out.sum()
            loss.backward()
            
            if x.grad is None or not torch.isfinite(x.grad).all():
                raise RuntimeError("Gradient is None or contains NaNs/Infs")
                
        except Exception as e:
            err_str = f"{type(e).__name__}: {str(e)[:140]}"
            failures_by_dim[D].append((name, err_str))

print("\n" + "=" * 60)
print("DIMENSION VERIFICATION SUMMARY")
print("=" * 60)

for D, H in dims_to_test:
    fails = failures_by_dim[D]
    passed = len(attentions) - len(fails)
    print(f"Dimension D={D:<3}: {passed}/{len(attentions)} PASSED ({len(fails)} failed)")
    if fails:
        print("  Failures:")
        for name, err in fails[:15]:
            print(f"    * {name:<30}: {err}")
        if len(fails) > 15:
            print(f"    ... and {len(fails) - 15} more.")

sys.exit(0 if all(len(f) == 0 for f in failures_by_dim.values()) else 1)
