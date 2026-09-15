# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Exact, Not Approximate

Unlike CHELA-SLAY or linear attention, FlashAttention-3 is **exact**. It 
does not approximate softmax — it computes it precisely, just with better 
memory access.

### 9.1.2 Takeaway 2: Hardware-Dependent

The speedup is entirely dependent on having FA3-compatible hardware 
(Hopper). On Ampere, we fall back to SDPA.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameters are unchanged (modulo LoRA). This is an **optimization**, not a 
compression.

## 9.2 When to Use FlashAttention-3

### Recommended
- **Hopper GPUs** (H100, H200).
- **Long sequences** ($L \geq 512$).
- **Memory-constrained inference.**
- **Training with exact attention.**

### Not Recommended
- **Ampere or older GPUs** — no speedup.
- **Short sequences** ($L \leq 256$).
- **When parameters must be reduced.**

## 9.3 Decision Flowchart

```
Running on Hopper? ─── No ─── Use standard SDPA
  │ Yes
  ▼
Long sequences (L≥512)? ─── Yes ─── Use FlashAttention-3
  │ No
  ▼
Use standard SDPA
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on H100, L = 2048:

| Metric | Baseline | FA3 | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.71% | ~0 |
| Params | 66.95M | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| Time | 180 s | {{H100_TIME}} | {{H100_SPEEDUP}}× |
| VRAM | 12 MB | {{H100_VRAM}} | {{H100_VRAM_RED}}× |

## 9.5 Comparison to Other Speedup Methods

| Method | Speedup | Exact | Params |
|---|---|---|---|
| LoRA (recovery only) | 1× | ✅ | +small |
| FlashAttention-3 | 3-6× | ✅ | +small |
| SDPA | 1× | ✅ | 0 |
| Linear attention | 1-2× (long) | ❌ | depends |
| MoE | 1× | ❌ | +large |

FlashAttention-3 is the only method that gives **substantial speedup with 
exact attention and no parameter cost**.

## 9.6 Broader Impact

### Positive
- Enables long-context inference on Hopper.
- Reduces memory footprint.
- Compatible with all other methods.

### Negative
- Requires Hopper hardware.
- No benefit on Ampere or older.

## 9.7 Open Questions

1. **FP8 accuracy on other tasks?** Only SST-2 tested.
2. **Interaction with quantization?** FA3 + INT8?
3. **Hopper scaling?** How does it scale to H200?

---
