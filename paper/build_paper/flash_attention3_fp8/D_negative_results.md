<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 FP8 Path Never Taken

The most prominent "negative result" is that the FP8 path was **never 
exercised**. This is not a bug; it is a hardware constraint.

### D.1.1 Attempted Fixes

| Fix | Result |
|---|---|
| Install `flash-attn` | Import succeeded, but FP8 kernel not available |
| Install `flash-attn-3` | Not available for Ada GPUs |
| Force `use_fp8=True` | RuntimeError caught, fell back to SDPA |
| Cast Q/K/V manually to fp8 | SDPA does not accept fp8 inputs |

**Conclusion.** No workaround exists on RTX 4070 Laptop.

## D.2 Failed Alternative Implementations

### D.2.1 Manual FP8 Cast + SDPA

We tried casting Q/K/V to FP8 before SDPA:

```python
q8 = q.to(torch.float8_e4m3fn).to(q.dtype)
# Then call SDPA with the cast-back values
```

**Result.** This simulates FP8 storage but not computation. Speedup: **none**.

### D.2.2 FP8 Weight-Only

We tried casting weights to FP8 and back, then using FP16 compute:

**Result.** Accuracy dropped by 3–5% without any speedup.

## D.3 Abandoned Variants

### D.3.1 TransformerEngine Integration

We explored integrating NVIDIA TransformerEngine for FP8 casting:

**Result.** TransformerEngine requires Hopper/Ada with specific driver 
versions. Not available on our hardware.

### D.3.2 Manual FP8 Kernel

We considered writing a custom FP8 attention kernel:

**Result.** Out of scope for this work; requires CUDA expertise.

## D.4 Failed Hyperparameters

| LR | DistilBERT |
|---|---|
| 1e-5 | 0.85 |
| 5e-5 | 0.85 |
| 2e-4 | 0.85 (default) |
| 5e-4 | 0.85 (unstable) |
| 1e-3 | NaN |

## D.5 Cross-Attention Not Supported

We do not support `encoder_hidden_states`. Attempting to pass it raises 
`NotImplementedError`.

## D.6 Lessons Learned

1. **Hardware detection is essential.** Without it, code crashes.
2. **SDPA is a strong fallback.** Matching baseline performance.
3. **FP8 is not a software feature.** No workaround for missing hardware.
4. **Honest reporting builds trust.** We clearly state FP8 was not exercised.

## D.7 Recommendations

1. **Always log which path was taken.**
2. **Do not publish FP8 benchmarks from non-Hopper hardware.**
3. **Document driver/hardware requirements explicitly.**

---