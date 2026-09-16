# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 FA3 on Ampere

**Goal.** Use FlashAttention-3 on RTX 4070 (Ampere).

**Result.** The FA3 kernel requires Hopper architecture. The call falls 
back to SDPA silently.

**Conclusion.** FA3 is Hopper-only.

### D.1.2 FP8 Without Careful Scaling

**Goal.** Enable FP8 on all inputs.

**Result.** Significant accuracy degradation on some inputs.

**Conclusion.** FP8 requires scaling; we do not implement it.

### D.1.3 Cross-Attention with Causal Flag

**Goal.** Set causal flag for cross-attention.

**Result.** Incorrect masking; degraded output.

**Conclusion.** Cross-attention is not causal.

### D.1.4 Auto-Enabled FP8

**Goal.** Auto-enable FP8 on Hopper.

**Result.** Marginal accuracy loss; not always faster.

**Conclusion.** Keep `use_fp8=False` by default.

### D.1.5 Custom SDPA Mask

**Goal.** Use a custom attention mask with SDPA.

**Result.** Slower than FA3's native mask.

**Conclusion.** Use `is_causal` when possible.

## D.2 Abandoned Variants

### D.2.1 FA3 with Native GQA

We tested passing $H \neq H_{kv}$ directly to FA3 instead of expanding via 
`repeat_kv`. Result: faster but harder to integrate with other paths.

### D.2.2 Mixed Precision

We tested automatic mixed precision (AMP) with FA3. Result: no additional 
benefit over explicit FP16.

### D.2.3 FA2 on Ampere

We could use FlashAttention-2 on Ampere. Result: modest speedup over SDPA 
(1.5×). We keep SDPA for simplicity.

## D.3 Failed Hyperparameters

### LoRA Rank

| Rank | DistilBERT |
|---|---|
| 4 | {{D_R4}} |
| 8 | {{D_R8}} |
| 16 | {{D_R16}} |
| 32 | {{D_R32}} |

### Sequence Length

| L | FA3 speedup |
|---|---|
| 64 | 0.9× (slower) |
| 128 | 1× |
| 512 | 2× |
| 2048 | 4× |

**FA3 is faster only for L ≥ 512.**

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **FA3 is Hopper-only.**
2. **FP8 needs careful scaling.**
3. **Cross-attention is not causal.**
4. **SDPA is a fine fallback.**
5. **No parameter reduction.**

---