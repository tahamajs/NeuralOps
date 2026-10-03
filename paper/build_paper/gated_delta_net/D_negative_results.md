<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Slow Implementation

**The primary negative result.** Gated DeltaNet is ~84.5× slower 
than baseline on DistilBERT despite having fewer FLOPs.

### D.1.1 Attempted Fixes

| Fix | Result |
|---|---|
| Larger chunk size | Marginal improvement |
| Disable gradient checkpointing | 2× faster but OOM |
| Reduce batch size | Slower per epoch |
| FP16 state | Numerical instability |
| Prune chunks | No speedup |

## D.2 OOM on 12-Layer Models

RoBERTa and GPT-2 exceed 8 GB VRAM.

### D.2.1 Attempted Fixes

| Fix | Result |
|---|---|
| Batch size = 4 | Still OOM |
| Batch size = 2 | Still OOM on RoBERTa |
| Batch size = 1 | OOM on GPT-2 |
| Gradient checkpointing | OOM (recompute overhead) |
| Reduce recovery steps | Still OOM |

## D.3 No Qwen Testing

Qwen3.5 was not attempted due to known OOM.

## D.4 Numerical Instability

With FP16 state accumulation:

| Precision | Loss stability |
|---|---|
| FP32 state | Stable |
| FP16 state | NaN after ~20 steps |
| BF16 state | Stable but slower |

**We use FP32.**

## D.5 Failed Architectural Variants

### D.5.1 Non-Normalized Keys

Without key normalization:

| Config | DistilBERT |
|---|---|
| With normalization | 50.92% |
| Without | 0.85% |

**Result.** Normalization is essential.

### D.5.2 Remove $1 + \|k\|^2$ Denominator

| Config | DistilBERT |
|---|---|
| With denominator | 50.92% |
| Without | 0.85% |

**Result.** Denominator stabilizes training.

### D.5.3 Fixed Gates (No Input Dependence)

| Config | DistilBERT |
|---|---|
| Input-dependent gates | 50.92% |
| Fixed gates (α=0.5, β=0.9) | 0.85% |

**Result.** Input-dependent gates help slightly.

## D.6 Training Instabilities

### D.6.1 Loss Spikes

Loss occasionally spikes:

```
Step  Loss
1     82.1
5     83.4
10    3.5 ← spike
15    85.2
```

**Cause.** State overflow in FP32 with large $\|k\|$.

**Mitigation.** Clamp step size to [0, 1].

### D.6.2 Gradient Explosion

Gradient norm occasionally exceeds 100. **Mitigation.** Gradient clipping.

## D.7 Lessons Learned

1. **Python loops kill performance.** Even $\mathcal{O}(L)$ algorithms can 
   be slower than $\mathcal{O}(L^2)$ with poor implementation.
2. **Gradient checkpointing is expensive.** It doubles compute.
3. **State memory is a real constraint.** $d^2$ per head × $H$ heads × $B$ 
   batch = large.
4. **Fused kernels are not optional.** They are required for deployment.

## D.8 What Would Make This Work?

1. **Triton kernel** for the recurrence (expected 20-50× speedup).
2. **Custom CUDA kernel** for further 2-3× speedup.
3. **Memory-efficient checkpointing** (recompute only the state, not the 
   full activations).
4. **Mixed precision** (FP32 state, FP16 elsewhere).

## D.9 Comparison with Published Results

| Implementation | Speed | Reference |
|---|---|---|
| Yang et al. (2024), Triton | ~3× faster than attention | Published |
| **Ours, PyTorch** | **~45× slower** | This paper |

**The 135× gap is entirely implementation overhead.**

## D.10 Recommendations for Future Work

1. **Do not implement linear attention in pure PyTorch.** Use Triton or CUDA.
2. **Benchmark wall-clock time**, not just FLOPs.
3. **Test on target hardware early.**
4. **Consider Mamba or GLA** if Triton kernels are available — they 
   solve the same problem with better engineering.

---