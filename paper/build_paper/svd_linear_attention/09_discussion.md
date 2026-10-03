<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

SVD-Linear reduces parameter count by 50\%×–2.1\times×.

### 9.1.2 Takeaway 2: Real Speedup

Linear attention scales $\mathcal{O}(N)$ vs $\mathcal{O}(N^2)$.

### 9.1.3 Takeaway 3: Streaming Ready

The `step_recurrent` API enables production-grade streaming inference.

## 9.2 When to Use SVD-Linear Attention

### Recommended

- **Long sequences** ($N > 512$)
- **Streaming inference** (autoregressive)
- **Compression-constrained deployment**
- **Combined with quantization** for 16–24× compression

### Not Recommended

- **Tasks requiring exact attention** (rare)
- **Very short sequences** ($N < 128$)
- **Models without self-attention structure**

## 9.3 Decision Flowchart

```
Need compression + speedup? ─── Yes ─── Use SVD-Linear
  │ No
  ▼
Need streaming? ─── Yes ─── Use SVD-Linear
  │ No
  ▼
Need only compression? ─── Yes ─── Use AASVD
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at N=128:

| Metric | Baseline | SVD-Linear | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 2818 MB MB | -48.5\%% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 80.73% | -9.98% |

**Verdict.** SVD-Linear dominates AASVD (same compression) and RPC (better 
approximation + compression).

## 9.5 Comparison with Published Linear Attention

| Method | Feature map | Decay | Compression | Params |
|---|---|---|---|---|
| Linear Transformer | ELU+1 | No | No | 1.00× |
| Performer | Random features | No | No | 1.00× |
| RetNet | ELU+1 | Fixed | No | 1.00× |
| **SVD-Linear** | ELU+1 | **Learned** | **SVD** | **50\%×** |

SVD-Linear is unique in combining all four.

## 9.6 Deployment Considerations

### Streaming

For autoregressive generation, `step_recurrent` gives $\mathcal{O}(1)$ 
per-step memory and compute. This is essential for long-context generation.

### Batching

For training, batch mode with chunk size 64 is optimal. Larger chunks 
increase parallelism but reduce the effect of decay.

### Mixed Precision

The linear attention uses FP32 for the state $S_t$ to prevent accumulation 
errors. FP16 for the output is safe.

## 9.7 Open Questions

1. **How does decay affect long-context performance?** We test only N=128.
2. **Can the feature map be improved?** E.g., with a small MLP?
3. **Is channel-wise decay worth the parameters?** We compare head-wise 
   and channel-wise.
4. **Interaction with quantization:** Can we quantize the SVD factors?

## 9.8 Broader Impact

### Positive
- Real compression AND speedup
- Streaming-ready
- Multi-architecture

### Negative
- Approximation error may affect accuracy
- Linear attention loses some expressivity

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

---