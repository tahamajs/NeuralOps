<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: RALA is Not a Compression Method

Unlike A3, RALA does not reduce parameter count. It **increases** parameters 
via LoRA and augmentation matrices. The value of RALA is architectural 
experimentation, not deployment.

### 9.1.2 Takeaway 2: The Scan is the Bottleneck

The Python prefix scan dominates inference time. Any production deployment 
would require a fused kernel.

### 9.1.3 Takeaway 3: Augmentation is Modest

Rank-$r$ augmentation with a scalar gate provides a small correction to linear 
attention. Its impact on accuracy is measurable but modest.

## 9.2 When to Use RALA

### 9.2.1 Recommended

- **Research on linear attention variants**
- **Prototyping with small models**
- **Baseline for future rank-augmented methods**

### 9.2.2 Not Recommended

- **Production deployment** (slow)
- **Compression tasks** (does not compress)
- **Long-context inference** (scan dominates)

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you need compression? ─── Yes ─── Use A3 or SDRM
  │
  No
  ▼
Do you need fast inference? ─── Yes ─── Use FlashAttention or SDPA
  │
  No
  ▼
Do you want to study linear attention? ─── Yes ─── Use RALA
  │
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Alternative | Complexity | Params | Accuracy |
|---|---|---|---|
| Softmax attention | $\mathcal{O}(L^2)$ | 0% | 90.71% |
| LinAttn (elu+1) | $\mathcal{O}(L)$ | 0% | 86.4\%% |
| Performer | $\mathcal{O}(L)$ | 0% | 86.4\%% |
| **RALA** | **$\mathcal{O}(L)$** | **+0.99%** | **60.09%** |

## 9.5 Open Questions

1. **Optimal rank $r$?** Rank 8 was chosen; is higher better?
2. **Per-head $\alpha$?** Would $H$ scalars per layer improve accuracy?
3. **Fused scan?** How much would a Triton kernel speed things up?
4. **Unfrozen $U, V$?** Would training the augmentation help?

## 9.6 Broader Impact

### 9.6.1 Positive

- Contributes to linear attention literature.
- Provides honest negative results.

### 9.6.2 Negative

- Not deployable as-is.
- Requires additional work for production.

## 9.7 Deployment Recommendations

If you must deploy RALA:

1. **Fuse the prefix scan** using Triton.
2. **Unfreeze $U, V$** to learn the augmentation.
3. **Per-head gate** instead of scalar.
4. **Benchmark on target hardware.**

Without these, RALA is not competitive with SDPA or FlashAttention.

---