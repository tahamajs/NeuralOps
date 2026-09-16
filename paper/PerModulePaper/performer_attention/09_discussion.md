<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: FAVOR+ is a Strong Linear Attention Baseline

Performer achieves competitive accuracy on all three models. The FAVOR+ 
feature map with orthogonal random features is well-founded.

### 9.1.2 Takeaway 2: Numerical Stability is Essential

The max-subtraction and clamp tricks are necessary for FP16. Without them, 
`exp(ω^T x - ||x||²/2)` overflows.

### 9.1.3 Takeaway 3: The Docstring Lies

`inject_performer` claims to return `None` on failure but raises. **Fix 
the code or fix the docstring.**

## 9.2 When to Use Performer

### 9.2.1 Recommended

- **Long sequences** where $\mathcal{O}(L)$ memory matters.
- **Research baseline** for kernel attention.
- **When FAVOR+ theory matters.**

### 9.2.2 Not Recommended

- **Latency-critical inference** — slower than FlashAttention-2 at short 
  sequences.
- **When you need exact attention** — FAVOR+ is approximate.
- **Cross-attention tasks** — not supported.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is L > 1024? ── No ── FlashAttention-2 is likely better
  │
  Yes
  ▼
Do you need exact attention? ── Yes ── Use FlashAttention
  │
  No
  ▼
Is cross-attention needed? ── Yes ── Not supported
  │
  No
  ▼
Consider Performer
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on L=128:

| Metric | Baseline | Performer | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}}M | +{{D_PARAM_PCT}}% |
| Time | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_R}}× |

**Verdict.** {{VERDICT}}

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Notes |
|---|---|---|---|
| Full attention | 90.71% | 1.0× | Baseline |
| FlashAttention-2 | ~90% | 1.1× | Exact, fast |
| Cosformer | {{COS_D}}% | 4.9× slower | cos/sin reweighting |
| **Performer** | **{{D_ACC}}%** | **{{D_TIME_R}}×** | FAVOR+ |

## 9.6 Open Questions

1. **Would larger $F$ help?** Diminishing returns observed.
2. **Is orthogonal init essential?** Theoretical justification, empirical 
   small effect.
3. **Would fused kernel help?** Likely yes.

## 9.7 Broader Impact

### 9.7.1 Positive

- Well-founded theory.
- Reproducible.

### 9.7.2 Negative

- The `inject_performer` raise bug could cause crashes.

## 9.8 Deployment Recommendations

**Fix the injection raise bug before deploying.** With the fix, Performer 
is a solid linear attention baseline.
```

---