<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Variance is a Weak Signal

Variance of K correlates weakly with attention received. FASA works better 
than random pruning but worse than attention-based methods (H2O).

### 9.1.2 Takeaway 2: Decoder Support is Missing

FASA's design explicitly excludes causal decoders to preserve causality. 
A different design would need to handle the causal case (e.g., with a 
prefix scan).

### 9.1.3 Takeaway 3: Speedup Requires Long Sequences

At L=128, gather + topk overhead exceeds FLOP savings. FASA would help at 
$L > 1024$.

## 9.2 When to Use FASA

### 9.2.1 Recommended

- **Bidirectional encoders** with long sequences ($L > 1024$).
- **No-attention-score settings** — FASA doesn't need attention weights.
- **Research baseline** for token pruning.

### 9.2.2 Not Recommended

- **Causal decoders** — no pruning, no benefit.
- **Short sequences** ($L < 256$) — overhead dominates.
- **Accuracy-critical tasks** — variance heuristic is imperfect.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is the model bidirectional? ── No ── FASA gives no benefit
  │
  Yes
  ▼
Is L > 1024? ── No ── Overhead dominates
  │
  Yes
  ▼
Is attention-score access available? ── Yes ── Use H2O instead
  │
  No
  ▼
Use FASA
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on L=128:

| Metric | Baseline | FASA | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}}M | +{{D_PARAM_PCT}}% |
| Time | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_RATIO}}× |

**Verdict.** {{VERDICT}}

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Requires Attention? |
|---|---|---|---|
| Full attention | 90.71% | 1.0× | — |
| H2O | ~{{H2O_ACC}}% | 0.6× | Yes |
| StreamingLLM | ~{{SLLM_ACC}}% | 0.6× | No |
| **FASA** | **{{D_ACC}}%** | **{{D_TIME_R}}×** | **No** |

## 9.6 Open Questions

1. **Would learned importance help?** Likely yes.
2. **Would layer-wise keep ratios help?** PyramidKV suggests yes.
3. **What about a causal prefix scan?** Would enable decoder support.
4. **Fused Triton kernel?** Would eliminate gather overhead.

## 9.7 Broader Impact

### 9.7.1 Positive

- Cheaper than H2O (no attention scores).
- Simple to implement.

### 9.7.2 Negative

- Accuracy drop on encoders.
- No benefit for decoders.
- Overhead at short sequences.

## 9.8 Deployment Recommendations

**Deploy only for long-sequence bidirectional encoders with a conservative 
keep ratio (≥ 0.75).**

---