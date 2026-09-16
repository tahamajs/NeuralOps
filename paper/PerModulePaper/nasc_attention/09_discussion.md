<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Cascade Works for Decoders

NASC's cascade path is used for GPT-2 and achieves {{G_DELTA}}% delta.

### 9.1.2 Takeaway 2: Encoder Default is Dense

By default, DistilBERT and RoBERTa use standard attention, so NASC's 
cascade doesn't apply.

### 9.1.3 Takeaway 3: Overhead is Significant

The three-branch cascade adds ~{{G_TIME_R}}× latency compared to baseline.

## 9.2 When to Use NASC

### 9.2.1 Recommended

- **Decoder-only models** where cascade is triggered.
- **When you need RFF global attention** with prefix-sum causality.
- **Research baseline** for hybrid attention.

### 9.2.2 Not Recommended

- **Encoders** (dense path by default).
- **Long sequences** ($L > 512$ for RFF cumsum OOM).
- **Latency-critical** inference (~{{G_TIME_R}}× slower).
- **When you need per-token adaptive gating.**

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is the model a decoder? ── No ── NASC's cascade doesn't apply
  │
  Yes
  ▼
Is L <= 512? ── No ── OOM risk in RFF cumsum
  │
  Yes
  ▼
Is latency critical? ── Yes ── Use FlashAttention instead
  │
  No
  ▼
Consider NASC
```

## 9.4 Cost-Benefit Analysis

For GPT-2 on L=128:

| Metric | Baseline | NASC | Change |
|---|---|---|---|
| Accuracy | 79.36% | {{G_ACC}}% | {{G_DELTA}}% |
| Params | 124.44M | {{G_PARAMS}}M | +{{G_PARAM_PCT}}% |
| Time | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× |
| VRAM | 793 MB | {{G_VRAM}} MB | {{G_VRAM_RATIO}}× |

**Verdict.** {{VERDICT}}

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Notes |
|---|---|---|---|
| Full attention | 79.36% | 1.0× | Baseline |
| FlashAttention-2 | ~79% | 1.1× | Faster |
| Cosformer | {{COS_G}}% | 4.9× slower | Linear attention |
| **NASC** | **{{G_ACC}}%** | **{{G_TIME_R}}×** | 3-branch cascade |

## 9.6 Open Questions

1. **Would per-token gating help?** Likely yes.
2. **Would vectorizing sparse block help?** Yes.
3. **Would fused RFF kernel reduce memory?** Yes.
4. **Why does the encoder default to dense?** Design or performance?

## 9.7 Broader Impact

### 9.7.1 Positive

- Demonstrates 3-branch hybrid attention.
- Documents `softpick` fix.

### 9.7.2 Negative

- Raises on failure (robustness regression).
- Significant inference overhead.

## 9.8 Deployment Recommendations

**Deploy only for decoders with L ≤ 512 and non-critical latency.**

---