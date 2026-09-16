<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression is Possible

ORION achieves **{{D_RED}}×** parameter reduction with **{{D_DELTA}}%** 
accuracy drop. This is a genuine, measurable improvement over baseline.

### 9.1.2 Takeaway 2: Head Bifurcation is Novel

The 2/3 kernel + 1/3 full split is not common in the literature. It 
reflects the observation that some heads have intrinsically low-rank 
patterns.

### 9.1.3 Takeaway 3: Two Bugs to Fix

The `copy_weights` raise and `rff_proj` persistence are fixable bugs. 
With them fixed, ORION would be production-ready.

## 9.2 When to Use ORION

### 9.2.1 Recommended

- **Parameter-constrained deployment** where model size matters.
- **When you need both compression and efficient attention.**
- **Research baseline for hybrid attention + SVD.**

### 9.2.2 Not Recommended

- **Latency-critical inference** — ORION adds kernel launches.
- **When `copy_weights` reliability is critical** — fix the bug first.
- **When you save/reload models** — fix the `rff_proj` bug first.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you need parameter reduction? ── No ── Use Cosformer or NHA
  │
  Yes
  ▼
Is latency critical? ── Yes ── Use A3 (no hybrid overhead)
  │
  No
  ▼
Fix ORION's bugs first
  │
  ▼
Deploy ORION
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | ORION | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RED}}×** |
| Time | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_R}}× |

**Verdict.** {{VERDICT}}

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Params | Speed |
|---|---|---|---|
| Baseline | 90.71% | 1.00× | 1.0× |
| A3 | {{A3_D}}% | 2.5× less | 1.2× |
| Cosformer | {{COS_D}}% | 1.06× | 4.9× slower |
| **ORION** | **{{D_ACC}}%** | **{{D_RED}}× less** | **{{D_TIME_R}}×** |

## 9.6 Open Questions

1. **Would per-head rank selection help?** Some heads need more rank.
2. **Would persistent RFF help?** Yes (bug fix).
3. **Would more kernel heads help or hurt?** Split ablation shows trade-off.
4. **Would softpick fallback help masked rows?** Probably.

## 9.7 Broader Impact

### 9.7.1 Positive

- Real parameter reduction.
- Novel head bifurcation idea.

### 9.7.2 Negative

- Two production bugs.
- Slower than baseline.

## 9.8 Deployment Recommendations

**Fix the two bugs before deploying.** With bugs fixed, ORION is a solid 
choice for parameter-constrained deployments.

---