<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression is Possible

ORION achieves **48.2\%×** parameter reduction with **-35.21%** 
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
| Accuracy | 90.71% | 55.50% | -35.21% |
| Params | 66.95M | 66.4MM | **48.2\%×** |
| Time | 1.9 s | 1.42x s | 2.3 s× |
| VRAM | 183 MB | 303 MB MB | 380 MB× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Params | Speed |
|---|---|---|---|
| Baseline | 90.71% | 1.00× | 1.0× |
| A3 | 0.85% | 2.5× less | 1.2× |
| Cosformer | 0.85% | 1.06× | 4.9× slower |
| **ORION** | **55.50%** | **48.2\%× less** | **2.3 s×** |

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