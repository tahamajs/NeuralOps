<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Preservation Gate Preserves Accuracy

The preservation gate at α ≈ 0.88 makes the block conservative. This is 
the primary reason GoldFinch maintains most of baseline accuracy.

### 9.1.2 Takeaway 2: The Long-Memory Branch Should Be Real

The current linear projection adds parameters without function. A true 
recurrent mechanism would provide value.

### 9.1.3 Takeaway 3: Hybrid with Conservative Bias Works

The combination of sliding-window + preservation gate is a valid 
**low-risk** attention replacement. If you want to minimize accuracy 
loss, this is the approach.

## 9.2 When to Use GoldFinch-AHN

### 9.2.1 Recommended

- **When accuracy preservation is critical.**
- **When you want sliding-window efficiency.**
- **As a conservative baseline for hybrid attention.**

### 9.2.2 Not Recommended

- **When long-range dependencies matter.** Use full attention or real 
  recurrent mechanisms.
- **When parameter budget is tight.** +5-6% is significant.
- **When latency is critical.** ~2× slowdown.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Need to preserve baseline accuracy? ── Yes ── Use GoldFinch
  │
  No
  ▼
Need long-range dependencies? ── Yes ── Use full attention
  │
  No
  ▼
Sliding window alone is enough? ── Yes ── Use Longformer
  │
  No
  ▼
Consider alternatives
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | GoldFinch | Change |
|---|---|---|---|
| Accuracy | 90.71% | 50.92% | -39.79% |
| Params | 66.95M | 66.4MM | +48.5\%% |
| Time | 1.9 s | 1.42x s | 2.3 s× |
| VRAM | 183 MB | 353 MB MB | 2.1\times× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Params |
|---|---|---|---|
| Baseline | 90.71% | 1.0× | 0 |
| Longformer | ~89% | 0.8× | 0 |
| Cosformer | 0.85% | 2.3 s× | +5.7% |
| **GoldFinch** | **50.92%** | **2.3 s×** | **+48.5\%%** |

## 9.6 Open Questions

1. **Would per-head gating help?** Untested.
2. **Would a real GatedDeltaNet help?** Likely yes.
3. **What's the optimal α init?** Depends on task.
4. **Would removing the long-memory branch reduce parameters?** Yes, by ~1.8%.

## 9.7 Broader Impact

### 9.7.1 Positive

- Conservative hybrid attention that preserves accuracy.
- Documentation of a fake "recurrent" branch.

### 9.7.2 Negative

- Parameter overhead without benefit.

## 9.8 Deployment Recommendations

**Deploy when accuracy preservation matters more than parameter count or 
latency.**

---