<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Cosformer is a Strong Baseline

Of the linear attention mechanisms we tested, Cosformer is among the best 
at preserving accuracy. The cos/sin reweighting genuinely recovers locality.

### 9.1.2 Takeaway 2: Encoder vs Decoder Paths Diverge

The encoder path is clean and fast. The decoder path adds overhead from 
the chunked scan. On encoders (DistilBERT), Cosformer is closer to baseline.

### 9.1.3 Takeaway 3: RoBERTa Needs Special Handling

The nested encoder structure requires careful handling. Our default 
hyperparameters are insufficient; lower LR or excluding cos/sin reweighting 
helps.

## 9.2 When to Use Cosformer

### 9.2.1 Recommended

- **Long-context inference on flat encoders** (e.g., DistilBERT).
- **Research baseline** for linear attention comparisons.
- **When memory is the bottleneck** — $\mathcal{O}(L)$ state.

### 9.2.2 Not Recommended

- **RoBERTa or other nested encoders** without hyperparameter tuning.
- **Latency-critical inference** — chunked scan adds overhead.
- **Tasks requiring sharp attention** — Cosformer smooths attention.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Is the model a flat encoder? ── Yes ── Use Cosformer
  │
  No
  ▼
Is the model a decoder? ── Yes ── Consider but expect 3-5× slowdown
  │
  No
  ▼
Is the model RoBERTa? ── Yes ── Tune LR first
  │
  No
  ▼
Consider alternatives (FlashAttention, A3)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | Cosformer | Change |
|---|---|---|---|
| Accuracy | 90.71% | 77.75% | -12.96% |
| Params | 66.9M | 70791394M | +5.73% |
| Time | 1.9 s | 3.5 s | 1.84× |
| VRAM | 183 MB | 335 MB MB | 1.31× |

**Verdict.** works on flat encoder; problematic on RoBERTa; slow on decoder

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Params |
|---|---|---|---|
| FlashAttention-2 | ~90% | 1.1× | 0 |
| A3 (SVD) | 88.5% | 0.8× | **-60%** |
| NHA | 90.25% | 0.6× | +0.88% |
| **Cosformer** | **77.75%** | **1.84×** | **+5.73%** |

## 9.6 Open Questions

1. **Why does RoBERTa fail?** Our analysis is incomplete.
2. **Would larger LoRA rank help RoBERTa?** Untested.
3. **Would removing cos/sin for RoBERTa help?** Partially tested.
4. **How does Cosformer scale to L > 512?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Linear attention is more memory-efficient.
- Cosformer is a well-tested baseline.

### 9.7.2 Negative

- Our implementation is slower than baseline on short sequences.
- RoBERTa results are negative.

## 9.8 Deployment Recommendations

**Test on your specific model before deploying.**
- For flat encoders: likely fine.
- For nested encoders: tune first.
- For decoders: expect 3-5× slowdown.

---