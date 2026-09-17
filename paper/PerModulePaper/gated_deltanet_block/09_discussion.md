<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Block-Level Replacement is Harder

Replacing the entire block introduces new parameters (LayerNorm) that are 
not pretrained. This is an additional source of error that does not exist 
in attention-level replacement.

### 9.1.2 Takeaway 2: Silent Failures are Real

The `try/except` pattern masks bugs. Always log when critical operations 
fail.

### 9.1.3 Takeaway 3: The Wrapper Adds No Value

Even when the underlying attention works, the block wrapper is not better 
than attention-level replacement. It adds:
- A fresh LayerNorm
- Residual structure complexity
- Nested encoder inconsistency

## 9.2 When to Use This Block

### 9.2.1 Recommended

**None.** We do not recommend this block for any tested use case.

### 9.2.2 Possibly Worth Exploring

- **When pretrained LayerNorm is available.** If you have access to the 
  pretrained model's LayerNorm parameters, you can copy them into the block.
- **When the model has no nested encoders.** If you only work with DistilBERT 
  or GPT-2 (flat models), the nested encoder issue disappears.
- **As a research baseline.** For comparing block-level vs attention-level 
  replacement.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you have access to pretrained LayerNorm? ── No ── Do not use this block
  │
  Yes
  ▼
Do you need block-level control? ── No ── Use attention-level replacement
  │
  Yes
  ▼
Is the model flat (no nested encoder)? ── No ── Test carefully
  │
  Yes
  ▼
Consider using this block
```

## 9.4 Comparison with Attention-Level Replacement

| Aspect | Attention-level | Block-level |
|---|---|---|
| Pretrained LN preserved | Yes | **No** |
| Residual structure | Unchanged | Replaced |
| Parameters added | LoRA only | LoRA + LN |
| Silent failure paths | Fewer | More |
| Nested encoder handling | Automatic | Manual |

## 9.5 Cost-Benefit Analysis

For DistilBERT with Path A (MHA):

| Metric | Baseline | Block MHA |
|---|---|---|
| Accuracy | 90.71% | 86.5% |
| Δ | — | 0.85 |
| Params | 66.9M | +66.4M |
| Time | 0.4 s | 2.3 s s |

**Verdict.** Not worth the additional complexity.

## 9.6 Open Questions

1. **Would copying pretrained LayerNorm help?** Likely yes, but requires 
   access to the original parameters.
2. **Would a residual-only block (no LN) work?** Untested.
3. **Would a pre-LN vs post-LN choice matter?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Documents a real failure mode.
- Warns against `try/except: pass` in critical code.
- Clarifies LayerNorm's role in weight transfer.

### 9.7.2 Negative

- Compute consumed for a non-working method.

## 9.8 Deployment Recommendations

**Do not deploy GatedDeltaNetBlock in production.**

## 9.9 Lessons Learned

1. **Always preserve pretrained LayerNorm.** If you must replace the block, 
   copy the LN parameters.
2. **Never use `try/except: pass` on critical operations.** Log failures.
3. **Attention-level replacement is usually sufficient.** Block-level adds 
   complexity for marginal benefit.
4. **Causal masking is not optional for decoders.** `nn.MultiheadAttention` 
   needs explicit masking.

---