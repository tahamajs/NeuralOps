# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Adaptive Rank is Better Than Fixed

Energy thresholding selects different ranks for Q, K, V, O — allocating 
capacity based on the intrinsic spectrum of each projection.

### 9.1.2 Takeaway 2: All Four Projections Can Be Factorized

Unlike SOA Fused Attention, we compress $W_O$ too. The residual adapter 
absorbs the additional error.

### 9.1.3 Takeaway 3: T5 and Cross-Attention Work

The method supports T5 relative attention bias and encoder-decoder cross-
attention, making it applicable to a broader range of models.

## 9.2 When to Use Adaptive SVD Attention

### Recommended
- **Models with heterogeneous projection spectra** (where fixed rank 
  under-performs).
- **T5-family models** (relative bias support).
- **Encoder-decoder models** (cross-attention).
- **Zero-shot deployments** (no calibration).
- **Parameter-sensitive applications** (2-3× attention reduction).

### Not Recommended
- **Long-context inference** (attention still quadratic).
- **Models where attention is a small fraction** (FFN dominates).

## 9.3 Decision Flowchart

```
Target model uses T5 relative bias? ─── Yes ─── Use Adaptive SVD
  │ No
  ▼
Target uses cross-attention? ─── Yes ─── Use Adaptive SVD
  │ No
  ▼
Want all-four projection compression? ─── Yes ─── Use Adaptive SVD
  │ No
  ▼
Consider SOA Fused (simpler)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at τ = 0.92:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Attention params | 2.36M | {{D_ATTN}} | **{{D_RATIO}}×** |
| Total params | 66.95M | {{D_TOTAL}} | {{D_TOTAL_DELTA}}% |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Avg rank | 384 | {{AVG_RANK}} | — |

## 9.5 Composition with Other Methods

Adaptive SVD Attention composes with:

- **GQA:** further reduces KV projection size.
- **KV quantization:** INT8 on factorized weights.
- **Token compression** (TCA): reduces KV length.
- **FFN compression** (A3/AASVD): compresses the other half.

A combined pipeline could achieve **4-5× total compression**.

## 9.6 Broader Impact

### Positive
- Principled rank selection.
- Broad architecture support (T5, cross-attention).
- Zero-shot — no calibration data.
- Real parameter reduction.

### Negative
- Attention remains quadratic.
- Requires more code than fixed-rank methods.

## 9.7 Open Questions

1. **Per-layer τ scheduling:** Should τ vary per layer?
2. **Interaction with quantization:** INT8-A3 vs INT8-A3+Adaptive SVD?
3. **Optimal min_rank:** Is 16 the right floor?
4. **Cross-attention with asymmetric K/V:** Does it change the optimal rank?

---
