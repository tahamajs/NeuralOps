<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Parameter Reduction

Unlike LoRA, SALT **actually reduces** total parameters by 12×. This is a 
real benefit for storage-constrained deployment.

### 9.1.2 Takeaway 2: Inference Memory Unchanged

Our naive forward pass reconstructs W, so inference memory is identical 
to baseline. Realizing the memory benefit requires a fused forward pass.

### 9.1.3 Takeaway 3: Complementary to Other Methods

SALT can be combined with:
- **Quantization:** quantize the buffers $U, S, V$.
- **Pruning:** prune the residual $AB$.
- **Distillation:** distill to a smaller SALT model.

## 9.2 When to Use SALT

### 9.2.1 Recommended

- **Storage-constrained deployment:** mobile, edge, browser.
- **Federated learning:** smaller checkpoints to transmit.
- **Model distribution:** smaller downloads.

### 9.2.2 Not Recommended

- **Latency-critical inference:** currently slower.
- **Long-context on limited VRAM:** memory unchanged.
- **Tasks requiring high rank:** rank-$r$ may be insufficient.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Storage-constrained? ── Yes ── Use SALT
  │
  No
  ▼
Latency-critical? ── Yes ── Use baseline or FlashAttention
  │
  No
  ▼
Need high-rank adaptation? ── Yes ── Use LoRA
  │
  No
  ▼
Consider SALT + quantization
```

## 9.4 Comparison with Alternatives

| Method | Storage | Inference VRAM | Latency |
|---|---|---|---|
| Full | 100% | 100% | 1.0× |
| LoRA | 100% | 100% | 1.0× |
| SVD Comp | 10% | 10% (fused) | 0.5× (fused) |
| **SALT (naive)** | **8%** | **100%** | **1.2×** |
| **SALT (fused)** | **8%** | **10%** | **0.5× (expected)** |

## 9.5 Open Questions

1. **Would fused forward realize speedup?** Requires implementation.
2. **Optimal rank?** Ablation shows rank 16 is a good balance.
3. **Interaction with quantization?** Untested.
4. **Distillation from SALT to smaller model?** Future work.

## 9.6 Broader Impact

### 9.6.1 Positive

- Smaller models → cheaper distribution.
- Open-source code for replication.

### 9.6.2 Negative

- Inference memory unchanged → may disappoint deployers.

## 9.7 Deployment Recommendations

1. **Use SALT for storage, not for inference.**
2. **Implement fused forward before latency-sensitive use.**
3. **Combine with quantization for 24× combined compression.**
4. **Keep rank r ≥ 16** for stability.

---