<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 The Long-Memory Branch is Fake

The `GatedDeltaNetLayer` is a linear projection. When removed:

| Config | DistilBERT |
|---|---|
| Full | {{D_ACC}}% |
| Without long-memory | {{D_NOLONG}}% |
| Δ | {{D_NOLONG_D}} |

**Conclusion.** The long-memory branch contributes little; its 590K 
parameters per layer are mostly wasted.

## D.2 Gate Mean-Pooling

We tested keeping per-head gating:

| Config | DistilBERT |
|---|---|
| Mean-pooled gate | {{D_ACC}}% |
| Per-head gate | {{D_PERHEAD}}% |

**Result.** {{PERHEAD_RESULT}}

## D.3 Preservation Gate Dominance

Varying α:

| α | DistilBERT |
|---|---|
| 0.50 | {{D_AG0}} |
| 0.73 | {{D_AG1}} |
| 0.88 | {{D_AG2}} |
| 0.99 | {{D_AG5}} |

**Observation.** Higher α preserves accuracy but reduces the block's 
contribution.

## D.4 Window Size

Smaller windows are faster but lose accuracy:

| Window | DistilBERT | Time |
|---|---|---|
| 16 | {{D_W16}} | {{D_W16_T}} |
| 32 | {{D_W32}} | {{D_W32_T}} |
| 64 | {{D_W64}} | {{D_W64_T}} |
| 128 | {{D_W128}} | {{D_W128_T}} |

## D.5 Failed Alternatives

| Attempt | Result |
|---|---|
| Use real GatedDeltaNet | OOM |
| Use Mamba-style SSM | Not implemented |
| Remove preservation gate | Accuracy drop |
| Increase gate init bias | No change |

## D.6 Lessons Learned

1. **Naming is not implementation.** `GatedDeltaNetLayer` is `nn.Linear`.
2. **Mean-pooling loses information.** Per-head gating would be better.
3. **Conservative init preserves accuracy.** α ≈ 0.88 works.
4. **Sliding-window is a good local approximation.**

## D.7 Recommendations

1. **Replace the long-memory branch** with a real recurrent mechanism.
2. **Keep per-head gating.**
3. **Test with sliding-window-only baseline** to isolate contributions.
4. **Log gate statistics** during training.

## D.8 What Would Make This Work?

1. Real GatedDeltaNet (or Mamba, RWKV) for the long-memory branch.
2. Per-head gating instead of mean-pooling.
3. Fused Triton kernel for the sliding window.
4. Careful initialization study.
```

---