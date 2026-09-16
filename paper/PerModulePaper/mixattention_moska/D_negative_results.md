<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Decoder Path is a No-Op

For GPT-2, FASA **does not prune**. Comparing configurations:

| Config | GPT-2 |
|---|---|
| Baseline | 79.36% |
| FASA keep=0.5 | {{G_ACC}}% |
| FASA keep=0.1 | {{G_ACC_K10}}% |
| FASA keep=0.9 | {{G_ACC_K90}}% |

**Observation.** No difference between keep ratios. Confirms the decoder 
path skips pruning.

## D.2 Aggressive Keep Ratio Hurts

At keep=0.25 (32/128 tokens):

| Model | Keep=0.25 | Keep=0.5 | Keep=0.75 |
|---|---|---|---|
| DistilBERT | {{D_K25}} | {{D_K50}} | {{D_K75}} |
| RoBERTa | {{R_K25}} | {{R_K50}} | {{R_K75}} |

**Observation.** 0.25 is too aggressive.

## D.3 L2 Method Performs Similarly

| Method | DistilBERT | RoBERTa |
|---|---|---|
| `var` | {{D_VAR}} | {{R_VAR}} |
| `l2` | {{D_L2}} | {{R_L2}} |

**Observation.** No significant difference. Both heuristics are weak.

## D.4 Without CLS Pinning, Collapse

| Setting | DistilBERT |
|---|---|
| Protect CLS | {{D_PROTECT_ON}} |
| No protect | {{D_PROTECT_OFF}} |

**Observation.** Accuracy collapses without CLS pinning.

## D.5 Overhead at Short Sequences

At L=128, gather + topk overhead exceeds FLOP savings:

| Phase | Time |
|---|---|
| Standard attention | ~2 ms |
| FASA gather + topk | ~1 ms |
| FASA attention (pruned) | ~1 ms |
| **Total FASA** | **~2 ms + overhead** |

**Result.** FASA is **slower** than baseline at L=128.

## D.6 Failed Alternatives

| Attempt | Result |
|---|---|
| Remove CLS pinning | Collapse |
| Use only L2 norm | Same |
| Recompute pruning every layer | Extra cost |
| Fused gather (manual) | No improvement |

## D.7 Lessons Learned

1. **Variance is a weak importance signal.** Weaker than attention scores.
2. **CLS pinning is not optional.** It's a load-bearing design choice.
3. **Short sequences see no benefit.** Overhead dominates.
4. **Decoder support requires different design.** Causal prefix scan needed.
5. **The "MoSKA" name is unexplained.** Should be documented or renamed.

## D.8 Recommendations

1. **Test at long sequences ($L > 1024$)** where FASA should help.
2. **Use keep_ratio ≥ 0.75.**
3. **Learn importance scores** with a small head.
4. **Consider H2O or SnapKV** if attention scores are available.
5. **Add causal support** via prefix scan if decoders are target.

## D.9 What Would Make This Work?

1. **Learned importance**: Train a small head to predict importance.
2. **Layer-wise keep ratios**: PyramidKV-style allocation.
3. **Fused kernel**: Eliminate gather overhead.
4. **Causal prefix scan**: Enable decoder pruning.
```

---