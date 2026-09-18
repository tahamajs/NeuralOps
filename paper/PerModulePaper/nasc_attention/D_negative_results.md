<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Encoder Default is Dense

By default, encoders use dense attention. NASC's cascade isn't exercised 
on DistilBERT/RoBERTa unless explicitly enabled.

| Config | DistilBERT |
|---|---|
| Cascade enabled | 0.85 |
| Cascade disabled | 0.85 |

**Observation.** 0.85

## D.2 Gate Collapse

In some configurations, the gate concentrates on one branch:

| Model | Dominant branch |
|---|---|
| GPT-2 | $g_0$ (local) |

**Result.** When the gate collapses, the other branches are dead weight.

## D.3 RFF Causal Memory

At $L = 512$, the `KV_cumsum` tensor is ~200 MB. At $L = 1024$: ~800 MB. 
At $L = 2048$: ~3.2 GB. **OOM** on 8 GB VRAM.

## D.4 Sparse Block Overhead

The Python loop in `_sparse_block_attention` adds 4 kernel launches 
(for $L = 128$, $B_s = 32$). Each launch has ~5-10 μs overhead.

## D.5 `softpick` Fallback Coarseness

The fallback triggers if **any** row has zero denominator, but the 
`torch.where` is element-wise. This means:
- If one row in a batch has zero denom, softmax fallback applies only to 
  that row (via torch.where).
- But the `if zero_mask.any()` is computed over the whole batch.

## D.6 `inject_nasc` Raises

Unlike other modules, `inject_nasc` raises on failure. This breaks the 
"safe injection" contract.

**Suggested fix:**

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    return None  # instead of raise
```

## D.7 Failed Alternatives

| Attempt | Result |
|---|---|
| Per-token gate | Slower, no benefit observed |
| Remove local branch | Accuracy drop |
| Remove sparse branch | Small drop |
| Remove global branch | Accuracy drop |
| Vectorize sparse loop | Not implemented |

## D.8 Lessons Learned

1. **Three branches provide redundancy.** Removing any hurts.
2. **Batch-level gate is a simplification.** Per-token might help but 
   wasn't tested.
3. **RFF memory is a constraint.** Fused kernel needed for long sequences.
4. **Robustness regression.** `inject_nasc` should return None.

## D.9 Recommendations

1. **Test encoder cascade** with `enable_cascade_on_encoders=True`.
2. **Vectorize sparse block loop.**
3. **Fused Triton kernel** for RFF.
4. **Fix `inject_nasc`** to return None.
5. **Per-token gating** as a future extension.

---