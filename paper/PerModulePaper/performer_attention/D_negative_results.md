<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 The `inject_performer` Raise Bug

Docstring: `Safe injection returning None on failure`. Code: `raise RuntimeError`.

**Impact.** Crashes the pipeline. Should be fixed.

## D.2 The `output_scale` Parameter

New parameter not in baseline. `strict=True` loading fails.

## D.3 Cross-Attention Not Supported

`encoder_hidden_states` → input unchanged, no warning.

## D.4 Feature Map Small $F$ Degrades

| $F$ | DistilBERT |
|---|---|
| 16 | 0.865 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.865 |

**Observation.** $F < 64$ loses significant accuracy.

## D.5 Kernel Type Trade-offs

| Kernel | Accuracy | Speed |
|---|---|---|
| `favor_plus` | Best | Baseline |
| `elu` | Slightly worse | Same |
| `relu` | Worst | Same |

## D.6 Failed Alternatives

| Attempt | Result |
|---|---|
| Remove eps from feature map | NaN possible |
| Remove max-subtraction | Overflow |
| Remove orthogonal init | Slightly worse |

## D.7 Lessons Learned

1. **Numerical stability is not optional.** Without it, FP16 overflows.
2. **Docstrings can lie.** Trust the code, not the docstring.
3. **`output_scale` is unnecessary.** No empirical benefit observed.

## D.8 Recommendations

1. **Fix the `inject_performer` raise.**
2. **Remove `output_scale`.**
3. **Add a warning for cross-attention bypass.**
4. **Use $F \geq 64$.**

---