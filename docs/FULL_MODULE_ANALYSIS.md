# Full Module Analysis

Investigation into the `full` module identifier, its origin in `FullAttention`, and its extensive presence in experimental results.

## 1. Registry Definition

In `neuralops/attentions/dense/full_attention.py`:
```python
@ATTENTION_REGISTRY.register(
    "full", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "full_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class FullAttention(BaseAttention):
```
- Both `"full"` and `"full_attention"` are registered on `FullAttention`.
- `FullAttention` represents the canonical uncompressed, dense quadratic multi-head attention baseline.

## 2. Why 1,205 Results in Master Summary?

- Every unified benchmark run, rank sweep, and fine-tuning suite includes an uncompressed dense baseline (`full`) evaluated under the exact same data split, seed, and hardware parameters to compute relative speedup, memory reduction, and retention metrics.
- As a consequence, `full` is evaluated alongside every compressed attention operator across all benchmark suites.

## 3. Recommendation

- **Keep all 1,205 records**: They provide essential ground-truth baselines for relative accuracy degradation and speedup comparisons.
- **Deduplicate Registry**: Keep `"full_attention"` as the canonical descriptive key in `ATTENTION_REGISTRY`, remove `@ATTENTION_REGISTRY.register("full")`, and route `"full"` via `deprecated_aliases.json`.
- In `_MASTER_SUMMARY.csv`, map `"full"` to `"full_attention"` for naming consistency across tables.
