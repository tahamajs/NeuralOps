<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 The `copy_weights` Raise Bug

```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```

**Impact.** Any UniversalAdapter failure aborts injection.

**Fallback.** `inject_orion` has its own try/except → returns None.

## D.2 The `rff_proj` Persistence Bug

```python
self.register_buffer("rff_proj", ..., persistent=False)
```

**Impact.** Train/test mismatch on model save/reload.

**Verification.**

```python
sd = model1.state_dict()
assert 'rff_proj' not in sd  # Confirms bug
```

## D.3 Softpick Masked Row Collapse

```python
p = F.relu(torch.exp(x_f) - 1.0)
return p / (p.sum(dim=dim, keepdim=True) + 1e-8)
```

If all keys in a row are masked, `p.sum() = 0` → output is zero.

**Impact.** For fully-masked query rows, attention is undefined.

## D.4 Failed Ablations

### Too Aggressive SVD

| Energy | DistilBERT |
|---|---|
| 0.80 | 0.85 |
| 0.90 | 0.85 |
| 0.95 | 0.85 |
| 0.99 | 0.85 |

**Observation.** 0.80 over-prunes.

### All-Kernel

| Split | DistilBERT |
|---|---|
| All kernel | 0.85 |
| All full | 0.85 |

**Observation.** All-kernel loses accuracy. Full heads are necessary.

### Small RFF Features

| Features | DistilBERT |
|---|---|
| 32 | 0.85 |
| 128 | 0.865 |
| 256 | 0.85 |

**Observation.** RFF with <128 features underperforms.

## D.5 Failed Alternatives

| Attempt | Result |
|---|---|
| Per-head rank | Not implemented |
| Learn split | Not implemented |
| Persistent RFF | Not fixed |

## D.6 Lessons Learned

1. **`copy_weights` robustness matters.** Raise-on-failure is dangerous.
2. **Persistent buffers matter.** `persistent=False` breaks save/reload.
3. **Softpick needs fallback.** Masked rows otherwise collapse.
4. **Head bifurcation works.** 2/3 : 1/3 split is a good default.

## D.7 Recommendations

1. **Fix the `copy_weights` raise.**
2. **Set `persistent=True`** for `rff_proj`.
3. **Add softpick fallback.**
4. **Test per-head rank selection.**

## D.8 What Would Make This Work?

1. **Fixed bugs** — production-ready.
2. **Fused kernel** — faster factorized projections.
3. **Learned head split** — adaptive bifurcation.

---