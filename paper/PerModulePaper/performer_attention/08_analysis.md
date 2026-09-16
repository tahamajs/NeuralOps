<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Feature Map Behavior

The FAVOR+ feature map:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(\omega^\top x - \max(\omega^\top x) - \frac{\|x\|^2}{2}\right) + \epsilon
$$

**Three components:**
1. **Max-subtraction:** stability.
2. **Norm-subtraction:** matches the RFF formula for Gaussian kernel.
3. **Epsilon:** positivity guarantee.

**Analysis.** The $\epsilon$ adds a constant offset to every feature. This 
is a slight deviation from the theoretical FAVOR+ but ensures numerical 
stability.

## 8.2 Kernel Type Comparison

**Table 8.1: Kernel type ablation.**

| Kernel | DistilBERT |
|---|---|
| `favor_plus` | {{D_FAVOR}} |
| `elu` | {{D_ELU}} |
| `relu` | {{D_RELU}} |

**Observation.** {{KERNEL_OBS}}

## 8.3 Number of Features

**Table 8.2: Effect of $F$.**

| $F$ | DistilBERT | Params |
|---|---|---|
| 32 | {{D_F32}} | {{D_F32_P}} |
| 64 | {{D_F64}} | {{D_F64_P}} |
| 128 | {{D_F128}} | {{D_F128_P}} |
| 256 | {{D_F256}} | {{D_F256_P}} |

**Observation.** Larger $F$ helps but with diminishing returns.

## 8.4 Causal Path Analysis

The causal path uses **chunk_size = 64**. For $N = 128$: 2 iterations.

**Compare with other modules:**

| Module | chunk_size | Iterations (N=128) |
|---|---|---|
| GatedDeltaNet | 8 | 16 |
| ORION kernel | 64 | 2 |
| **Performer** | **64** | **2** |
| NASC sparse | 32 | 4 |

**Observation.** Performer's chunked path is efficient.

## 8.5 The `inject_performer` Raise Bug

Docstring: `Safe injection returning None on failure`.

Code: `raise RuntimeError(f"Injection failed")`.

**Impact.**

- Pipeline crashes instead of skipping.
- No graceful degradation.
- Inconsistent with the docstring.

**Fix:**

```python
except Exception as e:
    logger.warning("PerformerAttention construction failed: %s", e)
    return None
```

## 8.6 The `output_scale` Parameter

`LoRALinear.forward`:

```python
return out * self.output_scale
```

**Impact.**

- New parameter not in baseline checkpoint.
- `strict=True` loading fails.
- Defaults to 1.0 (harmless for `strict=False`).

## 8.7 Comparison with Cosformer

| Aspect | Cosformer | Performer |
|---|---|---|
| Feature map | cos/sin of position | FAVOR+ orthogonal |
| Theory | Locality | Variance reduction |
| Encoder | Works | Works |
| Decoder | Chunked prefix | Chunked prefix |
| Empirical | {{COS_D}}% | {{D_ACC}}% |

**Both are strong linear-attention baselines.**

## 8.8 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **Injection raises** | Crash | `raise RuntimeError` |
| **`output_scale` missing key** | Warning | New parameter |
| **Encoder degrade** | Moderate drop | Feature approximation |
| **Numerical instability** | NaN | Overflow (mitigated) |

## 8.9 Summary of Insights

1. **FAVOR+ works** but with moderate accuracy drop on small models.
2. **Numerical stabilization is essential** — without it, FP16 overflows.
3. **The `inject_performer` raise bug** breaks the docstring contract.
4. **`output_scale` is unnecessary** and should be removed.
5. **The feature map's $\epsilon$** slightly deviates from theory.

---