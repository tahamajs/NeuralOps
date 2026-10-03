<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why SVD-Mamba Has a Hard Time

SVD-Mamba has three paths that must be jointly optimized:

1. Attention (initialized from pretrained).
2. SSM (random initialization, silenced).
3. Global mixing (random).

**This creates a fundamental tension.** The SSM and global paths add 
random noise that must be suppressed. The gate starts at 0.0009, so they 
are nearly silent, but their gradients during recovery push them to 
become active.

**Result.** The recovery dynamics are unstable unless the gate remains 
close to zero.

## 8.2 The Gate Dynamics

**Figure 8.1** shows $\gamma$ during recovery for all three models.

| Step | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.0009 | 0.0009 | 0.0009 |
| 10 | 0.85 | 0.85 | 0.85 |
| 25 | 0.85 | 0.85 | 0.85 |
| 50 | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 8.3 Numerical Stability Analysis

The SSM scan is numerically delicate. We use multiple clamps:

| Operation | Clamp | Why |
|---|---|---|
| `log_alpha` cumsum | $[-25, 0]$ | Prevent explosion |
| `h_chunk` | $[-10^4, 10^4]$ | Prevent overflow |
| `h_den_chunk` | $[0, 10^4]$ | Keep denom non-negative |
| `decay_inv` | $\leq 10^{10}$ | Prevent overflow |
| `denom` | $\geq 10^{-4}$ | Prevent divide-by-zero |
| `A_log` | $[-20, 10]$ | Keep $A$ bounded |

**Figure 8.2** shows the distribution of `h_chunk` values. 0.85

## 8.4 The Parameter Increase

**Table 8.1: Parameter breakdown per layer (DistilBERT).**

| Component | Baseline | SVD-Mamba | Δ |
|---|---|---|---|
| Q proj | 589,824 | 84.5 | 0.85 |
| K proj | 589,824 | 84.5 | 0.85 |
| V proj | 589,824 | 84.5 | 0.85 |
| O proj | 589,824 | 84.5 | 0.85 |
| **Attn total** | 2,359,296 | 84.5 | 0.85 |
| SSM branch | 0 | 84.5 | +84.5 |
| Gating | 0 | 0.94 | +0.94 |
| Global | 0 | 0.85 | +0.85 |
| **Total** | **2,359,296** | **66.4M** | **+0.85** |

**Net increase.** ~48.5\%% on DistilBERT.

## 8.5 Comparison with Pure Compression Methods

| Method | Params | Accuracy | Trade-off |
|---|---|---|---|
| AASVD (r=64) | −80% | −2.2% | Size ↓, acc slight ↓ |
| ASVD (r=64) | −82% | −2.2% | Same |
| **SVD-Mamba** | **+66.4M%** | -39.79% | Params ↑, new capability |

**SVD-Mamba is not a compressor.** It replaces attention with a hybrid 
mechanism.

## 8.6 SSM Branch Contribution

**Figure 8.3** shows the output contribution of each path over training:

| Step | Attn contribution | SSM contribution |
|---|---|---|
| 0 | 99.9% | 0.1% |
| 25 | 0.85% | 0.85% |
| 50 | 0.85% | 0.85% |

**Observation.** 0.85

## 8.7 Inference Cost Analysis

**Table 8.2: Time breakdown per layer (DistilBERT, L=128).**

| Component | Baseline | SVD-Mamba |
|---|---|---|
| Q/K/V/O proj | 0.5 ms | 2.3 s ms |
| Attention scores | 0.4 ms | 0.4 ms |
| SSM branch | 0 | 2.3 s ms |
| Global | 0 | 2.3 s ms |
| Gating | 0 | 2.3 s ms |
| LoRA | 0 | 2.3 s ms |
| **Total** | **1.4 ms** | **2.3 s ms** |

## 8.8 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| Collapse | Acc ≈ 50.92% | Weight transfer failed |
| NaN in SSM | Loss = NaN | Overflow in scan |
| Slower than baseline | >3× latency | Extra paths |
| Parameter bloat | +66.4M% | SSM + gating |

## 8.9 The `raise` Problem

SVD-Mamba uses `raise RuntimeError` on missing projections. Unlike 
`inject_rpc` (which also raises), the inject hook here catches the 
exception and returns `None`:

```python
except Exception as e:
    logger.error("inject_svd_mamba failed at layer %d: %s", layer_idx, e)
    return None
```

This means **silent failure is possible** if injection partially succeeds.

## 8.10 Summary of Insights

1. **SVD-Mamba is not a compressor.**
2. **Three-path architecture is complex.**
3. **Gate initialization is critical.**
4. **SSM scan needs careful clamping.**
5. **Recovery dynamics are unstable.**

---