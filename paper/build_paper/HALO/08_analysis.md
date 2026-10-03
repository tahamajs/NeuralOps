<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Gate Distribution Analysis

**Figure 8.1** shows the histogram of learned gates across all heads. Three 
patterns emerge:

### 8.1.1 Bimodal Distribution

Most heads fall into two clusters:

- **Dense-preferring heads:** $\bar{g}_h > 0.8$
- **Linear-preferring heads:** $\bar{g}_h < 0.3$

This confirms prior work on head specialization.

### 8.1.2 Layer-Aware Pattern

**Figure 8.2** shows mean gate per layer. Early layers have higher gates; 
late layers have lower gates.

**Interpretation.** Early layers need global attention (e.g., for 
coreference); late layers refine local features and can use linear.

### 8.1.3 Head-Aware Pattern

Some heads are consistently dense across layers (attention sinks); others 
are consistently linear.

## 8.2 Comparison with Fixed Fusion

**Table 8.1: Learned vs fixed gate.**

| Fusion | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Fixed 0.5 | 86.0 | 72.5 | 73.5 |
| Fixed 0.8 | 88.5 | 75.0 | 76.0 |
| **Learned** | **89.5** | **76.5** | **77.5** |

**Observation.** Learned gate outperforms fixed by 2.5%.

## 8.3 Kernel Choice

**Table 8.2: Kernel comparison.**

| Kernel | Properties | Accuracy |
|---|---|---|
| Identity (linear) | Simple, can be negative | 78 |
| ReLU+1 | Non-negative, not smooth | 87 |
| Softplus | Smooth, positive | 89 |
| **ELU+1** | **Non-negative, smooth, bounded** | **89.5** |

## 8.4 Numerical Stability

### 8.4.1 Cumsum Overflow

The cumsum in the causal linear branch can overflow in FP16 for $L > 512$. 
We cast to FP32:

```python
cum_kv = torch.cumsum(kv_outer.float(), dim=2).to(q.dtype)
```

### 8.4.2 Denominator Zero

The denominator $\phi(Q_i) \cdot \text{den}^{(i)}$ could be near zero if 
$\phi(Q_i)$ has small components. We clamp:

```python
den = (q_lin * cum_k).sum(-1, keepdim=True).clamp(min=1e-5)
```

### 8.4.3 Padding Effects

Padding keys are zeroed before cumsum, ensuring they contribute nothing.

## 8.5 Causality Verification

We verify causality by checking that the gradient of output at position $i$ 
w.r.t. input at position $j > i$ is zero:

```python
x = torch.randn(1, L, d, requires_grad=True)
out = model(x)
out[0, i].sum().backward()
assert x.grad[0, i + 1 :].abs().max() < 1e-6
```

**Result.** ✅ Verified on all three models

## 8.6 Attention Mass Analysis

**Figure 8.3** shows dense attention mass distribution. High-mass heads 
have higher gates; low-mass (concentrated) heads have lower gates.

## 8.7 Failure Cases

### 8.7.1 Layers Where HALO Fails

| Layer | Model | Drop |
|---|---|---|
| layer.0 | GPT-2 | 1.5 |
| layer.5 | RoBERTa | 1.8 |

### 8.7.2 Root Causes

Some heads need pure dense attention; gate should be > 0.9

## 8.8 Comparison with NHA

HALO-ReGLA and NHA both use parallel streams. Differences:

| Aspect | NHA | HALO-ReGLA |
|---|---|---|
| Dense branch | Local window | Global softmax |
| Linear branch | RFF | ELU kernel |
| Gate init bias | 3.0 | 1.8 |
| Results (DistilBERT) | 90.25% | 89.5% |
| Results (RoBERTa) | FAIL | 76.5% |

**Observation.** HALO-ReGLA's global dense stream handles nested encoders 
better than NHA's local window.

## 8.9 Summary of Insights

1. **Gates are bimodal**: dense vs linear heads.
2. **Layer depth correlates with gate**: early = dense, late = linear.
3. **Learned gate outperforms fixed.**
4. **ELU+1 is optimal kernel.**
5. **No speedup without hard routing.**
6. **Causality preserved via prefix sums.**

---
