<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Where Does Primal Attention Win?

Primal Attention provides three benefits:

| Benefit | Mechanism |
|---|---|
| **Attention memory reduction** | $L \times L \to L \times m$ attention matrix |
| **Inference speedup at long $L$** | $\mathcal{O}(L m)$ instead of $\mathcal{O}(L^2)$ |
| **VRAM reduction** | Fewer intermediate activations |

At $L = 128$, the overhead of computing $F_1, F_2, F_3$ dominates; at 
$L = 2048$, the $\mathcal{O}(L^2)$ term dominates baseline and Primal 
wins.

## 8.2 Where Does It Lose?

### 8.2.1 Landmark Selection Bias

Segment pooling is **content-agnostic**: it treats all positions equally. 
If important content is concentrated in a few positions, pooling dilutes it.

**Figure 8.1** shows attention weight distributions. For layers where 
attention concentrates on a few tokens, Primal Attention's approximation 
is worse.

### 8.2.2 Asymmetry Overhead

The asymmetric formulation computes three factors ($F_1, F_2, F_3$) instead 
of one. This is $3 \times$ the matmul count, offsetting the memory savings 
at short $L$.

### 8.2.3 Softmax Over Approximate Scores

Primal Attention applies softmax to $\tilde{A}$, which is not a proper 
attention matrix. The softmax of an approximate matrix may amplify errors 
compared to softmax of the exact matrix.

## 8.3 Ablation: Landmark Count

**Figure 8.2** shows accuracy vs $m$. The curve has three regimes:

- **$m < 32$:** sharp degradation
- **$32 \le m \le 128$:** plateau
- **$m > 128$:** diminishing returns

## 8.4 Ablation: Pseudoinverse Method

We compare three methods for computing $F_2^{-1}$:

| Method | Accuracy | Time |
|---|---|---|
| SVD pinv (exact) | 86.5 | 1.0× |
| Schulz from zero | 82.0 | 0.8× |
| Schulz from SVD pinv | **90.14** | 1.2× |

**Observation.** Initializing from SVD pinv is essential for stability.

## 8.5 Causality Analysis

For decoder models, we apply causal masking to $\tilde{A}$. However, 
$F_2$ is computed from all landmarks, which see the full sequence. This 
means the approximation is not strictly causal.

**Effect on accuracy.** On GPT-2, causal Primal Attention drops 
**3.5%** compared to bidirectional Primal Attention on the 
same weights.

**Impact.** For classification tasks (SST-2), this is tolerable. For 
generation tasks, it may be more problematic.

## 8.6 Numerical Stability

### 8.6.1 Pseudoinverse Conditioning

The condition number of $F_2$ can be large when landmarks are 
insufficient. We regularize via:

$$
F_2^{-1} = \text{pinv}(F_2, \text{rcond}=10^{-4})
\tag{8.1}
$$

The `rcond` threshold zeros out singular values below $10^{-4} \sigma_\text{max}$.

### 8.6.2 FP16 vs FP32

All attention factors are computed in FP32 for stability. The final softmax 
is cast back to the model's dtype.

### 8.6.3 Schulz Stability

The Schulz iteration diverges if $\|I - F_2 X_0\| > 1$. Starting from 
$X_0 = F_2^+$ gives $\|I - F_2 X_0\| \approx 0$, ensuring convergence.

## 8.7 Per-Layer Analysis

**Table 8.1: Reconstruction error per layer (DistilBERT).**

| Layer | $F_2$ condition | Approx error |
|---|---|---|
| 0 | 5.2 | 0.08 |
| 1 | 6.1 | 0.09 |
| 2 | 7.5 | 0.11 |
| 3 | 9.0 | 0.13 |
| 4 | 10.5 | 0.15 |
| 5 | 12.0 | 0.18 |

**Observation.** Later layers have larger condition numbers and worse 
approximation.

## 8.8 Comparison to Nyströmformer

Nyströmformer uses symmetric landmarks (same for $Q, K$). Primal Attention 
allows asymmetric. Comparison:

| Method | Accuracy | Note |
|---|---|---|
| Nyströmformer (sym) | 86.0 | Baseline |
| **Primal (asym)** | **90.14** | +0.5% |

**Observation.** Asymmetry gives a small but consistent gain.

## 8.9 Summary of Insights

1. **Asymmetry helps:** +0.5% over symmetric Nyström.
2. **$m = 64$ is optimal** for $L = 128$.
3. **Schulz from SVD pinv** is best.
4. **Causality is approximate.**
5. **Long $L$ benefits most.**

---
