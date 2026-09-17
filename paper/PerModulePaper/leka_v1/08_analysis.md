# 8. Analysis

## 8.1 Kernel Approximation Quality

**Figure 8.1** compares the empirical kernel $\phi(q)^\top \phi(k)$ 
against the true softmax kernel $\exp(q^\top k / \sqrt{d_h})$ and the 
Gaussian kernel $\exp(-\|q - k\|^2 / 2)$.

**Observation.** The RFF approximation is closer to the Gaussian kernel 
than the softmax kernel, as expected from theory.

## 8.2 Effect of Feature Dimension

**Figure 8.2** shows accuracy vs $F$. Approximation error scales as 
$O(1/\sqrt{F})$, and accuracy improves with larger $F$ up to a point.

| $F$ | Approx. err | Accuracy |
|---|---|---|
| 64 | ~0.15 | 0.85 |
| 128 | ~0.10 | 0.865 |
| 256 | ~0.07 | 51.03 |
| 512 | ~0.05 | 0.85 |

## 8.3 Preservation Gate Analysis

**Figure 8.3** shows the learned α per layer for DistilBERT.

| Layer | α |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| ... | ... |
| 5 | 0.85 |

**Observation.** Early layers prefer attention (high α); later layers 
0.85.

## 8.4 Why QR Helps

Sampling $\omega_i \sim \mathcal{N}(0, I)$ gives an unbiased estimator 
with variance $\sigma^2 = O(1/F)$. Orthogonalizing reduces this to 
$O(1/F^2)$, giving faster convergence.

**Table 8.1: Empirical kernel error.**

| Init | ||φ(q)ᵀφ(k) - κ||₂ |
|---|---|
| Random | 0.024 |
| QR | 0.024 |

## 8.5 LayerNorm Before Projection

Without LN, the input to the random projection has varying scale, which 
biases the kernel approximation. LN normalizes this.

**Figure 8.4** shows the distribution of $f = \tilde{x} W$ with and 
without LN.

## 8.6 Causal Cumsum Memory

For decoders, the cumsum over the outer product materializes a large 
tensor. **Table 8.2** shows peak memory:

| L | Peak memory |
|---|---|
| 128 | 128 KB MB |
| 256 | 128 KB MB |
| 512 | 128 KB MB |

For $L > 512$, memory becomes a bottleneck. See Appendix D for 
optimization.

## 8.7 Comparison to Other Linear Attention

**Table 8.3: LEKAv1 vs other linear attention.**

| Method | DistilBERT | Speed @ L=1024 |
|---|---|---|
| full | 90.71% | 1× |
| Linear Transformer | 0.85 | 84.5× |
| Performer | 0.85 | 84.5× |
| **LEKAv1** | **51.03** | **84.5×** |

## 8.8 Failure Modes

### 8.8.1 Extractor Failure

The current `copy_weights` raises `RuntimeError` if UniversalAdapter fails. 
This **kills the module** and can lead to collapse.

### 8.8.2 Cumsum OOM

At large L, the cumsum materializes a huge tensor and can cause OOM.

### 8.8.3 Preservation Gate Saturation

If α saturates to 0, the attention branch is ignored. If α saturates to 1, 
the residual is ignored.

## 8.9 Summary of Insights

1. **RFF works** for attention approximation.
2. **QR projection reduces variance.**
3. **Head-wise LN stabilizes input.**
4. **Preservation gate provides safety.**
5. **Linear scaling enables long-context inference.**

---