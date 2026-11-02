# NASC: NEXUS-AEGIS Spectral Cascade Attention

## Overview
**NASC (NEXUS-AEGIS Spectral Cascade)** is a complexity-adaptive hybrid attention architecture that unifies the strengths of localized causal sliding windows, block-sparse Mixture-of-Experts routing, and global Random Fourier Feature (RFF) linear kernel approximations into a single gated layer.

---

## Architectural Taxonomy & Gated Routing

```
                          Input Sequence X ∈ ℝ^{B × L × D}
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                   Local Window     Block-Sparse     Global Kernel
                   (AEGIS SVD)        (NEXUS)        (SPECTRA RFF)
                         │               │               │
                      O_loc           O_sp            O_glb
                         │               │               │
                         └───────────────┼───────────────┘
                                         ▼
                            Dynamic Gating Router
                       g = Softmax(Linear(Mean(X)))
                                         ▼
                     O = g₁·O_loc + g₂·O_sp + g₃·O_glb
                                         ▼
                                Output Projection
```

---

## Mathematical Formulation

1. **Local Window Attention ($O_{\text{loc}}$):**
   $$Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}} = \text{SVD-LoRA}(X)$$
   $$O_{\text{loc}} = \text{LocalWindowAttn}(Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}}, W_{\text{sz}})$$

2. **Block-Sparse Routing ($O_{\text{sp}}$):**
   $$S_{ij}^{\text{blk}} = \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}}, \quad \mathcal{I}_i = \operatorname{TopK}(S_{i,:}^{\text{blk}}, k)$$
   $$O_{\text{sp}} = \operatorname{Softpick}\left(\frac{Q_i K_{\mathcal{I}_i}^\top}{\sqrt{d_k}}\right) V_{\mathcal{I}_i}$$

3. **Global Spectral Kernel ($O_{\text{glb}}$):**
   $$\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_{\text{rff}}), \sin(x W_{\text{rff}})]$$
   $$O_{\text{glb}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + \epsilon}$$

4. **Composite Synthesis:**
   $$O = g_1 \cdot O_{\text{loc}} + g_2 \cdot O_{\text{sp}} + g_3 \cdot O_{\text{glb}}$$

---

## Computational Complexity Breakdown

| Regime | Dominant Mechanism | Complexity | Memory Scaling |
| :--- | :--- | :--- | :--- |
| $L \le W_{\text{sz}}$ | Local Sliding Window | $\mathcal{O}(L \cdot W_{\text{sz}} \cdot d)$ | $\mathcal{O}(W_{\text{sz}} \cdot d)$ |
| $W_{\text{sz}} < L \le k \cdot B_{\text{sz}}$ | Sparse Inter-Block Routing | $\mathcal{O}(L \cdot k B_{\text{sz}} \cdot d)$ | $\mathcal{O}(k B_{\text{sz}} \cdot d)$ |
| $L > k \cdot B_{\text{sz}}$ | Global Spectral RFF Kernel | $\mathcal{O}(L \cdot F \cdot d)$ | $\mathcal{O}(F \cdot d)$ |

**Overall Complexity:** Strictly $\mathcal{O}(L)$ linear time and memory in sequence length.
