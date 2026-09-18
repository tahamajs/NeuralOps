# 8. Detailed Analysis
On RoBERTa, `adasvd_compressor` compresses 40 linear projections, shedding ~14.9M parameters (11.96% parameter reduction) while retaining 73.39% accuracy. On DistilBERT and GPT-2, safety guards correctly protected sensitive layers.


## 4. Parameter Savings & Memory Footprint Profiling

<!-- source: Codes/frontend/dist/Docs/content/compressions/adasvd_compressor.md -->
### 4.1 Global Parameter Budget Formulas
Let $P_{\text{total}} = \sum_{l=1}^L (M_l \cdot N_l)$ be the initial total linear parameter count.
Target parameter removal fraction $\rho \in (0, 1)$ establishes the compression objective:

\[
P_{\text{target\_retained}} = (1 - \rho) \cdot P_{\text{total}}
\]

Under adaptive allocation $r_l$, the retained parameter count $P_{\text{retained}}$ is:

\[
P_{\text{retained}} = \sum_{l=1}^{L} r_l (M_l + N_l + 1)
\]

### 4.2 Comparative Parameter & Memory Profiling (LLM-7B Scale, FP16)

| Layer Type | Base Shape ($M \times N$) | Baseline Params | Uniform SVD Rank ($r=256$) | AdaSVD Rank ($r_l$) | AdaSVD Retained Params | Memory Savings |
|------------|---------------------------|-----------------|-----------------------------|----------------------|------------------------|----------------|
| **q_proj**  | $4096 \times 4096$        | 16,777,216      | $r=256 \to 2.09\text{M}$    | $r_l=384$            | 3,146,112              | 81.25%         |
| **k_proj**  | $4096 \times 4096$        | 16,777,216      | $r=256 \to 2.09\text{M}$    | $r_l=192$            | 1,573,056              | 90.62%         |
| **v_proj**  | $4096 \times 4096$        | 16,777,216      | $r=256 \to 2.09\text{M}$    | $r_l=448$            | 3,670,272              | 78.12%         |
| **o_proj**  | $4096 \times 4096$        | 16,777,216      | $r=256 \to 2.09\text{M}$    | $r_l=320$            | 2,621,696              | 84.37%         |
| **gate_proj**| $11008 \times 4096$       | 45,088,768      | $r=256 \to 3.86\text{M}$    | $r_l=512$            | 7,731,712              | 82.85%         |
| **up_proj** | $11008 \times 4096$       | 45,088,768      | $r=256 \to 3.86\text{M}$    | $r_l=512$            | 7,731,712              | 82.85%         |
| **down_proj**| $4096 \times 11008$       | 45,088,768      | $r=256 \to 3.86\text{M}$    | $r_l=640$            | 9,664,768              | 78.56%         |
| **Total**  | --                        | **201,675,008** | **21,940,224**              | --                   | **36,139,328**         | **82.08%**     |

---


## 5. Precision Loss, Perplexity & Sensitivity Bounds

<!-- source: Codes/frontend/dist/Docs/content/compressions/adasvd_compressor.md -->
### 5.1 Error Bounding via Adaptive Rank Allocation
Under AdaSVD's nuclear norm rank distribution, the overall model Frobenius error sum $\mathcal{E}_{\text{global}}$ is upper-bounded by:

\[
\mathcal{E}_{\text{global}} = \sum_{l=1}^{L} \|W^{(l)} - \hat{W}^{(l)}\|_F^2 \le \sum_{l=1}^{L} \sum_{i=r_l + 1}^{\min(M_l, N_l)} (\sigma_i^{(l)})^2
\]

Because $r_l$ scales monotonically with total spectral energy $\sum \sigma_i^{(l)}$, high-entropy layers retain a larger fraction of their tail singular values, preventing error spikes in sensitive layers.

### 5.2 Failure Modes & Edge Cases
1. **Degenerate Uniform Spectrum**: If a layer has flat singular values ($\sigma_1 \approx \sigma_2 \approx \dots \approx \sigma_R$), nuclear norm weighting allocates high rank, but low-rank truncation still loses substantial energy proportional to $(R - r_l) \cdot \sigma_1$.
2. **Matrix Inverse Instability**: During iterative refinement, computing $(U_{r_l}^T U_{r_l} + \epsilon I)^{-1}$ can become ill-conditioned if $\epsilon$ is too small ($< 10^-8$) and $U_{r_l}$ loses rank, requiring robust Tikhonov regularization.

---
