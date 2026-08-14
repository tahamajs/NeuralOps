import os

out_path = "/Users/tahamajs/.gemini/antigravity-cli/brain/eccd3f7b-8f47-4151-aaae-abb6b19a51db/complete_svd_compression_code_manual.md"

header = """# Architectural Manual & Theoretical Guide: SVD, Low-Rank, Matrix Factorization, and Adaptive Compression Engines

> **Document Status**: Production / Publication Ready  
> **Target Framework**: PyTorch Engine `Codes/framework/compression_engine/`  
> **Author**: Antigravity AI Engineering & Deep Learning Optimization Team  

---

## Executive Summary & Architectural Taxonomy

Low-rank approximation, matrix factorization, and singular value decomposition (SVD) represent the cornerstone of modern neural network compression. By factorizing heavy weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into compact low-rank factors $U \in \mathbb{R}^{d_{\text{out}} \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$, and $V^\top \in \mathbb{R}^{r \times d_{\text{in}}}$ (where $r \ll \min(d_{\text{out}}, d_{\text{in}})$), low-rank compression engines achieve dramatic parameter reduction, lower FLOP footprints, and reduced GPU VRAM consumption while preserving task accuracy.

This manual provides an exhaustive, code-level and mathematical walkthrough of all **16 compression engines** located in `Codes/framework/compression_engine/`.

### Comprehensive Taxonomy & Comparative Matrix

| Engine File | Primary Python Class / Entry | Factorization Family | Optimization Objective / Metric | Calibration Required? | FLOP Reduction Ratio | Parameter Count Ratio |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| [`svd_linear.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/svd_linear.py) | `SVDLinearLayer` | Truncated SVD | Static Rank Truncation $\|W - U_r \Sigma_r V_r^\top\|_F$ | No | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}} + 1)}{d_{\text{in}} d_{\text{out}}}$ |
| [`adasvd_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adasvd_compressor.py) | `AdaSVDCompressor` | Adaptive SVD | Nuclear Norm $\|W\|_*$ & Alt. Optimization | No | Dynamic per layer | Variable target ratio |
| [`aa_svd_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/aa_svd_compressor.py) | `AASVDCompressor` | Anchored SVD | Activation Subspace Projection $W A A^\top$ | Yes | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ |
| [`dobi_svd.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/dobi_svd.py) | `DobiSVDLinearLayer` | Dynamic Differentiable SVD | Learnable Rank Ratio $\theta \in [0,1]$ | No | Dynamic ($\theta$) | Dynamic ($\theta$) |
| [`erc_svd.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/erc_svd.py) | `ERCCompressor` | Error-Controlled SVD | Cumulative Singular Energy $E(r) \ge 1 - \epsilon$ | No | Adaptive per layer | Energy-bounded |
| [`mgie_svd_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/mgie_svd_compressor.py) | `MGIESVDBudgetCompressor` | Information Entropy SVD | Singular Distribution Entropy $H(W)$ Budget | No | Global Rank Allocation | Global Rank Budget |
| [`swift_svd_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/swift_svd_compressor.py) | `SwiftSVDActivationCompressor` | Cross-Covariance SVD | Act Cross-Covariance $C = \frac{1}{M} Y^\top X$ | Yes | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ |
| [`saes_svd_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/saes_svd_compressor.py) | `SAESCompressor` | Error-Compensated SVD | Multi-Step SVD Reconstruction | No | Fixed ratio $\alpha$ | Fixed ratio $\alpha$ |
| [`rpca_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/rpca_compressor.py) | `RPCACompressor` | Robust PCA ($L + S$) | Principal Component Pursuit $\|L\|_* + \lambda \|S\|_1$ | No | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ |
| [`nmf_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/nmf_compressor.py) | `NMFCompressor` | Non-negative Matrix Fact. | Shifted Non-negative $\|W^+ - H W_c\|_F^2$ | No | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} d_{\text{out}}}$ |
| [`cur_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cur_compressor.py) | `CURLinear`, `CURCompressor` | CUR Matrix Decomposition | Statistical Leverage Score Sampling | No | $\frac{r(d_{\text{in}} + d_{\text{out}} + r)}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}} + r)}{d_{\text{in}} d_{\text{out}}}$ |
| [`cur_linear.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cur_linear.py) | `CURLinear` | CUR Linear Layer | Leverage Score Column/Row Pseudoinverse | No | $\frac{r(d_{\text{in}} + d_{\text{out}} + r)}{d_{\text{in}} d_{\text{out}}}$ | $\frac{r(d_{\text{in}} + d_{\text{out}} + r)}{d_{\text{in}} d_{\text{out}}}$ |
| [`cur_moe_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cur_moe_compressor.py) | `CURMoECompressor` | MoE CUR Decomposition | MoE Expert Subspace Leverage Sampling | No | Per-expert low-rank | Per-expert low-rank |
| [`cure.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cure.py) | `cur_compress`, `CURMoECompressor` | Interpretable MoE CUR | Expert Pseudoinverse Selection $C U R$ | No | Per-expert low-rank | Per-expert low-rank |
| [`adaptive_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adaptive_compressor.py) | `AdaptiveTransformerCompressor` | Energy Multihead SVD | QKV / Out Energy Thresholding $\tau \ge 0.95$ | No | Dynamic Attention FLOPs | Dynamic Attention Params |
| [`adaptive_ranks.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adaptive_ranks.py) | `AdaptiveTransformerCompressor` | Attention Singular Spectrum | QKV / Out Energy Spectrum Allocation | No | Dynamic Attention FLOPs | Dynamic Attention Params |

---

## Unified Theoretical Framework & Execution Pipeline

```mermaid
flowchart TD
    A[Uncompressed Weight Matrix W in R^{d_out x d_in}] --> B{Calibration Needed?}
    B -- Yes: AASVD / Swift-SVD --> C[Run Calibration Forward Pass]
    C --> D1[Collect Activations X in R^{M x d_in}]
    C --> D2[Collect Outputs Y in R^{M x d_out}]
    D1 --> E1[Covariance X^T X or Cross-Covariance Y^T X]
    D2 --> E1
    E1 --> F[Singular Value Spectrum Decomposition]
    B -- No: Standard / Adaptive SVD / CUR / RPCA / NMF --> F
    F --> G{Rank Selection Engine}
    G -- Static Rank r --> H1[Truncate U_r, S_r, V_r^T]
    G -- Energy Threshold \tau --> H2[Energy Sum E_r >= \tau]
    G -- Entropy Budget --> H3[Layer Allocation H_l / sum H * B_rank]
    G -- Leverage Score Sampling --> H4[Sample Columns C & Rows R]
    H1 --> I[Factorized Layer Reconstruction]
    H2 --> I
    H3 --> I
    H4 --> I
    I --> J[Triton Fused Low-Rank GEMM Execution x @ V^T @ U^T]
```

---
"""

print("Base header compiled successfully.")
