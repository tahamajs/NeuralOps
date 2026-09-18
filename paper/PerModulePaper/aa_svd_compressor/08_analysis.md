# 8. Detailed Analysis

## 8.1 Activation Whitening Impact
Traditional SVD truncates based solely on weight singular value spectra. However, input activations $X$ are non-spherical. By multiplying $W$ by $C_X^{1/2}$, singular vectors corresponding to high-variance activation directions are amplified, ensuring that the truncated rank-$r$ subspace preserves the components most responsible for signal propagation.

## 8.2 Conv1D Transposition Handling
In causal models such as GPT-2, projections are parameterized as `Conv1D(nf, nx)` where weights have shape $(nx, nf)$ rather than PyTorch's conventional `nn.Linear` shape $(nf, nx)$. `aa_svd_compressor` detects 1D convolutions and applies explicit weight transposition prior to covariance multiplication and factor assignment.


## 4. Parameter Savings & Memory Footprint Profiling

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 4.1 Memory & FLOP Formulas
When AASVD is deployed as a weight reconstruction mechanism (overwriting `module.weight.data` in-place), parameter memory remains identical to the original uncompressed model ($d_{\text{out}} \times d_{\text{in}}$). When deployed as a factorized linear layer ($U_r \operatorname{diag}(S_r) V_r^T$):

Specifically, regarding \textbf{Original Parameters}, $P_{\text{orig}} = d_{\text{out}} \cdot d_{\text{in}}$ Furthermore, regarding \textbf{Factorized Parameters}, $P_{\text{AASVD}} = r(d_{\text{out}} + d_{\text{in}} + 1)$ Concurrently, regarding \textbf{Inference FLOPs / Token}, $2 r (d_{\text{out}} + d_{\text{in}})$


### 4.2 Benchmark Footprint Table ($d_{\text{out}} = 4096, d_{\text{in}} = 4096$, Rank $r = 128$)

| Stage / Component | Operation | Matrix Dimensions | Tensor Element Count | Memory (FP16) |
|-------------------|-----------|-------------------|----------------------|---------------|
| **Calibration Activation** | Flatten Input | $128 \times 4096$ | 524,288 | 1.06 MB |
| **Anchor Basis $A_{\text{anchor}}$** | Activation Basis | $4096 \times 128$ | 524,288 | 1.06 MB |
| **Anchored Weight $W_{\text{anchored}}$** | $W P_{\text{anchor}}$ | $4096 \times 4096$ | 16,777,216 | 33.55 MB |
| **Factorized $U_r$** | Left Basis | $4096 \times 128$ | 524,288 | 1.07 MB |
| **Factorized $V_r^T$** | Right Basis | $128 \times 4096$ | 524,288 | 1.08 MB |
| **Total Factorized Layer** | $U_r, S_r, V_r^T$ | -- | **1,048,704** | **2.10 MB (93.7% savings)** |

---


## 5. Precision Loss, Perplexity & Sensitivity Bounds

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 5.1 Error Bound relative to Activation Space
Let $X$ be the activation tensor with principal covariance $C_X = X^T X$. The output error norm $\|E_{\text{out}}\|_F^2 = \|X W^T - X \hat{W}^T\|_F^2$ under AASVD satisfies:

\[
\|E_{\text{out}}\|_F^2 \le \|X (I - P_{\text{anchor}}) W^T\|_F^2 + \|X P_{\text{anchor}} (W - \hat{W}_{\text{AASVD}})^T\|_F^2
\]

By selecting $A_{\text{anchor}}$ as the top-$r$ eigenvectors of $C_X$, the term $\|X (I - P_{\text{anchor}})\|_F^2 = \sum_{i=r+1}^{d_{\text{in}}} \lambda_i(C_X)$ is minimized. Consequently, AASVD guarantees lower activation output distortion than standard static SVD:

\[
\|X W^T - X \hat{W}_{\text{AASVD}}^T\|_F^2 \le \|X W^T - X \hat{W}_{\text{SVD}}^T\|_F^2
\]

### 5.2 Failure Modes
1. **Out-of-Distribution Calibration Drift**: If calibration text sequences do not match deployment token distributions, $A_{\text{anchor}}$ isolates irrelevant activation subspaces, degrading real-world inference accuracy.
2. **Small Calibration Sample Size**: Calibrating with short sequence lengths ($K < d_{\text{in}}$) causes rank deficiency in $\bar{X}$, generating spurious zero singular values in $V_X^T$.

---


## 4. Parameter Savings & Memory Footprint Profiling

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 4.1 Memory & FLOP Formulas
When AASVD is deployed as a weight reconstruction mechanism (overwriting `module.weight.data` in-place), parameter memory remains identical to the original uncompressed model ($d_{\text{out}} \times d_{\text{in}}$). When deployed as a factorized linear layer ($U_r \operatorname{diag}(S_r) V_r^T$):

Specifically, regarding \textbf{Original Parameters}, $P_{\text{orig}} = d_{\text{out}} \cdot d_{\text{in}}$ Furthermore, regarding \textbf{Factorized Parameters}, $P_{\text{AASVD}} = r(d_{\text{out}} + d_{\text{in}} + 1)$ Concurrently, regarding \textbf{Inference FLOPs / Token}, $2 r (d_{\text{out}} + d_{\text{in}})$


### 4.2 Benchmark Footprint Table ($d_{\text{out}} = 4096, d_{\text{in}} = 4096$, Rank $r = 128$)

| Stage / Component | Operation | Matrix Dimensions | Tensor Element Count | Memory (FP16) |
|-------------------|-----------|-------------------|----------------------|---------------|
| **Calibration Activation** | Flatten Input | $128 \times 4096$ | 524,288 | 1.08 MB |
| **Anchor Basis $A_{\text{anchor}}$** | Activation Basis | $4096 \times 128$ | 524,288 | 1.09 MB |
| **Anchored Weight $W_{\text{anchored}}$** | $W P_{\text{anchor}}$ | $4096 \times 4096$ | 16,777,216 | 33.55 MB |
| **Factorized $U_r$** | Left Basis | $4096 \times 128$ | 524,288 | 1.10 MB |
| **Factorized $V_r^T$** | Right Basis | $128 \times 4096$ | 524,288 | 1.11 MB |
| **Total Factorized Layer** | $U_r, S_r, V_r^T$ | -- | **1,048,704** | **2.10 MB (93.7% savings)** |

---


## 5. Precision Loss, Perplexity & Sensitivity Bounds

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 5.1 Error Bound relative to Activation Space
Let $X$ be the activation tensor with principal covariance $C_X = X^T X$. The output error norm $\|E_{\text{out}}\|_F^2 = \|X W^T - X \hat{W}^T\|_F^2$ under AASVD satisfies:

\[
\|E_{\text{out}}\|_F^2 \le \|X (I - P_{\text{anchor}}) W^T\|_F^2 + \|X P_{\text{anchor}} (W - \hat{W}_{\text{AASVD}})^T\|_F^2
\]

By selecting $A_{\text{anchor}}$ as the top-$r$ eigenvectors of $C_X$, the term $\|X (I - P_{\text{anchor}})\|_F^2 = \sum_{i=r+1}^{d_{\text{in}}} \lambda_i(C_X)$ is minimized. Consequently, AASVD guarantees lower activation output distortion than standard static SVD:

\[
\|X W^T - X \hat{W}_{\text{AASVD}}^T\|_F^2 \le \|X W^T - X \hat{W}_{\text{SVD}}^T\|_F^2
\]

### 5.2 Failure Modes
1. **Out-of-Distribution Calibration Drift**: If calibration text sequences do not match deployment token distributions, $A_{\text{anchor}}$ isolates irrelevant activation subspaces, degrading real-world inference accuracy.
2. **Small Calibration Sample Size**: Calibrating with short sequence lengths ($K < d_{\text{in}}$) causes rank deficiency in $\bar{X}$, generating spurious zero singular values in $V_X^T$.

---


## 4. Parameter Savings & Memory Footprint Profiling

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 4.1 Memory & FLOP Formulas
When AASVD is deployed as a weight reconstruction mechanism (overwriting `module.weight.data` in-place), parameter memory remains identical to the original uncompressed model ($d_{\text{out}} \times d_{\text{in}}$). When deployed as a factorized linear layer ($U_r \operatorname{diag}(S_r) V_r^T$):

Specifically, regarding \textbf{Original Parameters}, $P_{\text{orig}} = d_{\text{out}} \cdot d_{\text{in}}$ Furthermore, regarding \textbf{Factorized Parameters}, $P_{\text{AASVD}} = r(d_{\text{out}} + d_{\text{in}} + 1)$ Concurrently, regarding \textbf{Inference FLOPs / Token}, $2 r (d_{\text{out}} + d_{\text{in}})$


### 4.2 Benchmark Footprint Table ($d_{\text{out}} = 4096, d_{\text{in}} = 4096$, Rank $r = 128$)

| Stage / Component | Operation | Matrix Dimensions | Tensor Element Count | Memory (FP16) |
|-------------------|-----------|-------------------|----------------------|---------------|
| **Calibration Activation** | Flatten Input | $128 \times 4096$ | 524,288 | 1.11 MB |
| **Anchor Basis $A_{\text{anchor}}$** | Activation Basis | $4096 \times 128$ | 524,288 | 1.12 MB |
| **Anchored Weight $W_{\text{anchored}}$** | $W P_{\text{anchor}}$ | $4096 \times 4096$ | 16,777,216 | 33.55 MB |
| **Factorized $U_r$** | Left Basis | $4096 \times 128$ | 524,288 | 1.13 MB |
| **Factorized $V_r^T$** | Right Basis | $128 \times 4096$ | 524,288 | 1.13 MB |
| **Total Factorized Layer** | $U_r, S_r, V_r^T$ | -- | **1,048,704** | **2.10 MB (93.7% savings)** |

---


## 5. Precision Loss, Perplexity & Sensitivity Bounds

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 5.1 Error Bound relative to Activation Space
Let $X$ be the activation tensor with principal covariance $C_X = X^T X$. The output error norm $\|E_{\text{out}}\|_F^2 = \|X W^T - X \hat{W}^T\|_F^2$ under AASVD satisfies:

\[
\|E_{\text{out}}\|_F^2 \le \|X (I - P_{\text{anchor}}) W^T\|_F^2 + \|X P_{\text{anchor}} (W - \hat{W}_{\text{AASVD}})^T\|_F^2
\]

By selecting $A_{\text{anchor}}$ as the top-$r$ eigenvectors of $C_X$, the term $\|X (I - P_{\text{anchor}})\|_F^2 = \sum_{i=r+1}^{d_{\text{in}}} \lambda_i(C_X)$ is minimized. Consequently, AASVD guarantees lower activation output distortion than standard static SVD:

\[
\|X W^T - X \hat{W}_{\text{AASVD}}^T\|_F^2 \le \|X W^T - X \hat{W}_{\text{SVD}}^T\|_F^2
\]

### 5.2 Failure Modes
1. **Out-of-Distribution Calibration Drift**: If calibration text sequences do not match deployment token distributions, $A_{\text{anchor}}$ isolates irrelevant activation subspaces, degrading real-world inference accuracy.
2. **Small Calibration Sample Size**: Calibrating with short sequence lengths ($K < d_{\text{in}}$) causes rank deficiency in $\bar{X}$, generating spurious zero singular values in $V_X^T$.

---
