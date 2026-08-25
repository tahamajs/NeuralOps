# 📐 NeuralOps Empirical Validation & Calibration Protocol

> **Document Type**: Scientific Validation Standard & Diagnostic Procedures  
> **Target Framework**: NeuralOps Research Framework (`Codes/framework/`)  
> **Compliance Standard**: NeurIPS / ICLR / ICML Reproducibility Checklist

---

## 1. Calibration Protocols for Data-Dependent Modules

Several compression engines and attention variants require empirical calibration data to compute activation statistics, covariance matrices, or Hessian curvature approximations.

### 1.1 Activation-Aware SVD (AASVD / AA-SVD)
- **Calibration Corpus**: 128 sequences of 512 tokens sampled from C4 / WikiText-103 or the target training split.
- **Covariance Computation**:
  $$\Sigma_X = rac{1}{B \cdot N} \sum_{b=1}^B \sum_{t=1}^N x_{b,t} x_{b,t}^	op + \lambda I$$
  where $\lambda = 10^{-5} \cdot rac{\operatorname{Tr}(\Sigma_X)}{D}$ is the Tikhonov damping coefficient ensuring non-singularity ($\kappa(\Sigma_X + \lambda I) < 10^5$).
- **Numerical Precision**: Intermediate covariance matrices MUST be accumulated in FP32 (`torch.float32`) to prevent overflow before Cholesky/SVD factorizations.

### 1.2 SliceGPT & Orthogonal Transformations
- **Basis Invariance**: Given block-diagonal or global rotation matrices $Q \in O(D)$ ($Q^	op Q = I$), compute:
  $$\widetilde{W} = W Q, \quad \widetilde{x} = Q^	op x$$
- **Slicing Criterion**: Principal signals are concentrated in highest-eigenvalue coordinate bases; columns with minimal variance are pruned without altering uncompressed intermediate activations.

### 1.3 SparseGPT & ADMM Pruners
- **Inverse Hessian Approximation**:
  $$H = 2 X X^	op + \lambda I \implies H^{-1}$$
  Damped Cholesky decomposition $H^{-1} = L L^	op$ enables stable column-wise recursive optimal brain surgery updates.

---

## 2. Invariant Verification Checklist

Before publishing results or submitting new architectures to the registry, verify the following:

- [ ] **Linear Layer `.weight` Introspection**: Every custom `nn.Module` replacing `nn.Linear` implements `@property def weight(self) -> torch.Tensor` for universal framework compatibility.
- [ ] **[CLS] Token Integrity in Token Pruning**: Saliency masks protect sequence index 0 ($S_0 \ge 10^6$) to prevent pooler collapse on sequence classification.
- [ ] **Dynamic Rank Lower Bound**: SVD rank truncation incorporates an explicit floor ($r \ge 16$) to prevent representation collapse in low-entropy layers.
- [ ] **Bounded Recurrence Dynamics**: In all SSM and linear attention modules, decay rates satisfy $\gamma_t = \sigma(\lambda_t) \in (0, 1)$ or real spectral eigenvalues $\operatorname{Re}(\lambda) < 0$.
- [ ] **Reproducible Seeds**: Evaluation scripts record seeds $s \in \{42, 1337, 2026\}$ and report standard deviations over minimum 3 seeds.
