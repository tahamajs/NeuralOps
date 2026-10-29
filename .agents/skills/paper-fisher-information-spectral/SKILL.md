---
name: paper-fisher-information-spectral
description: "Author mathematically rigorous sections on Fisher Information Matrix (FIM) curvature, empirical activation covariance, natural gradient metrics, and Fisher-weighted SVD for research papers"
version: "1.0.0"
tags: ["neuralops", "paper", "theory", "math", "fisher-information", "fim", "curvature", "svd"]
---

# Skill: Fisher Information Matrix Curvature, Natural Gradients & Spectral SVD Weighting

## Overview & Scope
Use this skill when drafting, verifying, deriving, or polishing mathematically rigorous theoretical sections, formal proofs, theorem environments, and algorithmic descriptions concerning **Fisher Information Matrix (FIM) curvature, Riemannian information geometry, Kronecker-factored approximate curvature (KFAC), and Fisher-weighted Singular Value Decomposition (FWSVD / GFWS)** for top-tier machine learning conferences (NeurIPS, ICLR, ICML, MLSys) and doctoral dissertations.

In deep neural networks—particularly autoregressive Transformers—parameter perturbations induce highly anisotropic deformations on the model's predictive distribution. Standard Euclidean loss surrogates (such as Frobenius-norm matrix truncation via naive Eckart-Young-Mirsky SVD) treat all parameter directions as isotropic, failing catastrophically in the presence of heavy-tailed activation outliers. Grounding low-rank compression in information geometry requires:
1. **The Statistical Riemannian Metric**: Modeling parameter space as a smooth Riemannian manifold whose metric tensor is the expected Fisher Information Matrix $F(\theta)$.
2. **Kronecker-Factored Curvature (KFAC)**: Decoupling layer-wise parameter Fisher tensors into Kronecker products of input activation second-moments and pre-activation gradient covariances ($F_W \approx A \otimes G$).
3. **Fisher-Weighted SVD Factorization**: Formulating model compression as functional divergence minimization on the Riemannian manifold: $\min_{\operatorname{rank}(\widehat{W}) \le r} \operatorname{Tr}\left( (W - \widehat{W}) F_W (W - \widehat{W})^\top \right)$.
4. **Cholesky Covariance Pullback Isometry**: Mapping anisotropic Riemannian curvature to an isometric Euclidean space via triangular Cholesky factorization $\Sigma_X = L L^\top$, guaranteeing global optimality and stable triangular solve without explicit matrix inversion.

---

## 1. Information Geometry & Foundations of the Fisher Information Matrix

```
═════════════════════════════════════════════════════════════════════════════════════════
          STATISTICAL RIEMANNIAN MANIFOLD & FISHER CURVATURE METRIC
═════════════════════════════════════════════════════════════════════════════════════════

                 Euclidean Parameter Space: Θ ⊆ R^P
        ┌─────────────────────────────────────────────────────────────┐
        │  Naive Euclidean Metric:  ds^2 = ||dθ||_2^2                 │
        │  (Assumes isotropic parameter sensitivity)                  │
        └──────────────────────────────┬──────────────────────────────┘
                                       │ Information Geometric Pullback
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │        Statistical Probability Manifold: M = {p_θ}          │
        │                                                             │
        │   Riemannian Metric Tensor:                                 │
        │     ds^2 = dθ^T F(θ) dθ = 2 * D_KL(p_θ || p_{θ+dθ})         │
        │                                                             │
        │   Natural Gradient Direction:                               │
        │     ~∇_θ L = F(θ)^{-1} ∇_θ L                                │
        │     (Steepest descent direction in distribution space)      │
        │                                                             │
        │   Layer-wise Kronecker Factorization (KFAC):                │
        │     F_W ≈ A ⊗ G                                             │
        │     A = E[x x^T]  (Activation Covariance)                   │
        │     G = E[g g^T]  (Output Gradient Covariance)              │
        └─────────────────────────────────────────────────────────────┘
═════════════════════════════════════════════════════════════════════════════════════════
```

### 1.1 The Statistical Manifold and the Fisher Metric Tensor
Let $p_\theta(y|x)$ denote a parametric conditional probability density over outputs $y \in \mathcal{Y}$ given inputs $x \in \mathcal{X}$, parameterized by vector $\theta \in \Theta \subseteq \mathbb{R}^P$. The family of distributions $\mathcal{M} = \{ p_\theta : \theta \in \Theta \}$ forms a statistical manifold.

The sensitivity of the model log-likelihood to infinitesimal variations in $\theta$ is captured by the **score function** $s_\theta(x, y) \in \mathbb{R}^P$:
$$s_\theta(x, y) \triangleq \nabla_\theta \log p_\theta(y|x)$$

Under standard regularity conditions (differentiability under the integral sign):
$$\mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \right] = \int_{\mathcal{Y}} \frac{\nabla_\theta p_\theta(y|x)}{p_\theta(y|x)} p_\theta(y|x) \, dy = \nabla_\theta \int_{\mathcal{Y}} p_\theta(y|x) \, dy = \nabla_\theta 1 = \mathbf{0}$$

#### Definition (True Fisher Information Matrix):
The **Fisher Information Matrix (FIM)** $F(\theta) \in \mathbb{R}^{P \times P}$ is the covariance of the score function under the joint data-model distribution:
$$F(\theta) \triangleq \mathbb{E}_{x \sim p_{\text{data}}(x)} \mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right]$$

The FIM is symmetric and positive semi-definite ($F(\theta) \succeq 0$). When $p_\theta$ is identifiable, $F(\theta) \succ 0$, defining a Riemannian metric tensor on $\Theta$. The Riemannian distance between two infinitesimally close models $p_\theta$ and $p_{\theta + d\theta}$ is given by the metric line element:
$$ds^2 = \langle d\theta, d\theta \rangle_{F(\theta)} = d\theta^\top F(\theta) d\theta$$

---

### 1.2 Tripartite Divergence: True Fisher vs. Empirical Fisher vs. Hessian of KL Divergence
A common point of ambiguity in machine learning literature is the distinction between the True Fisher, the Empirical Fisher, and the loss Hessian. Rigorous paper sections must sharply distinguish these quantities:

#### 1. True Fisher Information Matrix ($F_{\text{true}}$):
Defined with expectation over model predictions $y \sim p_\theta(y|x)$:
$$F_{\text{true}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right]$$
- **Invariance**: Invariant under reparameterization of $\theta$.
- **Connection to KL Divergence**: Exactly equals the Hessian of the expected Kullback-Leibler (KL) divergence at the current parameter value:
  $$\left. \nabla_{\Delta\theta}^2 \mathbb{E}_{x \sim p(x)} \left[ \mathcal{D}_{\mathrm{KL}}(p_\theta(\cdot|x) \,\|\, p_{\theta + \Delta\theta}(\cdot|x)) \right] \right|_{\Delta\theta = 0} = F_{\text{true}}(\theta)$$

#### 2. Empirical Fisher Matrix ($\bar{F}_{\text{emp}}$):
Replaces the expectation over model predictions with empirical labels $y \sim p_{\text{data}}(y|x)$:
$$\bar{F}_{\text{emp}}(\theta) \triangleq \mathbb{E}_{(x, y) \sim p_{\text{data}}(x, y)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right]$$
- **Theoretical Flaw**: $\bar{F}_{\text{emp}}$ is not a valid Riemannian metric for the model manifold because $\mathbb{E}_{(x, y) \sim p_{\text{data}}}[s_\theta(x, y)] \neq \mathbf{0}$ when the model has non-zero training gradient.
- **Confounding Effect**: $\bar{F}_{\text{emp}}$ confounds curvature with gradient variance. If the model is close to convergence ($\nabla_\theta \mathcal{L} \approx \mathbf{0}$), $\bar{F}_{\text{emp}} \approx F_{\text{true}}$ under well-specified models ($p_{\text{data}}(y|x) \approx p_\theta(y|x)$).
- **Practical Advantage**: Does not require sampling from model conditional distributions; computationally accessible from backpropagation minibatches.

#### 3. Hessian of Negative Log-Likelihood ($H$):
The second-derivative matrix of the empirical risk $\mathcal{L}(\theta) = -\mathbb{E}_{(x, y)} [\log p_\theta(y|x)]$:
$$H(\theta) = -\mathbb{E}_{(x, y) \sim p_{\text{data}}} \left[ \nabla_\theta^2 \log p_\theta(y|x) \right]$$
By differentiating the identity $\int p_\theta(y|x) \, dy = 1$ twice:
$$\mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta^2 \log p_\theta(y|x) \right] = - \mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right] = -F(\theta; x)$$
Consequently:
$$H(\theta) = F_{\text{true}}(\theta) - \mathbb{E}_{x \sim p(x)} \int_{\mathcal{Y}} \left( p_{\text{data}}(y|x) - p_\theta(y|x) \right) \nabla_\theta^2 \log p_\theta(y|x) \, dy$$
The second term vanishes when the model matches the data-generating distribution ($p_\theta = p_{\text{data}}$) or when the model function is linear in parameters. In overparameterized deep networks near local minima, $H(\theta) \approx F_{\text{true}}(\theta)$.

---

### 1.3 Amari's Natural Gradient
In Euclidean space, gradient descent updates parameters along the direction of steepest loss decrease within an $\ell_2$ ball: $\Delta\theta_{\text{euc}} \propto -\nabla_\theta \mathcal{L}$.

However, the $\ell_2$ metric $\|\Delta\theta\|_2^2$ depends entirely on arbitrary coordinate choices and fails to account for predictive change. Amari's **Natural Gradient** seeks the steepest descent direction constrained within an infinitesimal KL-divergence ball:
$$\Delta\theta^* = \arg\min_{\Delta\theta} \left\{ \mathcal{L}(\theta + \Delta\theta) \right\} \quad \text{s.t.} \quad \mathcal{D}_{\mathrm{KL}}(p_\theta \,\|\, p_{\theta + \Delta\theta}) \le \epsilon$$

Using the second-order Taylor expansion $\mathcal{D}_{\mathrm{KL}}(p_\theta \,\|\, p_{\theta + \Delta\theta}) \approx \frac{1}{2} \Delta\theta^\top F(\theta) \Delta\theta$, Lagrangian optimization yields:
$$\widetilde{\nabla}_\theta \mathcal{L}(\theta) = F(\theta)^{-1} \nabla_\theta \mathcal{L}(\theta)$$

Natural gradient descent updates parameters as:
$$\theta_{t+1} = \theta_t - \eta \, F(\theta_t)^{-1} \nabla_\theta \mathcal{L}(\theta_t)$$
This update is asymptotically invariant to smooth, invertible reparameterizations of $\Theta$.

---

## 2. Kronecker-Factored Approximate Curvature (KFAC) for Linear Layers

For large neural networks with millions or billions of parameters ($P \ge 10^7$), computing, storing, and inverting the full Fisher matrix $F(\theta) \in \mathbb{R}^{P \times P}$ (requiring $\mathcal{O}(P^2)$ memory and $\mathcal{O}(P^3)$ computation) is intractable. KFAC provides an analytical block-separable approximation.

### 2.1 Matrix-Variate Parameterization
Consider a standard dense linear projection layer:
$$s = W x \in \mathbb{R}^{d_{\text{out}}}, \quad a = \phi(s) \in \mathbb{R}^{d_{\text{out}}}$$
where $x \in \mathbb{R}^{d_{\text{in}}}$ is the input activation vector, $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the weight matrix, and $s$ is the pre-activation vector.

Let $g \triangleq \nabla_s \log p_\theta(y|x) \in \mathbb{R}^{d_{\text{out}}}$ denote the gradient of the log-likelihood with respect to the layer's pre-activation vector. By the chain rule:
$$\frac{\partial \log p_\theta(y|x)}{\partial W_{i, j}} = \frac{\partial \log p_\theta(y|x)}{\partial s_i} \frac{\partial s_i}{\partial W_{i, j}} = g_i x_j \implies \nabla_W \log p_\theta(y|x) = g x^\top$$

Using the column-stacking vectorization operator $\operatorname{vec}(\cdot)$ and the Kronecker identity $\operatorname{vec}(u v^\top) = v \otimes u$:
$$\operatorname{vec}\left( \nabla_W \log p_\theta(y|x) \right) = \operatorname{vec}(g x^\top) = x \otimes g \in \mathbb{R}^{d_{\text{in}} d_{\text{out}}}$$

---

### 2.2 Derivation of the KFAC Approximation
The Fisher Information Matrix block corresponding to the weights $W$ of this layer is:
$$F_W = \mathbb{E}\left[ \operatorname{vec}(\nabla_W \log p_\theta) \operatorname{vec}(\nabla_W \log p_\theta)^\top \right] = \mathbb{E}\left[ (x \otimes g) (x \otimes g)^\top \right]$$

Applying the mixed-product property of the Kronecker product $(P \otimes Q)(R \otimes S) = (P R) \otimes (Q S)$:
$$F_W = \mathbb{E}\left[ (x x^\top) \otimes (g g^\top) \right] \in \mathbb{R}^{d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}}$$

#### The KFAC Expectation Factorization Hypothesis:
KFAC approximates the expectation of the Kronecker product as the Kronecker product of expectations:
$$F_W \approx \mathbb{E}[x x^\top] \otimes \mathbb{E}[g g^\top] \triangleq A \otimes G$$
where:
- $A \triangleq \mathbb{E}_{x}[x x^\top] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$: The input activation second-moment (covariance) matrix.
- $G \triangleq \mathbb{E}_{g}[g g^\top] \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$: The pre-activation output gradient second-moment matrix.

```
═════════════════════════════════════════════════════════════════════════════════
             KFAC BLOCK STRUCTURE FOR A LINEAR LAYER W in R^{d_out x d_in}
═════════════════════════════════════════════════════════════════════════════════

   Exact Fisher Tensor:                       KFAC Decoupled Factorization:
   F_W = E[ (x x^T) ⊗ (g g^T) ]               F_W ≈ A ⊗ G
   Dimension: (d_in * d_out)^2                A = E[x x^T] in R^{d_in x d_in}
                                              G = E[g g^T] in R^{d_out x d_out}

   ┌───────────────────────────────────┐      ┌─────────────┐       ┌─────────────┐
   │                                   │      │             │       │             │
   │           Full F_W                │      │      A      │   ⊗   │      G      │
   │     (Dense Gigantic Matrix)       │  ≈   │ (Activation │       │ (Gradient   │
   │                                   │      │ Covariance) │       │ Covariance) │
   │                                   │      │             │       │             │
   └───────────────────────────────────┘      └─────────────┘       └─────────────┘
   Storage: O(d_in^2 * d_out^2)               Storage: O(d_in^2 + d_out^2)
   Inversion: O(d_in^3 * d_out^3)             Inversion: O(d_in^3 + d_out^3)
═════════════════════════════════════════════════════════════════════════════════
```

---

### 2.3 Curvature Inversion & Kronecker Algebra
The Kronecker product structure provides immense computational reductions via the inverse property $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$:
$$F_W^{-1} \approx (A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$$

To compute the natural gradient direction on matrix weights $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$\operatorname{vec}(\widetilde{\nabla}_W \mathcal{L}) = (A \otimes G)^{-1} \operatorname{vec}(\nabla_W \mathcal{L}) = (A^{-1} \otimes G^{-1}) \operatorname{vec}(\nabla_W \mathcal{L})$$

Using the identity $(P \otimes Q) \operatorname{vec}(M) = \operatorname{vec}(Q M P^\top)$:
$$\widetilde{\nabla}_W \mathcal{L} = G^{-1} (\nabla_W \mathcal{L}) A^{-1}$$

- **Complexity Reduction**: Inverting $F_W \in \mathbb{R}^{d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}}$ takes $\mathcal{O}(d_{\text{in}}^3 d_{\text{out}}^3)$ operations. With KFAC, inverting $A$ and $G$ separately requires only $\mathcal{O}(d_{\text{in}}^3 + d_{\text{out}}^3)$ operations.
- **Memory Footprint**: Drops from $\mathcal{O}(d_{\text{in}}^2 d_{\text{out}}^2)$ to $\mathcal{O}(d_{\text{in}}^2 + d_{\text{out}}^2)$.

---

## 3. Fisher-Weighted Low-Rank SVD (FWSVD & GFWS)

### 3.1 Objective Formulation: Functional Divergence Minimization
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be a dense projection matrix in a pre-trained model. We seek a low-rank factorized surrogate $\widehat{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with $\operatorname{rank}(\widehat{W}) \le r \ll \min(d_{\text{in}}, d_{\text{out}})$.

Under a perturbation $\Delta W = W - \widehat{W}$, the second-order expansion of the KL divergence between the original model and the compressed model is:
$$\mathcal{D}_{\mathrm{KL}}(P_W \,\|\, P_{\widehat{W}}) \approx \frac{1}{2} \operatorname{vec}(\Delta W)^\top F_W \operatorname{vec}(\Delta W)$$

Substituting the KFAC approximation $F_W \approx A \otimes G$:
$$\begin{aligned}
\operatorname{vec}(\Delta W)^\top (A \otimes G) \operatorname{vec}(\Delta W)
&= \operatorname{vec}(\Delta W)^\top \operatorname{vec}\left( G \Delta W A^\top \right) \\
&= \operatorname{Tr}\left( \Delta W^\top G \Delta W A \right) \\
&= \operatorname{Tr}\left( (W - \widehat{W}) A (W - \widehat{W})^\top G \right)
\end{aligned}$$

#### The Bilateral Fisher SVD Optimization Problem:
$$\min_{\operatorname{rank}(\widehat{W}) \le r} \mathcal{J}_{\text{Fisher}}(\widehat{W}) \triangleq \operatorname{Tr}\left( (W - \widehat{W}) A (W - \widehat{W})^\top G \right)$$

#### Isotropic Gradient Output Simplification:
In post-training compression regimes where backward gradients are unavailable or where output gradients are assumed isotropic ($G \approx \gamma I_{d_{\text{out}}}$):
$$\mathcal{J}_{\text{Fisher}}(\widehat{W}) = \gamma \operatorname{Tr}\left( (W - \widehat{W}) A (W - \widehat{W})^\top \right) = \frac{\gamma}{M} \|(W - \widehat{W}) X\|_F^2$$
where $X \in \mathbb{R}^{d_{\text{in}} \times M}$ is the calibration activation matrix ($A = \frac{1}{M} X X^\top$).

---

### 3.2 Diagonal vs. Full Covariance Curvature
Existing compression engines exhibit distinct approximations of $F_W$:

| Method | Curvature Structure $F_W$ | Computational Cost | Cross-Channel Decorrelation |
| :--- | :--- | :--- | :--- |
| **Naive SVD** | $I_{d_{\text{in}}} \otimes I_{d_{\text{out}}}$ (Isotropic Euclidean) | $\mathcal{O}(d^3)$ | None (Ignores activations) |
| **FWSVD** | $\operatorname{diag}(A) \otimes I_{d_{\text{out}}}$ (Diagonal Input Fisher) | $\mathcal{O}(d \cdot M + d^3)$ | None (Scales columns independently) |
| **GFWS / SVD-LLM v2** | $A \otimes I_{d_{\text{out}}}$ (Full Input Covariance) | $\mathcal{O}(d^2 \cdot M + d^3)$ | Full (Exact Cholesky pullback) |
| **Bilateral KFAC** | $A \otimes G$ (Full Input + Output Covariance) | $\mathcal{O}((d_{\text{in}}^2 + d_{\text{out}}^2) M + d^3)$ | Full bilateral decorrelation |

#### Analytical Comparison: Why Diagonal Fisher Fails Under Outlier Collinearity:
In `FWSVDCompressor`, the Fisher weights are approximated as diagonal elements:
$$F_i = \mathbb{E}[x_i^2] = A_{i, i}$$
The weight matrix is scaled by $W_{\text{weighted}} = W \operatorname{diag}(\sqrt{F_1}, \dots, \sqrt{F_{d_{\text{in}}}})$, decomposed via SVD, and unweighted:
$$V_r \leftarrow V_r \operatorname{diag}(1/\sqrt{F_1}, \dots, 1/\sqrt{F_{d_{\text{in}}}})$$

While this accounts for individual feature variances, it completely neglects the **off-diagonal covariance entries** $A_{i, j} = \mathbb{E}[x_i x_j]$ ($i \neq j$). In Transformers, activation outliers do not occur in isolated axes; they lie along correlated low-dimensional linear manifolds.

---

## 4. Activation Covariance Whitening & Cholesky Pullback Isometry

To solve the functional Fisher minimization problem with full input covariance, we establish the Cholesky Pullback Isometry.

```
═════════════════════════════════════════════════════════════════════════════════════════
                   THE CHOLESKY PULLBACK ISOMETRY ON WEIGHT SPACE
═════════════════════════════════════════════════════════════════════════════════════════

        Original Anisotropic Metric Space:            Whitened Isotropic Euclidean Space:
          ||ΔW||_FIM^2 = Tr(ΔW Σ_X ΔW^T)                ||ΔW_whitened||_F^2 = ||M - K||_F^2
        ┌───────────────────────────────┐             ┌───────────────────────────────┐
        │  Curvature: Σ_X = L L^T       │             │  Curvature: I_d               │
        │  κ(Σ_X) >> 10^5               │  L^T (·)    │  κ(I) = 1                     │
        │  Ellipsoidal error contours   ├────────────►│  Spherical error contours     │
        │  Naive SVD cuts across axes   │  Pushforward│  Eckart-Young-Mirsky optimal! │
        └───────────────────────────────┘             └───────────────┬───────────────┘
                        ▲                                             │ Truncated SVD
                        │ Pullback: L^{-T} (·)                        │ on M = L^T W
                        │ (Via cuBLAS trsm)                           ▼
        ┌───────────────┴───────────────┐             ┌───────────────────────────────┐
        │  Optimal Compressed Weights:  │             │  Optimal Whitened Low-Rank:   │
        │  W^* = L^{-T} U_r Σ_r V_r^T   │◄────────────┤  K^* = U_r Σ_r V_r^T          │
        └───────────────────────────────┘             └───────────────────────────────┘
═════════════════════════════════════════════════════════════════════════════════════════
```

### 4.1 Regularized Empirical Covariance & Tikhonov Conditioning
Given calibration activations $X \in \mathbb{R}^{M \times d_{\text{in}}}$, the empirical second-moment matrix is:
$$\mathbf{S} = \frac{1}{M} X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$
When $M < d_{\text{in}}$ or when activation features exhibit extreme collinearity, $\mathbf{S}$ is singular or ill-conditioned ($\kappa(\mathbf{S}) \gg 10^6$). We apply Tikhonov regularization (damping):
$$\mathbf{S}_{\text{reg}} = \mathbf{S} + \lambda \mathbf{I}_{d_{\text{in}}}, \quad \lambda = \epsilon \cdot \frac{\operatorname{Tr}(\mathbf{S})}{d_{\text{in}}}$$
where $\epsilon \in [10^{-6}, 10^{-4}]$ is a dimensionless damping parameter.

#### Condition Number Guarantee:
$$\kappa(\mathbf{S}_{\text{reg}}) = \frac{\lambda_{\max}(\mathbf{S}) + \lambda}{\lambda_{\min}(\mathbf{S}) + \lambda} \le 1 + \frac{\lambda_{\max}(\mathbf{S})}{\lambda} \le 1 + \frac{d_{\text{in}}}{\epsilon}$$
This prevents metric degeneracy and ensures strict positive-definiteness: $\mathbf{S}_{\text{reg}} \succ 0$.

---

### 4.2 The Cholesky Pullback Isometry
Since $\mathbf{S}_{\text{reg}} \succ 0$, there exists a unique lower-triangular Cholesky factor $L \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ with positive diagonal entries such that:
$$\mathbf{S}_{\text{reg}} = L L^\top$$

Let $\Delta W = W - \widehat{W}$. The Fisher-induced error metric evaluates to:
$$\begin{aligned}
\|\Delta W\|_{\mathbf{S}_{\text{reg}}}^2
&= \operatorname{Tr}\left( \Delta W^\top \mathbf{S}_{\text{reg}} \Delta W \right) \\
&= \operatorname{Tr}\left( \Delta W^\top (L L^\top) \Delta W \right) \\
&= \operatorname{Tr}\left( (L^\top \Delta W)^\top (L^\top \Delta W) \right) \\
&= \|L^\top \Delta W\|_F^2
\end{aligned}$$

#### Transformation to Whitened Space:
Define the whitened target matrix $M$ and low-rank approximation $K$:
$$M \triangleq L^\top W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}, \quad K \triangleq L^\top \widehat{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$$

Because $L^\top$ is invertible, $\operatorname{rank}(\widehat{W}) = \operatorname{rank}(K)$. The Riemannian optimization problem transforms into an unweighted Frobenius-norm minimization problem:
$$\min_{\operatorname{rank}(\widehat{W}) \le r} \|L^\top (W - \widehat{W})\|_F^2 \iff \min_{\operatorname{rank}(K) \le r} \|M - K\|_F^2$$

By the classical Eckart-Young-Mirsky theorem, the unique global minimizer $K^*$ is obtained by computing the truncated SVD of $M$:
$$M = L^\top W = U \Sigma V^\top \implies K^* = U_r \Sigma_r V_r^\top$$

#### Exact Metric Pullback:
Pulling the optimal solution back to the original parameter space:
$$\widehat{W}^* = L^{-\top} K^* = L^{-\top} U_r \Sigma_r V_r^\top$$

---

### 4.3 Computational Efficiency: Triangular Solve vs. Explicit Inversion
In production implementations, explicit matrix inversion $L^{-1}$ must **never** be performed. Explicit inversion squares condition number sensitivity and introduces rounding instabilities.

Instead, factorized matrices are formed via **triangular back-substitution** (`cuBLAS trsm` / `torch.linalg.solve_triangular`):
$$L^\top Z = U_r \implies Z = L^{-\top} U_r$$
$$\widehat{W}^* = Z \Sigma_r V_r^\top = (Z \sqrt{\Sigma_r}) (\sqrt{\Sigma_r} V_r^\top)$$

- **Triangular Factorization Cost**: $\frac{1}{3} d_{\text{in}}^3$ FLOPs (compared to $2 d_{\text{in}}^3$ for symmetric eigendecomposition $\Sigma^{1/2}$).
- **Condition Number**: $\kappa(L) = \sqrt{\kappa(\mathbf{S}_{\text{reg}})}$, halving the logarithmic condition index.

---

## 5. Formal Theorems & Analytic Proofs

### Lemma 1: Expectation of Score and KL Hessian Equivalence
Let $p_\theta(y|x)$ be twice continuously differentiable with respect to $\theta$. Then:
1. $\mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \right] = \mathbf{0}$.
2. $\mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta^2 \log p_\theta(y|x) \right] = - \mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right]$.

```latex
\begin{proof}
For statement (1), since $\int p_\theta(y|x) dy = 1$:
\[
\nabla_\theta \int p_\theta(y|x) dy = \int \nabla_\theta p_\theta(y|x) dy = \mathbf{0}.
\]
Using $\nabla_\theta \log p_\theta(y|x) = \frac{\nabla_\theta p_\theta(y|x)}{p_\theta(y|x)}$:
\[
\mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta \log p_\theta(y|x) \right]
= \int \frac{\nabla_\theta p_\theta(y|x)}{p_\theta(y|x)} p_\theta(y|x) dy
= \int \nabla_\theta p_\theta(y|x) dy = \mathbf{0}.
\]

For statement (2), differentiate the identity $\nabla_\theta \log p_\theta = \frac{\nabla_\theta p_\theta}{p_\theta}$ again:
\[
\nabla_\theta^2 \log p_\theta(y|x) = \frac{\nabla_\theta^2 p_\theta(y|x)}{p_\theta(y|x)} - \frac{\nabla_\theta p_\theta(y|x) \nabla_\theta p_\theta(y|x)^\top}{p_\theta(y|x)^2}.
\]
Taking expectations on both sides with respect to $y \sim p_\theta(y|x)$:
\[
\begin{aligned}
\mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta^2 \log p_\theta(y|x) \right]
&= \int \frac{\nabla_\theta^2 p_\theta(y|x)}{p_\theta(y|x)} p_\theta(y|x) dy - \int \frac{\nabla_\theta p_\theta \nabla_\theta p_\theta^\top}{p_\theta^2} p_\theta dy \\
&= \nabla_\theta^2 \int p_\theta(y|x) dy - \mathbb{E}_{y \sim p_\theta} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right] \\
&= \mathbf{0} - F(\theta; x) = -F(\theta; x).
\end{aligned}
\]
\end{proof}
```

---

### Theorem 1: Global Optimality of Cholesky-Whitened Fisher Truncation
Let $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and calibration activations $X \in \mathbb{R}^{M \times d_{\text{in}}}$ with regularized second-moment matrix $\mathbf{C} = \frac{1}{M} X^\top X + \epsilon I \succ 0$. Let $\mathbf{C} = L L^\top$ be the Cholesky factorization of $\mathbf{C}$. Then:

$$\widehat{W}^* = \arg\min_{\operatorname{rank}(\widehat{W}) \le r} \operatorname{Tr}\left( (W - \widehat{W})^\top \mathbf{C} (W - \widehat{W}) \right)$$
is uniquely given by:
$$\widehat{W}^* = L^{-\top} U_r \Sigma_r V_r^\top$$
where $L^\top W = U \Sigma V^\top$ is the singular value decomposition of the whitened weight matrix, and $U_r, \Sigma_r, V_r$ denote the top-$r$ truncated components. The minimum error is:
$$\min_{\operatorname{rank}(\widehat{W}) \le r} \mathcal{J}(\widehat{W}) = \sum_{i=r+1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_i^2(L^\top W)$$

```latex
\begin{proof}
We proceed in five analytical steps:

\paragraph{Step 1: Metric Pullback Formulation.}
By algebraic expansion of the objective function:
\[
\mathcal{J}(\widehat{W}) = \Tr\left( (W - \widehat{W})^\top \mathbf{C} (W - \widehat{W}) \right)
= \Tr\left( (W - \widehat{W})^\top L L^\top (W - \widehat{W}) \right).
\]
Using cyclic trace invariance $\Tr(Z^\top Z) = \|Z\|_F^2$ where $Z = L^\top (W - \widehat{W})$:
\[
\mathcal{J}(\widehat{W}) = \|L^\top W - L^\top \widehat{W}\|_F^2.
\]

\paragraph{Step 2: Bijective Change of Variables.}
Define $M \triangleq L^\top W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and $K \triangleq L^\top \widehat{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.
Since $\mathbf{C} \succ 0$, $L$ is strictly non-singular with non-zero diagonal entries. Thus $L^\top$ is invertible, establishing a rank-preserving bijection:
\[
\operatorname{rank}(K) = \operatorname{rank}(L^\top \widehat{W}) = \operatorname{rank}(\widehat{W}) \le r.
\]
The optimization problem is equivalent to:
\[
K^* = \arg\min_{\operatorname{rank}(K) \le r} \|M - K\|_F^2.
\]

\paragraph{Step 3: Eckart-Young-Mirsky Optimality in Whitened Space.}
By the classical Eckart-Young-Mirsky theorem, the unique best rank-$r$ approximation of $M$ in Frobenius norm is its truncated SVD:
\[
M = L^\top W = \sum_{i=1}^p \sigma_i u_i v_i^\top \implies K^* = \sum_{i=1}^r \sigma_i u_i v_i^\top = U_r \Sigma_r V_r^\top.
\]
The residual error achieved by $K^*$ is:
\[
\|M - K^*\|_F^2 = \sum_{i=r+1}^p \sigma_i^2(M) = \sum_{i=r+1}^p \sigma_i^2(L^\top W).
\]

\paragraph{Step 4: Pullback to Parameter Space.}
Inverting the bijective transformation $K = L^\top \widehat{W}$:
\[
\widehat{W}^* = L^{-\top} K^* = L^{-\top} U_r \Sigma_r V_r^\top.
\]
Because the transformation is bijective and $K^*$ is the unique global minimizer in whitened space, $\widehat{W}^*$ is the unique global minimizer in the original parameter space.

\paragraph{Step 5: Strict Attainment of Pareto Frontier.}
For any rank budget $r \in \{1, \dots, p\}$, the minimal achievable error $\mathcal{E}^*(r) = \sum_{i=r+1}^p \sigma_i^2(L^\top W)$ forms a strictly monotonically non-increasing sequence with respect to $r$. Any other rank-$r$ parameterization produces an error strictly greater than or equal to $\mathcal{E}^*(r)$, completing the proof.
\end{proof}
```

---

### Theorem 2: Excess Output Error Bound of Naive SVD vs. Fisher-Whitened SVD
Let $\widehat{W}_{\text{naive}} = \sum_{k=1}^r \sigma_k(W) u_k v_k^\top$ be the naive (unweighted) SVD truncation of $W$. Let $\widehat{W}^*$ be the Cholesky-whitened solution from Theorem 1.

The **Excess Output Error** $\Delta \mathcal{E} \triangleq \mathcal{J}(\widehat{W}_{\text{naive}}) - \mathcal{J}(\widehat{W}^*)$ satisfies:
$$\Delta \mathcal{E} = \sum_{k=r+1}^{d_{\text{in}}} \sigma_k^2(W) \left( v_k^\top \mathbf{C} v_k \right) - \sum_{i=r+1}^{p} \sigma_i^2(L^\top W) \ge 0$$
Equality holds if and only if $\mathbf{C} = \alpha I_{d_{\text{in}}}$ (isotropic activation geometry). Furthermore, if $\mathbf{C}$ contains outlier directions with condition number $\kappa(\mathbf{C}) \gg 1$, the excess error scales with $\lambda_{\max}(\mathbf{C})$:
$$\Delta \mathcal{E} \ge \sigma_{r+1}^2(W) \lambda_{\max}(\mathbf{C}) \cos^2\angle(v_{r+1}, q_{\max}) - \sigma_{r+1}^2(L^\top W)$$
where $q_{\max}$ is the dominant eigenvector of $\mathbf{C}$.

```latex
\begin{proof}
The error of the naive SVD truncation under the metric $\mathbf{C}$ is:
\[
\begin{aligned}
\mathcal{J}(\widehat{W}_{\text{naive}})
&= \Tr\left( (W - \widehat{W}_{\text{naive}})^\top \mathbf{C} (W - \widehat{W}_{\text{naive}}) \right) \\
&= \Tr\left( \left(\sum_{k=r+1}^p \sigma_k(W) u_k v_k^\top\right)^\top \mathbf{C} \left(\sum_{j=r+1}^p \sigma_j(W) u_j v_j^\top\right) \right) \\
&= \sum_{k=r+1}^p \sum_{j=r+1}^p \sigma_k(W) \sigma_j(W) \Tr(v_k u_k^\top u_j v_j^\top \mathbf{C}).
\end{aligned}
\]
Since the left singular vectors are orthonormal ($u_k^\top u_j = \delta_{k, j}$):
\[
\mathcal{J}(\widehat{W}_{\text{naive}}) = \sum_{k=r+1}^p \sigma_k^2(W) \left( v_k^\top \mathbf{C} v_k \right).
\]
Since $\operatorname{rank}(\widehat{W}_{\text{naive}}) \le r$ and $\widehat{W}^*$ is the unique global minimizer of $\mathcal{J}$ over all rank-$r$ matrices:
\[
\Delta \mathcal{E} = \mathcal{J}(\widehat{W}_{\text{naive}}) - \mathcal{J}(\widehat{W}^*) \ge 0.
\]
Substitute $\mathcal{J}(\widehat{W}^*) = \sum_{i=r+1}^p \sigma_i^2(L^\top W)$:
\[
\Delta \mathcal{E} = \sum_{k=r+1}^p \sigma_k^2(W) (v_k^\top \mathbf{C} v_k) - \sum_{i=r+1}^p \sigma_i^2(L^\top W) \ge 0.
\]
Equality holds if and only if $L^\top \widehat{W}_{\text{naive}} = K^*$, which requires the right-singular subspace of $W$ to coincide with the right-singular subspace of $L^\top W$ for all $W$. Since $\mathbf{C} = L L^\top$, this holds identically for all matrices $W$ if and only if $L = \sqrt{\alpha} I$, meaning $\mathbf{C} = \alpha I$ ($\kappa(\mathbf{C}) = 1$).

For an ill-conditioned covariance matrix with spectral decomposition $\mathbf{C} = \sum_{j=1}^{d_{\text{in}}} \lambda_j q_j q_j^\top$:
\[
v_k^\top \mathbf{C} v_k = \sum_{j=1}^{d_{\text{in}}} \lambda_j (v_k^\top q_j)^2 \ge \lambda_{\max}(\mathbf{C}) (v_k^\top q_{\max})^2.
\]
If a discarded singular vector $v_{r+1}$ has non-zero inner product with $q_{\max}$, this term is magnified by $\lambda_{\max}(\mathbf{C}) \gg 10^5$, establishing the catastrophic excess error bound.
\end{proof}
```

---

## 6. Publication-Grade AMS-LaTeX Templates

Use these ready-to-insert LaTeX blocks for academic papers and dissertation chapters.

### 6.1 Preamble & Theorem Environment Setup
```latex
% Preamble requirements
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{bm}
\usepackage{microtype}

% Theorem Definitions
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
```

### 6.2 Drop-in Paper Section: Fisher Curvature & Activation Whitening
```latex
\section{Information Geometry and Curvature-Aware Model Factorization}
\label{sec:fisher_whitening_theory}

Let $\mathcal{M} = \{ p_{\mathbf{W}} : \mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \}$ represent the statistical manifold of output distributions generated by a linear projection layer $\mathbf{y} = \mathbf{W} \mathbf{x}$. The Riemannian metric tensor on $\mathcal{M}$ is defined by the Fisher Information Matrix (FIM):
\begin{equation}
\mathbf{F}_{\mathbf{W}} \triangleq \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} \mathbb{E}_{\mathbf{y} \sim p_{\mathbf{W}}(\mathbf{y}|\mathbf{x})} \left[ \operatorname{vec}(\nabla_{\mathbf{W}} \log p_{\mathbf{W}}) \operatorname{vec}(\nabla_{\mathbf{W}} \log p_{\mathbf{W}})^\top \right].
\end{equation}

Under the Kronecker-factored approximate curvature (KFAC) framework \cite{martens2015optimizing}, the expectation decouples into:
\begin{equation}
\label{eq:kfac_decomposition}
\mathbf{F}_{\mathbf{W}} \approx \mathbf{A} \otimes \mathbf{G}, \quad \mathbf{A} = \mathbb{E}_{\mathbf{x}}[\mathbf{x} \mathbf{x}^\top], \quad \mathbf{G} = \mathbb{E}_{\mathbf{g}}[\mathbf{g} \mathbf{g}^\top],
\end{equation}
where $\mathbf{g} = \nabla_{\mathbf{s}} \log p_{\mathbf{W}}$ is the pre-activation gradient. Assuming isotropic output gradients $\mathbf{G} \approx \gamma \mathbf{I}_{d_{\text{out}}}$, the second-order Taylor expansion of the Kullback-Leibler divergence under low-rank parameter substitution $\widehat{\mathbf{W}}$ reduces to:
\begin{equation}
\mathcal{D}_{\mathrm{KL}}(p_{\mathbf{W}} \,\|\, p_{\widehat{\mathbf{W}}}) \approx \frac{\gamma}{2} \operatorname{Tr}\left( (\mathbf{W} - \widehat{\mathbf{W}}) \mathbf{\Sigma}_{\mathbf{x}} (\mathbf{W} - \widehat{\mathbf{W}})^\top \right),
\end{equation}
where $\mathbf{\Sigma}_{\mathbf{x}} = \frac{1}{M} \mathbf{X} \mathbf{X}^\top$ is the empirical input activation covariance.

\subsection{Cholesky Pullback Isometry}
Direct truncation via the classical Eckart-Young-Mirsky theorem minimizes Euclidean weight distance $\|\mathbf{W} - \widehat{\mathbf{W}}\|_F^2$, ignoring the severe anisotropy of $\mathbf{\Sigma}_{\mathbf{x}}$ ($\kappa(\mathbf{\Sigma}_{\mathbf{x}}) \gg 10^5$). To restore global optimality, we apply Tikhonov regularization $\mathbf{\Sigma}_{\text{reg}} = \mathbf{\Sigma}_{\mathbf{x}} + \lambda \mathbf{I}$ and compute its Cholesky factor $\mathbf{\Sigma}_{\text{reg}} = \mathbf{L} \mathbf{L}^\top$. 

Defining the whitened target $\mathbf{M} \triangleq \mathbf{L}^\top \mathbf{W}$, the Riemannian objective satisfies an isometric pullback:
\begin{equation}
\operatorname{Tr}\left( (\mathbf{W} - \widehat{\mathbf{W}}) \mathbf{\Sigma}_{\text{reg}} (\mathbf{W} - \widehat{\mathbf{W}})^\top \right) = \|\mathbf{L}^\top (\mathbf{W} - \widehat{\mathbf{W}})\|_F^2 = \|\mathbf{M} - \mathbf{L}^\top \widehat{\mathbf{W}}\|_F^2.
\end{equation}

By Theorem~\ref{thm:cholesky_optimality}, computing the truncated SVD $\mathbf{M} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^\top$ and pulling back via triangular back-substitution:
\begin{equation}
\widehat{\mathbf{W}}^* = \mathbf{L}^{-\top} \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^\top
\end{equation}
yields the unique, globally optimal rank-$r$ parameterization under the Fisher information metric.
```

---

## 7. NeuralOps Codebase Implementation Cross-Reference

To maintain consistency between manuscript mathematics and the NeuralOps codebase, refer to these canonical modules:

### 7.1 Architecture Mapping
- **`neuralops/compression/svd/fwsvd_compressor.py` (`FWSVDCompressor`)**:
  - Implements **Diagonal Fisher Information** weighting:
    $$F_i = \mathbb{E}[x_i^2] \implies W_{\text{weighted}} = W \operatorname{diag}(\sqrt{F})$$
  - Decomposes $W_{\text{weighted}} = U \Sigma V^\top$ and de-weights $V_r \leftarrow V_r \operatorname{diag}(1/\sqrt{F})$.
  - Best for memory-constrained regimes where storing full covariance $d_{\text{in}} \times d_{\text{in}}$ is prohibitive.

- **`neuralops/compression/svd/gfws_d_compressor.py` (`GFWSCompressor`)**:
  - Implements **Generalized Fisher & Covariance Whitening**:
    $$\mathbf{S} = \frac{1}{M} X^\top X, \quad \mathbf{S}_{\text{reg}} = \mathbf{S} + \lambda \mathbf{I}$$
  - Decomposes covariance via symmetric eigendecomposition $\mathbf{S}_{\text{reg}} = Q \Lambda Q^\top$, applies $\mathbf{S}^{1/2} = Q \Lambda^{1/2} Q^\top$, computes SVD on $W \mathbf{S}^{1/2}$, and unwhitens by $\mathbf{S}^{-1/2}$.

- **`neuralops/compression/svd/hnc_svd_continuation.py` (`HNC-SVD`)**:
  - Integrates **Hessian Null-space Constraints (HNC)**:
    Projects weight updates orthogonally into the null-space of the empirical Hessian/Fisher curvature, preventing degradation of pre-trained knowledge.

- **`academic/thesis/chapters/chapter02_theory_part2.tex` & `chapter03_methodology_part3a.tex`**:
  - Section 2.4 / Subsection 2.4.2: Mathematical foundations of Cholesky whitening and FIM pullback (`\ref{sec:cholesky_whitening}`).
  - Theorem 1 (`\ref{thm:svdllm_v2_optimality}` / `\ref{thm:cholesky_optimality}`): Formal statement and step-by-step analytical proof of Cholesky-whitened SVD optimality.

---

## 8. Verification & Paper-Readiness Checklist

Before submitting a theory section or paper draft incorporating Fisher information and spectral weighting, verify the following quality gates:

- [ ] **Expectation Distinction**: Explicitly stated whether expectations are taken over model predictions $y \sim p_\theta(y|x)$ (True Fisher) or empirical data $y \sim p_{\text{data}}(y|x)$ (Empirical Fisher).
- [ ] **KFAC Independence Assumption**: Noted that $F_W \approx A \otimes G$ assumes independence of activation covariance $A$ and pre-activation gradient covariance $G$.
- [ ] **Tikhonov Damping**: Included condition number bounds $\kappa(\mathbf{S}_{\text{reg}}) \le 1 + \lambda_{\max}/\lambda$ to prevent metric degeneracy.
- [ ] **Triangular Solve Stability**: Clarified that pullback $\widehat{W}^* = L^{-\top} K^*$ is solved via forward/back-substitution (`trsm`), avoiding explicit matrix inversion.
- [ ] **Notation Consistency**:
  - Weight matrix: $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (or $d_{\text{in}} \times d_{\text{out}}$), keeping transposes consistent.
  - Activation covariance: $A = \mathbb{E}[x x^\top]$ or $\mathbf{\Sigma}_X = \frac{1}{M} X^\top X$.
  - Fisher matrix: $F(\theta) \succeq 0$.
- [ ] **No Execution Hardware Secrets**: Ensured zero references to private testbed addresses, remote hosts, or local workstation paths.
