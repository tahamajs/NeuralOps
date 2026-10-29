---
name: paper-hessian-eigenspectrum-loss
description: "Author theoretical and empirical sections on loss landscape geometry, Hessian eigenspectrum decay, top eigenvalue bounds, and flat minima generalization in compressed LLMs"
version: "1.0.0"
tags: ["neuralops", "paper", "theory", "hessian", "loss-landscape", "eigenspectrum", "generalization"]
---

# Skill: Loss Landscape Geometry, Hessian Eigenspectra, and Flat Minima Generalization

## Overview & Scope
Use this skill when authoring, formalizing, proving, or visualizing theoretical and empirical sections concerning **loss landscape curvature, Hessian eigenspectrum distributions, top eigenvalue bounds, and flat minima generalization** for top-tier machine learning conferences (NeurIPS, ICLR, ICML, MLSys, ACL) and doctoral dissertations.

In deep transformer optimization and post-training model surgery (e.g., singular value decomposition, linear attention substitution, activation quantization), parameters are projected or perturbed away from their pre-trained equilibrium. The curvature of the local loss surface dictates both **optimization stiffness** (gradient conditioning and maximum stable learning rate) and **out-of-distribution (OOD) generalization stability**:
1. **Sharp Minima ($\lambda_{\max} \gg 1$)**: High local curvature causes extreme sensitivity to weight perturbations $\Delta \theta$. Under distribution shift or low-rank truncation, the loss escalates quadratically: $\Delta \mathcal{L} \approx \frac{1}{2} \Delta \theta^\top \mathbf{H} \Delta \theta$, inducing catastrophic performance degradation.
2. **Flat Minima ($\lambda_{\max} \ll 1$, rapid spectral decay)**: Broad basin geometry exhibits wide margins of tolerance along most parameter directions. PAC-Bayes generalization bounds guarantee tighter test-set risk bounds.
3. **Hessian Eigenspectrum Hierarchy**: Neural network Hessians decouple into a bulk of near-zero eigenvalues (approximated by Marchenko-Pastur random matrix distributions) and a discrete outlier spectrum of isolated top eigenvalues ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_k$) corresponding to task-relevant low-dimensional feature subspaces.
4. **Curvature Compensation via HNC-SVD & Multi-Pillar KD**: Constraining weight surgery perturbations into the empirical Hessian null-space ($\Delta W \in \operatorname{Null}(\mathbf{H}_W)$) and smoothing post-surgery calibration via multi-objective knowledge distillation directly dampens $\lambda_{\max}$, restoring flat-basin geometry.

```
═════════════════════════════════════════════════════════════════════════════════════════════════
                LOSS LANDSCAPE GEOMETRY: SHARP VS. FLAT MINIMA UNDER COVARIATE SHIFT
═════════════════════════════════════════════════════════════════════════════════════════════════

      Loss L(θ)                                                Loss L(θ)
         ▲                                                        ▲
         │           Sharp Minimum (Pre-Surgery)                  │           Flat Minimum (Post-KD Healed)
         │           λ_max >> 1, High Curvature                   │           λ_max << 1, Broad Basin
         │                     │                                  │                     │
         │          \          │          /                       │          \          │          /
         │           \         │         /                        │           \         │         /
         │            \   *    │    *   /                         │            \        │        /
         │             \  │\   │   /│  /                          │             \   *   │   *   /
         │              \ │ \  │  / │ /                           │              \  │   │   │  /
         │               \│  \ │ /  │/                            │               \_│___│___│_/
         │                ▼   \│/   ▼                             │                 ▼   │   ▼
         │               Loss Spike ΔL >> 0                       │             Loss Invariant ΔL ≈ 0
         └─────────────────────┼──────────────────► θ             └─────────────────────┼──────────────────► θ
                              θ*                                                       θ*
                      Perturbation Δθ                                           Perturbation Δθ
═════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Mathematical Formulation of the Hessian & Eigenspectrum

### 1.1 The Full Empirical Hessian and Gauss-Newton Decomposition
Let $\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i)$ denote the empirical risk over dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ with parameter vector $\theta \in \mathbb{R}^P$. The Hessian matrix $\mathbf{H} \in \mathbb{R}^{P \times P}$ is the symmetric second-order partial derivative operator:
$$\mathbf{H} \triangleq \nabla_\theta^2 \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_\theta^2 \ell(f(x_i; \theta), y_i)$$

Applying the chain rule decomposes $\mathbf{H}$ into the **Gauss-Newton matrix** $\mathbf{H}_{\text{GN}}$ and the **residual curvature matrix** $\mathbf{H}_{\text{res}}$:
$$\mathbf{H} = \underbrace{\frac{1}{N} \sum_{i=1}^N \mathbf{J}_i^\top \left[ \nabla_{\hat{y}}^2 \ell(\hat{y}_i, y_i) \right] \mathbf{J}_i}_{\mathbf{H}_{\text{GN}} \;\succeq\; 0} \;+\; \underbrace{\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C \left[ \nabla_{\hat{y}} \ell(\hat{y}_i, y_i) \right]_k \nabla_\theta^2 f_k(x_i; \theta)}_{\mathbf{H}_{\text{res}}}$$
where $\mathbf{J}_i = \frac{\partial f(x_i; \theta)}{\partial \theta} \in \mathbb{R}^{C \times P}$ is the network Jacobian for sample $i$.

**Properties**:
1. At or near local minima ($\nabla_\theta \mathcal{L}(\theta) \approx 0$) or when model residuals are small, the second term vanishes ($\mathbf{H}_{\text{res}} \approx 0$), making $\mathbf{H} \approx \mathbf{H}_{\text{GN}} \succeq 0$ positive semi-definite.
2. For cross-entropy loss with softmax outputs $\hat{y} = \operatorname{softmax}(z)$, the output Hessian is $\nabla_z^2 \ell(z, y) = \operatorname{diag}(\hat{y}) - \hat{y} \hat{y}^\top$, ensuring positive semi-definiteness.
3. The empirical Fisher Information Matrix (FIM) $\mathbf{F} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell_i \nabla_\theta \ell_i^\top$ asymptotically aligns with $\mathbf{H}_{\text{GN}}$ under true distribution expectations: $\mathbb{E}_{x, y}[\mathbf{H}(\theta)] = \mathbb{E}_{x}[\mathbf{F}(\theta)]$.

---

### 1.2 Matrix-Free Hessian-Vector Products (Pearlmutter Trick)
Because modern language models have $P \sim 10^8 - 10^{11}$ parameters, explicitly materializing $\mathbf{H} \in \mathbb{R}^{P \times P}$ requires $\mathcal{O}(P^2)$ memory ($>400$ petabytes for a 1B parameter model) and is intractable.

Matrix-free second-order optimization computes directional curvature along an arbitrary vector $v \in \mathbb{R}^P$ via **Hessian-Vector Products (HVPs)** without allocating the full matrix:
$$\operatorname{HVP}(v) \triangleq \mathbf{H} v = \lim_{\epsilon \to 0} \frac{\nabla_\theta \mathcal{L}(\theta + \epsilon v) - \nabla_\theta \mathcal{L}(\theta)}{\epsilon} = \nabla_\theta \left( \langle \nabla_\theta \mathcal{L}(\theta), v \rangle \right)$$

```
═════════════════════════════════════════════════════════════════════════════════════════════════
                     TWO-PASS AUTOMATIC DIFFERENTIATION FOR EXACT HVP
═════════════════════════════════════════════════════════════════════════════════════════════════

  Input: Parameter θ, Direction Vector v, Loss Function L(θ)

  Pass 1 (Forward + Backward Graph Creation):
  Forward Pass:   x  ──► [ f(x; θ) ] ──► Loss L(θ)
  Backward Pass:  g(θ) = ∇_θ L(θ)  [retain_graph=True, create_graph=True]

  Pass 2 (Inner Product + Second Backward):
  Scalar Product: s = ⟨ g(θ), v ⟩ = ∑_j g_j * v_j
  HVP Vector:     H v = ∇_θ s = ∇_θ ⟨ ∇_θ L(θ), v ⟩

  Complexity: Memory O(P), Compute 2 x Standard Backward Pass FLOPs
═════════════════════════════════════════════════════════════════════════════════════════════════
```

**Implementation Invariant**:
- Requires PyTorch double-backward graph traversal (`create_graph=True` on initial `torch.autograd.grad`).
- Invariant: Vector $v$ must be detached from the computation graph to prevent higher-order gradients with respect to $v$.

---

### 1.3 Krylov Subspace & Lanczos Iteration for Eigenspectrum Extraction
To extract the top $K$ extreme eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_K$ and their associated eigenvectors $u_1, \dots, u_K$, the Lanczos algorithm iteratively constructs an orthonormal basis for the $m$-dimensional Krylov subspace:
$$\mathcal{K}_m(\mathbf{H}, v_1) \triangleq \operatorname{span}\left\{ v_1, \, \mathbf{H} v_1, \, \mathbf{H}^2 v_1, \, \dots, \, \mathbf{H}^{m-1} v_1 \right\}$$
where $v_1 \in \mathbb{R}^P$ is a normalized random probe vector ($\|v_1\|_2 = 1$).

#### Lanczos Recurrence Algorithm
For step $j = 1, 2, \dots, m$:
1. Compute directional matrix-free curvature:
   $$w_j = \mathbf{H} v_j$$
2. Compute diagonal alpha coefficient:
   $$\alpha_j = \langle w_j, v_j \rangle = v_j^\top \mathbf{H} v_j$$
3. Compute orthogonalized step:
   $$\tilde{w}_j = w_j - \alpha_j v_j - \beta_{j-1} v_{j-1} \quad (\text{with } \beta_0 = 0, \, v_0 = \mathbf{0})$$
4. Compute subdiagonal beta coefficient:
   $$\beta_j = \|\tilde{w}_j\|_2$$
5. Generate next basis vector:
   $$v_{j+1} = \frac{\tilde{w}_j}{\beta_j}$$

In compact matrix form, after $m$ steps with basis $\mathbf{V}_m = [v_1, v_2, \dots, v_m] \in \mathbb{R}^{P \times m}$ ($\mathbf{V}_m^\top \mathbf{V}_m = \mathbf{I}_m$):
$$\mathbf{H} \mathbf{V}_m = \mathbf{V}_m \mathbf{T}_m + \beta_m v_{m+1} e_m^\top$$
where $\mathbf{T}_m \in \mathbb{R}^{m \times m}$ is the symmetric tridiagonal Rayleigh quotient matrix:
$$\mathbf{T}_m = \begin{bmatrix}
\alpha_1 & \beta_1 & 0 & \dots & 0 \\
\beta_1 & \alpha_2 & \beta_2 & \dots & 0 \\
0 & \beta_2 & \alpha_3 & \ddots & \vdots \\
\vdots & \vdots & \ddots & \ddots & \beta_{m-1} \\
0 & 0 & \dots & \beta_{m-1} & \alpha_m
\end{bmatrix}$$

#### Ritz Values and Spectrum Reconstruction
The eigenvalues of the small tridiagonal matrix $\mathbf{T}_m = \mathbf{Q} \boldsymbol{\Theta} \mathbf{Q}^\top$ (where $\boldsymbol{\Theta} = \operatorname{diag}(\theta_1, \dots, \theta_m)$ with $\theta_1 \ge \dots \ge \theta_m$) are the **Ritz values** of $\mathbf{H}$, and the columns of $\mathbf{U}_m = \mathbf{V}_m \mathbf{Q}$ are the **Ritz vectors**.
- As $m \to P$, $\theta_i \to \lambda_i(\mathbf{H})$.
- In practice, for $m \ll P$ (e.g., $m = 60$ to $120$), the extreme eigenvalues $\theta_1, \dots, \theta_K$ converge exponentially to the true top eigenvalues $\lambda_1, \dots, \lambda_K$.

---

## 2. Spectral Norm Bounds, Condition Number & Optimization Stiffness

### 2.1 Spectral Norm Bounds
The spectral norm of the Hessian corresponds directly to its dominant eigenvalue:
$$\|\mathbf{H}\|_2 \triangleq \sup_{v \neq 0} \frac{\|\mathbf{H} v\|_2}{\|v\|_2} = \lambda_{\max}(\mathbf{H}) = \lambda_1$$
For a model factorized into $L$ layers $\theta = (\theta^{(1)}, \dots, \theta^{(L)})$, the global spectral norm is bounded by the block-layer operators:
$$\max_{1 \le \ell \le L} \|\mathbf{H}_{\ell\ell}\|_2 \le \|\mathbf{H}\|_2 \le \sum_{\ell=1}^L \|\mathbf{H}_{\ell\ell}\|_2 + 2 \sum_{\ell < k} \|\mathbf{H}_{\ell k}\|_2$$
where $\mathbf{H}_{\ell k} = \nabla_{\theta^{(\ell)}} \nabla_{\theta^{(k)}} \mathcal{L}(\theta)$ denotes cross-layer interaction curvature.

### 2.2 Condition Number and Learning Rate Stability
The condition number $\kappa(\mathbf{H})$ quantifies landscape anisotropy (stiffness):
$$\kappa(\mathbf{H}) \triangleq \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}^+(\mathbf{H})} = \frac{\lambda_1}{\lambda_{P_+}}$$
where $\lambda_{\min}^+(\mathbf{H})$ is the smallest non-zero eigenvalue on the non-degenerate parameter manifold.

#### Maximum Stable Learning Rate Bound
For standard gradient descent $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$ on an $L$-smooth objective ($\|\nabla^2 \mathcal{L}\|_2 \le \lambda_{\max}$):
$$\eta < \eta_{\text{crit}} \triangleq \frac{2}{\lambda_{\max}(\mathbf{H})}$$
- If $\eta > \frac{2}{\lambda_{\max}(\mathbf{H})}$, gradient descent diverges along the principal eigenvector direction $u_1$.
- **Edge of Stability (EOS)**: During Adam / SGD training, neural networks often sharpen until $\lambda_{\max}(\mathbf{H}) \approx \frac{2}{\eta}$, oscillating along the edge of stability.
- **Model Surgery Catastrophe**: Uncalibrated low-rank truncation or attention replacement alters parameter trajectories, causing a sudden spike in $\lambda_{\max}(\mathbf{H})$ that violates $\eta < \frac{2}{\lambda_{\max}}$, triggering loss divergence during downstream fine-tuning.

---

### 2.3 Bulk vs Outlier Spectrum (Random Matrix Theory)
The empirical Hessian spectrum $\rho(\lambda) = \frac{1}{P} \sum_{i=1}^P \delta(\lambda - \lambda_i)$ universally decomposes into two distinct regimes:
1. **Continuous Marchenko-Pastur Bulk**: Over $99\%$ of eigenvalues cluster near zero in a dense bulk distribution $[\lambda_-, \lambda_+]$:
   $$\rho_{\text{bulk}}(\lambda) = \frac{1}{2\pi \gamma \sigma^2 \lambda} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)} \cdot \mathbf{1}_{[\lambda_-, \lambda_+]}(\lambda)$$
   representing flat directions in parameter space (overparameterization null-space).
2. **Discrete Outlier Spectrum**: A small set of $K \ll P$ dominant eigenvalues $\lambda_1 > \lambda_2 > \dots > \lambda_K \gg \lambda_+$ detached from the bulk:
   $$\mathbf{H} \approx \sum_{k=1}^K \lambda_k u_k u_k^\top + \mathbf{H}_{\text{bulk}}$$
   These outlier directions capture the core task-specific semantic constraints and input features learned by the transformer.

```
═════════════════════════════════════════════════════════════════════════════════════════════════
                    DECOMPOSED HESSIAN EIGENSPECTRUM DENSITY ρ(λ)
═════════════════════════════════════════════════════════════════════════════════════════════════

  Density ρ(λ)
      ▲
      │    Marchenko-Pastur Bulk
      │    (Near-Zero Flat Directions)
      │          ┌───┐
      │         /     \
      │        /       \
      │       /         \                          Discrete Outlier Spectrum (Task Features)
      │      /           \                                   │
      │     /             \                         │        │        │               │
      │    /               \                        ▼        ▼        ▼               ▼
      │   /                 \                      λ_4      λ_3      λ_2             λ_max
      └──┴───────────────────┴──────────────────────┼────────┼────────┼───────────────┼───► λ
        λ_-                 λ_+
═════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 3. Sharpness, Flat Minima, and Generalization Theory

### 3.1 Formal Sharpness Formulations
Let $\mathcal{B}_\epsilon(\theta) = \{ \theta + \delta \mid \|\delta\|_2 \le \epsilon \}$ define an $\epsilon$-ball around parameter $\theta$.
1. **Worst-Case Directional Sharpness**:
   $$S_\epsilon(\theta) \triangleq \max_{\|\delta\|_2 \le \epsilon} \left[ \mathcal{L}(\theta + \delta) - \mathcal{L}(\theta) \right]$$
2. **Second-Order Taylor Characterization**:
   Expanding $\mathcal{L}(\theta + \delta)$ around a local minimum ($\nabla \mathcal{L}(\theta) \approx 0$):
   $$\mathcal{L}(\theta + \delta) - \mathcal{L}(\theta) = \nabla \mathcal{L}(\theta)^\top \delta + \frac{1}{2} \delta^\top \mathbf{H} \delta + \mathcal{O}(\|\delta\|^3) \approx \frac{1}{2} \delta^\top \mathbf{H} \delta$$
   Maximizing over the $\epsilon$-sphere $\|\delta\|_2 \le \epsilon$:
   $$S_\epsilon(\theta) \approx \frac{1}{2} \max_{\|\delta\|_2 \le \epsilon} \delta^\top \mathbf{H} \delta = \frac{1}{2} \epsilon^2 \lambda_{\max}(\mathbf{H})$$
3. **Trace Sharpness (Average Curvature)**:
   Taking the expectation over isotropic Gaussian perturbations $\delta \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$:
   $$\mathbb{E}_{\delta}[\mathcal{L}(\theta + \delta) - \mathcal{L}(\theta)] \approx \frac{1}{2} \mathbb{E}_{\delta}[\delta^\top \mathbf{H} \delta] = \frac{1}{2} \sigma^2 \operatorname{Tr}(\mathbf{H}) = \frac{1}{2} \sigma^2 \sum_{i=1}^P \lambda_i(\mathbf{H})$$

---

### 3.2 PAC-Bayes Generalization Bound
Under the PAC-Bayes framework (McAllester, 1999; Dziugaite & Roy, 2017), the expected population risk over a perturbed posterior distribution $Q = \mathcal{N}(\theta, \sigma^2 \mathbf{I})$ with prior $P_0 = \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I})$ is bounded with probability at least $1 - \delta$ over the draw of training set $\mathcal{S}$ of size $N$:

$$\mathbb{E}_{\delta \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \left[ \mathcal{L}_{\mathcal{D}}(\theta + \delta) \right] \le \mathbb{E}_{\delta \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \left[ \mathcal{L}_{\mathcal{S}}(\theta + \delta) \right] + \sqrt{\frac{\operatorname{KL}(Q \parallel P_0) + \ln \frac{2\sqrt{N}}{\delta}}{2(N - 1)}}$$

Substituting the Taylor expansion of empirical loss into the left-hand side:
$$\mathbb{E}_{\delta}[\mathcal{L}_{\mathcal{S}}(\theta + \delta)] \approx \mathcal{L}_{\mathcal{S}}(\theta) + \frac{1}{2} \sigma^2 \operatorname{Tr}(\mathbf{H}) \le \mathcal{L}_{\mathcal{S}}(\theta) + \frac{1}{2} \sigma^2 \left( K \lambda_{\max}(\mathbf{H}) + P_{\text{bulk}} \lambda_+ \right)$$

The Kullback-Leibler divergence between isotropic Gaussians evaluates to:
$$\operatorname{KL}(Q \parallel P_0) = \frac{1}{2} \left[ \frac{\|\theta\|_2^2}{\sigma_0^2} + P \left( \frac{\sigma^2}{\sigma_0^2} - 1 - \ln \frac{\sigma^2}{\sigma_0^2} \right) \right]$$

**Fundamental Generalization Insight**:
$$\mathcal{L}_{\mathcal{D}}(\theta) \le \mathcal{L}_{\mathcal{S}}(\theta) + \underbrace{\frac{1}{2} \sigma^2 \lambda_{\max}(\mathbf{H})}_{\text{Sharpness Penalty}} + \underbrace{\sqrt{\frac{\|\theta\|_2^2 / (2\sigma_0^2) + \ln(2\sqrt{N}/\delta)}{2(N-1)}}}_{\text{Complexity Penalty}} + \mathcal{O}(\sigma^4)$$
- A large top eigenvalue $\lambda_{\max}(\mathbf{H})$ directly inflates the generalization gap.
- Models with identical training loss $\mathcal{L}_{\mathcal{S}}(\theta_A) = \mathcal{L}_{\mathcal{S}}(\theta_B)$ achieve vastly different out-of-distribution transfer if $\lambda_{\max}(\mathbf{H}_A) \ll \lambda_{\max}(\mathbf{H}_B)$.

---

## 4. Impact of SVD Rank Truncation on Hessian Curvature

### 4.1 Low-Rank Weight Factorization & Curvature Relaxation
In low-rank transformer compression (e.g., SVD of attention projection matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into $U_r \Sigma_r V_r^\top$), the weight perturbation is:
$$\Delta W = \hat{W} - W = \sum_{i=r+1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i u_i v_i^\top$$

The second-order change in layer loss is governed by the layer Hessian $\mathbf{H}_W = \nabla_W^2 \mathcal{L} \in \mathbb{R}^{(d_{\text{out}} d_{\text{in}}) \times (d_{\text{out}} d_{\text{in}})}$:
$$\Delta \mathcal{L} = \operatorname{vec}(\Delta W)^\top \mathbf{H}_W \operatorname{vec}(\Delta W) + \mathcal{O}(\|\Delta W\|_F^3)$$

By Von Neumann's trace inequality and Rayleigh quotient bounds:
$$\Delta \mathcal{L} \le \frac{1}{2} \|\mathbf{H}_W\|_2 \|\Delta W\|_F^2 = \frac{1}{2} \lambda_{\max}(\mathbf{H}_W) \sum_{i=r+1}^d \sigma_i^2$$

---

### 4.2 Hessian Null-Space Projection (HNC-SVD) Mechanics
Standard SVD minimizes Frobenius weight discrepancy $\|\Delta W\|_F^2 = \sum_{i=r+1}^d \sigma_i^2$ without regard to the curvature metric $\mathbf{H}_W$. If the truncated singular vectors align with the principal eigenvector of $\mathbf{H}_W$, catastrophic output collapse occurs.

HNC-SVD resolves this by projecting weight reconstruction error into the approximate null-space of the empirical Gauss-Newton Hessian:
$$\operatorname{Null}_\epsilon(\mathbf{H}_W) \triangleq \left\{ \Delta W \;\middle|\; \operatorname{vec}(\Delta W)^\top \mathbf{H}_W \operatorname{vec}(\Delta W) \le \epsilon \|\Delta W\|_F^2 \right\}$$

Under input covariance Cholesky whitening $\mathbf{\Sigma}_X = \frac{1}{B} X X^\top = \mathbf{L} \mathbf{L}^\top$, the output error becomes:
$$\mathcal{L}_{\text{output}} = \frac{1}{B} \| W X - \hat{W} X \|_F^2 = \| (W - \hat{W}) \mathbf{L} \|_F^2 = \| \Delta W \mathbf{L} \|_F^2$$
Decomposing $\tilde{W} = W \mathbf{L} = \tilde{U} \tilde{\boldsymbol{\Sigma}} \tilde{V}^\top$ guarantees that the truncated components satisfy:
$$\operatorname{vec}(\Delta W)^\top \mathbf{H}_W \operatorname{vec}(\Delta W) \le \lambda_{r+1}(\tilde{\mathbf{\Sigma}})^2 \to 0$$
ensuring that the truncation error lies strictly within the Hessian null-space, damping post-surgery $\lambda_{\max}$.

---

## 5. Post-Surgery Multi-Pillar KD: Landscape Flattening Mechanics

### 5.1 Multi-Objective Distillation Formulation
Following structural model surgery, the surrogate model $\theta_{\text{surr}}$ is healed using the 6-pillar distillation objective:
$$\mathcal{L}_{\text{total}}(\theta) = \alpha_1 \mathcal{L}_{\text{task}} + \alpha_2 \mathcal{L}_{\text{logit}}^{\text{KL}} + \alpha_3 \mathcal{L}_{\text{hidden}}^{\text{MSE}} + \alpha_4 \mathcal{L}_{\text{rep}}^{\text{cos}} + \alpha_5 \mathcal{L}_{\text{attn}}^{\text{KL}} + \alpha_6 \mathcal{L}_{\text{geom}}^{\text{ortho}}$$

```
═════════════════════════════════════════════════════════════════════════════════════════════════
                 6-PILLAR DISTILLATION: HESSIAN FLATTENING CONTRIBUTIONS
═════════════════════════════════════════════════════════════════════════════════════════════════

  1. Task Loss (L_task):             Anchors predictions to true ground-truth targets.
  2. Logit KL (L_logit, Temp τ):      Softens softmax targets; acts as Gaussian spatial filter.
  3. Hidden State MSE (L_hidden):     Provides linear isotropic pullback across layer trajectories.
  4. Cosine Alignment (L_rep):        Regularizes angular curvature; eliminates radial stiffness.
  5. Attention Map KL (L_attn):       Preserves token interaction topology; dampens head oscillations.
  6. Orthogonal Stiefel (L_geom):     Forces basis condition number κ(U) = 1; bounds layer spectral norm.

  Net Curvature Effect:  λ_max(H_post) << λ_max(H_raw_svd)   (Restores Flat Basin Geometry)
═════════════════════════════════════════════════════════════════════════════════════════════════
```

---

### 5.2 Landscape Smoothing via Temperature-Scaled Soft Labels
Consider the logit distillation objective with temperature $\tau > 1$:
$$\mathcal{L}_{\text{logit}}(\theta) = \tau^2 D_{\text{KL}}\left( \operatorname{softmax}\left(\frac{z_{\text{teach}}}{\tau}\right) \;\middle\|\; \operatorname{softmax}\left(\frac{z_{\text{stud}}(\theta)}{\tau}\right) \right)$$

The Hessian of $\mathcal{L}_{\text{logit}}$ with respect to student logits $z$ is:
$$\nabla_z^2 \mathcal{L}_{\text{logit}} = \frac{1}{\tau} \left[ \operatorname{diag}(p_\tau) - p_\tau p_\tau^\top \right]$$
where $p_\tau = \operatorname{softmax}(z / \tau)$.
- As temperature $\tau \to \infty$, $p_\tau \to \frac{1}{C} \mathbf{1}_C$, and $\nabla_z^2 \mathcal{L}_{\text{logit}} \to \frac{1}{\tau C} \left[ \mathbf{I} - \frac{1}{C} \mathbf{1} \mathbf{1}^\top \right] \to \mathbf{0}$.
- **Curvature Contraction**: Higher distillation temperature $\tau$ acts as an isotropic low-pass Gaussian spatial filter over the loss surface:
  $$\mathcal{L}_\tau(\theta) = (\mathcal{L} * \mathcal{G}_{\sigma(\tau)})(\theta)$$
  eradicating sharp local minima and contracting the dominant eigenvalue $\lambda_{\max}(\mathbf{H})$.

---

## 6. Publication-Grade LaTeX Theorem Blocks & Proofs

When authoring theory chapters or conference submissions, format mathematical results using standard AMS-LaTeX environments (`theorem`, `lemma`, `proof`, `corollary`).

### 6.1 Theorem 1: Lanczos Iteration Convergence Rate on Top Eigenvalues
```latex
\begin{theorem}[Kaniel-Paige Lanczos Convergence on Dominant Hessian Curvature]
\label{thm:lanczos_convergence}
Let $\mathbf{H} \in \mathbb{R}^{P \times P}$ be a symmetric, positive semi-definite Hessian matrix with ordered eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_P \ge 0$. Let $\theta_1^{(m)}$ denote the maximal Ritz value obtained after $m$ steps of the matrix-free Lanczos iteration initialized with unit vector $v_1$. Then the relative approximation error of the top eigenvalue satisfies:
\begin{equation}
\label{eq:kaniel_paige_bound}
\frac{\lambda_1 - \theta_1^{(m)}}{\lambda_1 - \lambda_P} \le \left( \frac{\tan \angle(v_1, u_1)}{c_{m-1}\left( 1 + 2 \gamma_1 \right)} \right)^2
\end{equation}
where $u_1$ is the normalized eigenvector corresponding to $\lambda_1$, $\gamma_1 = \frac{\lambda_1 - \lambda_2}{\lambda_2 - \lambda_P} > 0$ is the relative eigengap, and $c_{m-1}(x)$ denotes the Chebyshev polynomial of degree $m - 1$ evaluated at $x$. In particular, the convergence is exponential in the iteration depth $m$:
\begin{equation}
0 \le \lambda_1 - \theta_1^{(m)} \le 4 (\lambda_1 - \lambda_P) \tan^2 \angle(v_1, u_1) \left( \frac{\sqrt{\kappa_1} - 1}{\sqrt{\kappa_1} + 1} \right)^{2(m - 1)}
\end{equation}
where $\kappa_1 \triangleq \frac{\lambda_1 - \lambda_P}{\lambda_2 - \lambda_P}$.
\end{theorem}

\begin{proof}
Let $\mathcal{K}_m(\mathbf{H}, v_1) = \operatorname{span}\{v_1, \mathbf{H}v_1, \dots, \mathbf{H}^{m-1}v_1\}$. By the Rayleigh-Ritz variational characterization, the maximal Ritz value satisfies $\theta_1^{(m)} = \max_{x \in \mathcal{K}_m, x \neq 0} \frac{x^\top \mathbf{H} x}{x^\top x}$. Any $x \in \mathcal{K}_m$ can be parameterized as $x = p(\mathbf{H}) v_1$ for some polynomial $p \in \mathcal{P}_{m-1}$. Decomposing $v_1 = \sum_{i=1}^P c_i u_i$ in the orthonormal eigenbasis of $\mathbf{H}$, we obtain:
\begin{equation}
\lambda_1 - \theta_1^{(m)} \le \lambda_1 - \frac{v_1^\top p(\mathbf{H}) \mathbf{H} p(\mathbf{H}) v_1}{v_1^\top p(\mathbf{H})^2 v_1} = \frac{\sum_{i=2}^P c_i^2 (\lambda_1 - \lambda_i) p(\lambda_i)^2}{c_1^2 p(\lambda_1)^2 + \sum_{i=2}^P c_i^2 p(\lambda_i)^2}
\end{equation}
Choosing $p(\lambda) = c_{m-1}\left( \frac{2\lambda - (\lambda_2 + \lambda_P)}{\lambda_2 - \lambda_P} \right)$ maximizes the ratio $p(\lambda_1) / \max_{\lambda \in [\lambda_P, \lambda_2]} |p(\lambda)|$ via the minimax optimality of Chebyshev polynomials on compact intervals. Noting that $|p(\lambda_i)| \le 1$ for all $i \ge 2$ and $c_1 = \cos \angle(v_1, u_1)$, bounding the sum yields~\eqref{eq:kaniel_paige_bound}.
\end{proof}
```

---

### 6.2 Theorem 2: Hessian Spectral Norm Bound on Generalization Gap
```latex
\begin{theorem}[PAC-Bayes Generalization Bound via Hessian Spectral Sharpness]
\label{thm:pac_bayes_hessian_sharpness}
Let $\mathcal{L}_{\mathcal{D}}(\theta)$ and $\mathcal{L}_{\mathcal{S}}(\theta)$ denote the population risk and empirical training risk on sample size $N$, respectively. Assume the local loss is twice continuously differentiable with Hessian $\mathbf{H}(\theta) = \nabla^2 \mathcal{L}_{\mathcal{S}}(\theta) \succeq 0$ possessing dominant eigenvalue $\lambda_{\max}(\mathbf{H})$ and effective rank $r_{\text{eff}} = \operatorname{Tr}(\mathbf{H}) / \lambda_{\max}(\mathbf{H})$. For any prior $P_0 = \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I}_P)$ and perturbation radius $\sigma^2 > 0$, with probability at least $1 - \delta$ over the sample draw:
\begin{equation}
\mathcal{L}_{\mathcal{D}}(\theta) \le \mathcal{L}_{\mathcal{S}}(\theta) + \frac{1}{2} \sigma^2 \lambda_{\max}(\mathbf{H}) r_{\text{eff}} + \sqrt{\frac{\frac{\|\theta\|_2^2}{2\sigma_0^2} + \frac{P}{2}\left( \frac{\sigma^2}{\sigma_0^2} - 1 - \ln \frac{\sigma^2}{\sigma_0^2} \right) + \ln \frac{2\sqrt{N}}{\delta}}{2(N - 1)}} + \mathcal{O}(\sigma^4)
\end{equation}
Furthermore, if $\theta$ resides in an anisotropic flat valley where the outlier spectrum has cardinality $K \ll P$ with $\lambda_k \le \lambda_{\max}$ and bulk eigenvalues satisfy $\lambda_{\text{bulk}} \le \epsilon \ll \lambda_{\max}$, the generalization gap satisfies:
\begin{equation}
\mathcal{L}_{\mathcal{D}}(\theta) - \mathcal{L}_{\mathcal{S}}(\theta) \le \frac{1}{2} \sigma^2 \left( K \lambda_{\max}(\mathbf{H}) + P \epsilon \right) + \mathcal{O}\left( \frac{\|\theta\|_2}{\sqrt{N}} \right)
\end{equation}
\end{theorem}
```

---

### 6.3 Theorem 3: Curvature Minimization via Hessian Null-Space Projection
```latex
\begin{theorem}[Optimality of Hessian Null-Space SVD Factorization]
\label{thm:hnc_svd_optimality}
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote an attention projection weight matrix, and let $\mathbf{H}_W = \mathbb{E}_{X \sim \mathcal{D}}[ (X X^\top) \otimes \mathbf{I}_{d_{\text{out}}} ]$ denote the layer Gauss-Newton Hessian under input activation covariance $\mathbf{\Sigma}_X = \mathbb{E}[X X^\top] = \mathbf{L} \mathbf{L}^\top$. Consider rank-$r$ truncated surrogates $\hat{W} = A B$ with $A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{r \times d_{\text{in}}}$. 
The second-order loss perturbation $\Delta \mathcal{L} = \frac{1}{2} \operatorname{vec}(W - \hat{W})^\top \mathbf{H}_W \operatorname{vec}(W - \hat{W})$ is globally minimized if and only if $\hat{W}$ is computed via Cholesky-whitened SVD:
\begin{equation}
\hat{W}_{\text{HNC}} = \tilde{U}_r \tilde{\boldsymbol{\Sigma}}_r \tilde{V}_r^\top \mathbf{L}^{-1}
\end{equation}
where $\tilde{U} \tilde{\boldsymbol{\Sigma}} \tilde{V}^\top = \operatorname{SVD}(W \mathbf{L})$. Moreover, the excess curvature loss satisfies the strict infimum bound:
\begin{equation}
\min_{\operatorname{rank}(\hat{W}) \le r} \operatorname{Tr}\left( (W - \hat{W}) \mathbf{\Sigma}_X (W - \hat{W})^\top \right) = \sum_{j=r+1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_j(W \mathbf{L})^2
\end{equation}
\end{theorem}
```

---

### 6.4 Theorem 4: Distillation-Induced Landscape Smoothing
```latex
\begin{theorem}[Curvature Damping via Multi-Pillar Distillation]
\label{thm:distillation_curvature_damping}
Let $\mathcal{L}_{\text{task}}(\theta)$ possess local Hessian $\mathbf{H}_{\text{task}} = \nabla^2 \mathcal{L}_{\text{task}}$ with top eigenvalue $\lambda_{\max}(\mathbf{H}_{\text{task}})$. Let $\mathcal{L}_{\text{KD}}(\theta; \tau)$ denote the Kullback-Leibler divergence with teacher model $f_{\text{teach}}$ at temperature $\tau > 1$, and let $\mathbf{H}_{\text{total}} = (1 - \alpha) \mathbf{H}_{\text{task}} + \alpha \mathbf{H}_{\text{KD}}$. Under bounded teacher prediction entropy $H(p_{\text{teach}}) \ge H_0$, the top eigenvalue of the composite Hessian satisfies:
\begin{equation}
\lambda_{\max}(\mathbf{H}_{\text{total}}) \le (1 - \alpha) \lambda_{\max}(\mathbf{H}_{\text{task}}) + \frac{\alpha}{\tau^2} \sup_{x} \|\mathbf{J}_{\text{stud}}(x)\|_2^2 \cdot \left( 1 - \frac{1}{C} \right)
\end{equation}
where $\mathbf{J}_{\text{stud}}$ is the student Jacobian and $C$ is the class dimension. Consequently, for any $\tau > \sqrt{\frac{\alpha (1 - 1/C) \|\mathbf{J}\|_2^2}{\lambda_{\max}(\mathbf{H}_{\text{task}})}}$, distillation strictly contracts the maximum landscape curvature:
\begin{equation}
\lambda_{\max}(\mathbf{H}_{\text{total}}) < \lambda_{\max}(\mathbf{H}_{\text{task}})
\end{equation}
\end{theorem}
```

---

## 7. Empirical Profiling & 300 DPI Visualization Guidelines

### 7.1 Matrix-Free Lanczos & Power Iteration Implementation Pattern
When validating theory against empirical models, implement the matrix-free double autograd loop:

```python
import torch
import torch.nn as nn
from typing import List, Tuple

def compute_hvp(loss: torch.Tensor, params: List[torch.Tensor], v: List[torch.Tensor]) -> List[torch.Tensor]:
    """Computes exact Hessian-Vector Product H * v using double automatic differentiation.
    
    Memory: O(P), Compute: 2x backward pass FLOPs.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    inner_prod = sum((g * vi).sum() for g, vi in zip(grads, v))
    hvp_tensors = torch.autograd.grad(inner_prod, params, retain_graph=True)
    return [h.detach() for h in hvp_tensors]

def lanczos_top_eigenvalues(
    model: nn.Module, 
    loss_fn: nn.Module, 
    x_batch: torch.Tensor, 
    y_batch: torch.Tensor, 
    k_dim: int = 50, 
    target_param: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Matrix-free Lanczos tridiagonalization extracting top-k Ritz eigenvalues."""
    model.eval()
    if target_param is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [target_param]
        
    # Forward loss
    out = model(x_batch)
    loss = loss_fn(out, y_batch)
    
    # Initialize normalized probe vector v_1
    v = [torch.randn_like(p) for p in params]
    norm_v = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / (norm_v + 1e-12) for vi in v]
    
    alphas = []
    betas = []
    v_prev = [torch.zeros_like(p) for p in params]
    beta_prev = 0.0
    
    for j in range(k_dim):
        Hv = compute_hvp(loss, params, v)
        # alpha_j = <Hv, v>
        alpha_j = sum((h * vi).sum() for h, vi in zip(Hv, v)).item()
        alphas.append(alpha_j)
        
        # w_tilde = Hv - alpha_j * v - beta_{j-1} * v_{j-1}
        w_tilde = [h - alpha_j * vi - beta_prev * vp for h, vi, vp in zip(Hv, v, v_prev)]
        beta_j = torch.sqrt(sum((wt ** 2).sum() for wt in w_tilde)).item()
        
        if j < k_dim - 1:
            betas.append(beta_j)
            v_prev = v
            v = [wt / (beta_j + 1e-12) for wt in w_tilde]
            beta_prev = beta_j
            
    # Construct tridiagonal matrix T_m
    T = torch.zeros(k_dim, k_dim)
    for i in range(k_dim):
        T[i, i] = alphas[i]
        if i < k_dim - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
            
    ritz_values, ritz_vectors = torch.linalg.eigh(T)
    # Sort descending
    sorted_indices = torch.argsort(ritz_values, descending=True)
    return ritz_values[sorted_indices], ritz_vectors[:, sorted_indices]
```

---

### 7.2 Publication Visualization Standards (300 DPI Dashboard)
When producing figures for manuscripts, strictly adhere to publication aesthetic standards:
1. **Resolution**: `dpi=300`, vector-exportable PDF / SVG compatibility.
2. **Palette**: `sns.color_palette("deep")` or `mako` / `viridis`; never saturated pure primaries.
3. **Typography**: Matplotlib LaTeX font rendering (`usetex=True` or `mathtext.fontset = 'stix'`), font sizes: Titles 11pt, Labels 9pt, Legend/Ticks 8pt.
4. **4-Panel Layout Strategy**:
   - **Panel A (Log-Log Eigenspectrum Decay)**: Plot $\ln \lambda_i$ vs $\ln i$ illustrating the power-law decay rate $\alpha$ and the spectral gap between outlier eigenvalues and the Marchenko-Pastur bulk.
   - **Panel B (Spectral Density $\rho(\lambda)$)**: Gaussian kernel density estimation (KDE) or Lanczos-Stieltjes spectral density showing the separation of discrete spikes from the continuous bulk.
   - **Panel C (Sharpness vs Generalization Gap)**: Scatter/line plot comparing $\lambda_{\max}$ across model variants (Dense, Raw SVD, HNC-SVD, Distilled) against out-of-distribution transfer error.
   - **Panel D (Loss Surface 2D Contour Slices)**: Filter-normalized contour plot along top two eigenvector directions $u_1, u_2$ demonstrating the convex basin broadening achieved by multi-pillar distillation.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_publication_hessian_dashboard(
    eigenvalues_dict: dict, 
    contour_grids: dict, 
    save_path: str = "figures/hessian_diagnostics_300dpi.pdf"
):
    """Generates a 300 DPI, 4-panel publication-grade Hessian diagnostics dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), dpi=300)
    palette = sns.color_palette("deep")
    
    # Panel A: Eigenspectrum Decay (Log-Log)
    ax1 = axes[0, 0]
    for idx, (label, evals) in enumerate(eigenvalues_dict.items()):
        ranks = np.arange(1, len(evals) + 1)
        ax1.loglog(ranks, evals, marker='o', markersize=3, label=label, color=palette[idx], alpha=0.85)
    ax1.set_xlabel(r"Eigenvalue Index $i$", fontsize=9)
    ax1.set_ylabel(r"Eigenvalue $\lambda_i(\mathbf{H})$", fontsize=9)
    ax1.set_title(r"\textbf{(a) Hessian Eigenspectrum Decay}", fontsize=10, loc="left")
    ax1.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    ax1.legend(frameon=True, fontsize=8)
    
    # Panel B: Spectral Density
    ax2 = axes[0, 1]
    for idx, (label, evals) in enumerate(eigenvalues_dict.items()):
        sns.kdeplot(evals, ax=ax2, label=label, color=palette[idx], fill=True, alpha=0.2, bw_adjust=0.5)
    ax2.set_xlabel(r"Curvature $\lambda$", fontsize=9)
    ax2.set_ylabel(r"Spectral Density $\rho(\lambda)$", fontsize=9)
    ax2.set_title(r"\textbf{(b) Empirical Spectral Density}", fontsize=10, loc="left")
    ax2.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax2.legend(frameon=True, fontsize=8)
    
    # Panel C: Condition Number Progression
    ax3 = axes[1, 0]
    stages = list(eigenvalues_dict.keys())
    l_max = [evals[0] for evals in eigenvalues_dict.values()]
    bars = ax3.bar(stages, l_max, color=palette[:len(stages)], width=0.5, edgecolor='black', lw=0.6)
    ax3.set_ylabel(r"Top Eigenvalue $\lambda_{\max}(\mathbf{H})$", fontsize=9)
    ax3.set_title(r"\textbf{(c) Landscape Sharpness ($\lambda_{\max}$) Across Surgery}", fontsize=10, loc="left")
    ax3.grid(True, axis="y", ls="--", lw=0.4, alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    # Panel D: 2D Loss Landscape Contour Slices
    ax4 = axes[1, 1]
    X, Y, Z = contour_grids["X"], contour_grids["Y"], contour_grids["Z"]
    cp = ax4.contourf(X, Y, Z, levels=25, cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(cp, ax=ax4, shrink=0.85)
    cbar.set_label(r"Loss $\mathcal{L}(\theta^* + \delta_1 u_1 + \delta_2 u_2)$", fontsize=8)
    ax4.set_xlabel(r"Principal Eigenvector $u_1$", fontsize=9)
    ax4.set_ylabel(r"Second Eigenvector $u_2$", fontsize=9)
    ax4.set_title(r"\textbf{(d) 2D Loss Basin Geometry (Li et al. Normalization)}", fontsize=10, loc="left")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
```

---

### 7.3 Canonical LaTeX Summary Table Template
```latex
\begin{table*}[t]
\centering
\small
\caption{\textbf{Empirical Hessian Diagnostics & Curvature Telemetry across Transformer Layers}. Evaluated on 12-layer RoBERTa-base on SST-2 calibration data ($B=32, S=128$). Dominant eigenvalue $\lambda_{\max}$, spectral condition number $\kappa(\mathbf{H})$, effective rank $r_{\text{eff}}$, and maximal stable gradient descent step size $\eta_{\text{crit}} = 2 / \lambda_{\max}$.}
\label{tab:hessian_diagnostics_master}
\begin{tabular}{lcccccc}
\toprule
\textbf{Layer / Projection} & \textbf{SVD Rank $r$} & $\lambda_{\max}(\mathbf{H})$ & $\kappa(\mathbf{H})$ & $r_{\text{eff}} = \frac{\operatorname{Tr}(\mathbf{H})}{\lambda_{\max}}$ & $\eta_{\text{crit}} = \frac{2}{\lambda_{\max}}$ & \textbf{Post-Surgery Status} \\
\midrule
Layer 0 $W_q$ (Pre-surgery) & $768$ (Full) & $142.84$ & $8.9 \times 10^4$ & $4.12$ & $0.0140$ & Baseline Reference \\
Layer 0 $W_q$ (Raw Truncation) & $32$ (SVD) & $489.12$ & $3.4 \times 10^5$ & $1.84$ & $0.0041$ & Severe Sharpening \\
Layer 0 $W_q$ (HNC-SVD) & $32$ (HNC) & $128.40$ & $7.2 \times 10^4$ & $4.95$ & $0.0156$ & Curvature Preserved \\
Layer 0 $W_q$ (6-Pillar KD) & $32$ (Healed) & $\mathbf{68.15}$ & $\mathbf{3.1 \times 10^4}$ & $\mathbf{7.82}$ & $\mathbf{0.0293}$ & Optimal Flat Minimum \\
\midrule
Layer 6 $W_{\text{mlp1}}$ (Raw Truncation) & $64$ (SVD) & $812.30$ & $1.1 \times 10^6$ & $1.42$ & $0.0024$ & Gradient Divergence \\
Layer 6 $W_{\text{mlp1}}$ (HNC-SVD) & $64$ (HNC) & $241.05$ & $1.8 \times 10^5$ & $3.68$ & $0.0083$ & Null-space Contained \\
Layer 6 $W_{\text{mlp1}}$ (6-Pillar KD) & $64$ (Healed) & $\mathbf{115.42}$ & $\mathbf{6.4 \times 10^4}$ & $\mathbf{6.21}$ & $\mathbf{0.0173}$ & Basin Broadened \\
\midrule
Layer 11 $W_o$ (Raw Truncation) & $32$ (SVD) & $315.60$ & $4.2 \times 10^5$ & $2.11$ & $0.0063$ & High OOD Error \\
Layer 11 $W_o$ (HNC-SVD) & $32$ (HNC) & $98.70$ & $5.1 \times 10^4$ & $5.30$ & $0.0202$ & Stable Transfer \\
Layer 11 $W_o$ (6-Pillar KD) & $32$ (Healed) & $\mathbf{49.80}$ & $\mathbf{2.3 \times 10^4}$ & $\mathbf{8.44}$ & $\mathbf{0.0401}$ & Maximum Flatness \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 8. Verification Checklist for Theoretical & Empirical Content

Before submitting manuscripts, book chapters, or research reports containing Hessian and loss landscape analyses, verify every item against this checklist:

```
[ ] Exact Autograd Formulation:
    Matrix-free HVP explicitly verified with create_graph=True, retain_graph=True, and detached direction vector v.
[ ] Spectral Decomposition Hierarchy:
    Formulation distinguishes between Marchenko-Pastur continuous bulk and discrete outlier eigenvalues.
[ ] Maximum Stable Step Size:
    Learning rate analysis grounds the Edge of Stability threshold η_crit = 2 / λ_max.
[ ] PAC-Bayes Alignment:
    Generalization bounds explicitly connect population test risk with trace Tr(H) and top eigenvalue λ_max.
[ ] Null-Space Proof Integrity:
    HNC-SVD theorem demonstrates that Cholesky activation whitening maps truncation error into the kernel of H_GN.
[ ] Distillation Smoothing Physics:
    Multi-pillar knowledge distillation explained via temperature-induced spatial convolution contracting λ_max.
[ ] AMS-LaTeX Standard Compliance:
    All proofs end with \end{proof} (Q.E.D. symbol), all equations properly numbered with \label{} and cross-referenced.
[ ] Figure Aesthetic Integrity:
    Visualizations formatted to 300 DPI, vector PDF compatible, labeled with LaTeX math symbols and professional palettes.
```
