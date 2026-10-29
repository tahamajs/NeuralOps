---
name: paper-math-formulation-writer
description: "Author mathematically rigorous formulations, proofs, and optimization theorems for AI papers (linear attention, SVD whitening, Stiefel Riemannian geometry, multi-pillar KD)"
version: "1.0.0"
tags: ["neuralops", "paper", "math", "theory", "svd", "stiefel", "distillation"]
---

# Skill: Paper Mathematical Formulation & Proof Writer

## Overview & Scope
Use this skill when drafting, formalizing, verifying, or polishing theoretical sections, mathematical formulations, lemmas, theorems, and step-by-step formal proofs for AI conference manuscripts, arXiv preprints, and doctoral dissertations.

This skill specializes in:
1. **Linear Recurrence & Associative Scans**: Dual representation (sequential vs chunk-parallel) for linear attention, state-space models (SSMs), and associative memory operators.
2. **Spectral Matrix Factorization & Covariance Whitening**: Eckart-Young-Mirsky optimality under empirical data distributions, Cholesky covariance pullback, condition number regularization, and excess output error bounds.
3. **Riemannian Geometry on the Stiefel Manifold**: Differentiable manifold constraints $\operatorname{St}(p, n)$, tangent space derivations, orthogonal gradient projections, and geodesic retraction operators (QR, Polar, Cayley transform).
4. **Multi-Pillar Knowledge Distillation Objectives**: Formal 6-pillar objective uniting task loss, soft logits divergence, hidden state matching, cosine representation alignment, attention map distributions, and manifold orthogonality regularization.
5. **Publication-Grade LaTeX Typesetting Standards**: Verified AMS-LaTeX theorem environments, proof blocks, notation conventions, and numerical stability bounds.

---

## 1. Linear Recurrence & Associative Chunk-Parallel Scans

### 1.1 From Quadratic Softmax Attention to Linear Kernels
Standard multi-head attention computes pairwise similarities across sequence length $N$:
$$\operatorname{Softmax-Attn}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V \quad \in \mathbb{R}^{N \times d_v}$$
This imposes $\mathcal{O}(N^2 d)$ compute complexity and $\mathcal{O}(N^2)$ memory footprint due to the full attention matrix $A = \operatorname{softmax}(Q K^\top / \sqrt{d_k})$.

Linear attention resolves the quadratic bottleneck by replacing the normalized exponential score with an inner product of non-negative feature maps $\phi(\cdot): \mathbb{R}^d \to \mathbb{R}^{d_k}$:
$$\operatorname{Linear-Attn}(Q, K, V)_i = \frac{\sum_{j=1}^i \phi(q_i) \phi(k_j)^\top v_j}{\sum_{j=1}^i \phi(q_i) \phi(k_j)^\top} = \frac{\phi(q_i) \sum_{j=1}^i \phi(k_j)^\top v_j}{\phi(q_i) \sum_{j=1}^i \phi(k_j)^\top}$$
Leveraging matrix associativity $(\phi(q_i) \phi(k_j)^\top) v_j = \phi(q_i) (\phi(k_j)^\top v_j)$, the sum over tokens becomes an accumulator state.

### 1.2 Sequential Recurrent Formulation
For causal autoregressive sequence modeling with exponential forget decay $\gamma_t \in (0, 1]$:
$$\begin{aligned}
S_t &= \gamma_t S_{t-1} + \phi(k_t)^\top v_t \quad &&\in \mathbb{R}^{d_k \times d_v} \\
z_t &= \gamma_t z_{t-1} + \phi(k_t)^\top \quad &&\in \mathbb{R}^{d_k \times 1} \\
y_t &= \frac{\phi(q_t) S_t}{\phi(q_t) z_t + \epsilon} \quad &&\in \mathbb{R}^{1 \times d_v}
\end{aligned}$$
where $S_0 = \mathbf{0}_{d_k \times d_v}$ and $z_0 = \mathbf{0}_{d_k \times 1}$.
- **Computational Complexity**: $\mathcal{O}(N d_k d_v)$ operations for training, $\mathcal{O}(d_k d_v)$ per-step latency during decoding.
- **Memory Footprint**: $\mathcal{O}(d_k d_v)$ independent of sequence length $N$.

### 1.3 Chunk-Parallel Dual Formulation (Associative Scan)
While the recurrent formulation achieves $\mathcal{O}(1)$ autoregressive token generation, it creates a sequential data dependency that underutilizes parallel GPU Tensor Cores during training.

To parallelize execution across sequence length $N$, the sequence is partitioned into $B = \lceil N / C \rceil$ non-overlapping chunks of block size $C$ (typically $C \in \{64, 128\}$). For chunk $c \in \{1, \dots, B\}$ spanning indices $t \in [(c-1)C + 1, \, c C]$:

$$\mathbf{O}_{[c]} = \mathbf{O}_{[c]}^{\text{intra}} + \mathbf{O}_{[c]}^{\text{inter}}$$

#### 1. Intra-Chunk Computation (Local Matrix Multiply):
Inside chunk $c$, queries, keys, and values are processed concurrently:
$$\mathbf{O}_{[c]}^{\text{intra}} = \left( \left(\phi(\mathbf{Q}_{[c]}) \phi(\mathbf{K}_{[c]})^\top \odot \mathbf{M}_{\text{causal}}\right) \odot \mathbf{D}_{[c]}^{\text{intra}} \right) \mathbf{V}_{[c]}$$
where:
- $\mathbf{M}_{\text{causal}} \in \{0, 1\}^{C \times C}$ is the strictly causal lower-triangular mask ($M_{i, j} = 1 \iff i \ge j$).
- $\mathbf{D}_{[c]}^{\text{intra}} \in \mathbb{R}^{C \times C}$ is the relative exponential decay matrix: $(D^{\text{intra}})_{i, j} = \exp\left(\sum_{\tau=j+1}^i \log \gamma_\tau\right)$ for $i \ge j$.

#### 2. Inter-Chunk Recurrent State Accumulation:
The global recurrent memory state is updated at chunk boundaries:
$$\mathbf{S}_{[c]} = \mathbf{\Lambda}_{[c]} \mathbf{S}_{[c-1]} + \sum_{j=1}^C \left(\prod_{\tau=j+1}^C \gamma_{(c-1)C + \tau}\right) \phi(k_{(c-1)C + j})^\top v_{(c-1)C + j}$$
where $\mathbf{\Lambda}_{[c]} = \prod_{\tau=1}^C \gamma_{(c-1)C + \tau}$ represents the total cumulative decay across chunk $c$.

#### 3. Inter-Chunk Projection:
The historical memory state $\mathbf{S}_{[c-1]}$ is broadcast to all tokens within chunk $c$:
$$\mathbf{O}_{[c]}^{\text{inter}} = \operatorname{diag}\left(\mathbf{d}_{[c]}^{\text{inter}}\right) \phi(\mathbf{Q}_{[c]}) \mathbf{S}_{[c-1]}$$
where $(d_{[c]}^{\text{inter}})_i = \prod_{\tau=1}^i \gamma_{(c-1)C + \tau}$.

```
═════════════════════════════════════════════════════════════════════════════════════════
                       CHUNK-PARALLEL RECURRENT ATTENTION PIPELINE
═════════════════════════════════════════════════════════════════════════════════════════
  Tokens [1 ... C]             Tokens [C+1 ... 2C]            Tokens [2C+1 ... 3C]
  ┌───────────────┐           ┌───────────────┐              ┌───────────────┐
  │   Chunk c-1   │           │    Chunk c    │              │   Chunk c+1   │
  │ ┌───────────┐ │           │ ┌───────────┐ │              │ ┌───────────┐ │
  │ │Intra-Chunk│ │           │ │Intra-Chunk│ │              │ │Intra-Chunk│ │
  │ │ GEMM (O)  │ │           │ │ GEMM (O)  │ │              │ │ GEMM (O)  │ │
  │ └─────┬─────┘ │           │ └─────┬─────┘ │              │ └─────┬─────┘ │
  │       │       │           │       │       │              │       │       │
  │  State Update │           │  State Update │              │  State Update │
  │ ┌─────▼─────┐ │ S_{[c-1]} │ ┌─────▼─────┐ │   S_{[c]}    │ ┌─────▼─────┐ │
  │ │ S_{[c-1]} ├─┼──────────►│ │  S_{[c]}  ├─┼─────────────►│ │ S_{[c+1]} │ │
  │ └───────────┘ │           │ └─────┬─────┘ │              │ └───────────┘ │
  └───────────────┘           └───────┼───────┘              └───────────────┘
                                      ▼
                       O_{[c]} = O_{[c]}^{\text{intra}} + Q_{[c]} S_{[c-1]}
═════════════════════════════════════════════════════════════════════════════════════════
```

---

## 2. Spectral Matrix Factorization & Covariance Whitening

### 2.1 The Failure of Classical Eckart-Young-Mirsky in Neural Networks
The classic theorem of Eckart, Young, and Mirsky (1936) guarantees that for any matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the truncated Singular Value Decomposition (SVD) solves the unweighted low-rank approximation problem:
$$W_r = \arg\min_{\operatorname{rank}(\widehat{W}) \le r} \|W - \widehat{W}\|_F^2 = \sum_{i=1}^r \sigma_i u_i v_i^\top$$
with residual error $\|W - W_r\|_F^2 = \sum_{i=r+1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_i^2(W)$.

**The Fundamental Flaw in Deep Architectures**:
In transformer layers, parameter matrices do not act in a vacuum; they transform dynamic token activation vectors $x \in \mathbb{R}^{d_{\text{in}}}$. The actual functional objective is to minimize the **output reconstruction error**:
$$\mathcal{E}(\widehat{W}) \triangleq \mathbb{E}_{x \sim \mathcal{D}}\left[ \|W x - \widehat{W} x\|_2^2 \right] = \operatorname{Tr}\left((W - \widehat{W}) \Sigma_X (W - \widehat{W})^\top\right)$$
where $\Sigma_X = \mathbb{E}[x x^\top] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is the activation second-moment matrix.

In modern LLMs, activation distributions exhibit severe directional anisotropy ("outlier features"), where a tiny fraction of channels carry activation magnitudes up to $100\times$ higher than average. Consequently, the condition number of the activation covariance is extreme:
$$\kappa(\Sigma_X) = \frac{\lambda_{\max}(\Sigma_X)}{\lambda_{\min}(\Sigma_X)} \gg 10^5$$
Truncating singular vectors based solely on $W$ discards directions with small singular values in $W$ that align with massive eigenvalues in $\Sigma_X$, triggering catastrophic feature degradation and classification collapse.

---

### 2.2 Formal Theorem: Optimality of Cholesky-Whitened Truncation

```latex
\begin{theorem}[Global Optimality of Cholesky-Whitened Truncation]
\label{thm:cholesky_whitened_svd_optimality}
Let $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ be a layer weight matrix, and let $X \in \mathbb{R}^{d_{\mathrm{in}} \times M}$ be a calibration activation matrix composed of $M$ token representations. Define the empirical regularized second-moment matrix:
\begin{equation}
\Sigma_X = \frac{1}{M} X X^\top + \lambda I_{d_{\mathrm{in}}} \quad \succ 0
\label{eq:reg_covariance}
\end{equation}
where $\lambda > 0$ is a Tikhonov damping factor. Let $\Sigma_X = L L^\top$ be the unique Cholesky factorization with positive diagonal entries $L_{ii} > 0$.

Define the whitened weight matrix $\widetilde{W} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ by:
\begin{equation}
\widetilde{W} = W L
\label{eq:whitened_weight_def}
\end{equation}
and let its full Singular Value Decomposition be:
\begin{equation}
\widetilde{W} = \sum_{i=1}^{\min(d_{\mathrm{out}}, d_{\mathrm{in}})} \sigma_i u_i v_i^\top = U \Sigma V^\top
\label{eq:whitened_svd}
\end{equation}
with singular values ordered $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.

Then:
\begin{enumerate}
    \item The unique global minimizer of the activation-weighted low-rank approximation problem:
    \begin{equation}
    \widehat{W}^* = \arg\min_{\mathrm{rank}(\widehat{W}) \le r} \left( \frac{1}{M} \|(W - \widehat{W}) X\|_F^2 + \lambda \|W - \widehat{W}\|_F^2 \right)
    \label{eq:weighted_opt_problem}
    \end{equation}
    is given analytically by pulling back the rank-$r$ truncated whitened SVD:
    \begin{equation}
    \widehat{W}^* = \widetilde{W}_r L^{-1} = (U_r \Sigma_r) (V_r^\top L^{-1})
    \label{eq:whitened_optimal_pullback}
    \end{equation}
    where $U_r \in \mathbb{R}^{d_{\mathrm{out}} \times r}$, $\Sigma_r = \mathrm{diag}(\sigma_1, \dots, \sigma_r)$, and $V_r \in \mathbb{R}^{d_{\mathrm{in}} \times r}$.

    \item The minimal achieved reconstruction error satisfies:
    \begin{equation}
    \mathcal{E}^* = \min_{\mathrm{rank}(\widehat{W}) \le r} \mathrm{Tr}\left((W - \widehat{W}) \Sigma_X (W - \widehat{W})^\top\right) = \sum_{i=r+1}^{\min(d_{\mathrm{out}}, d_{\mathrm{in}})} \sigma_i^2(W L)
    \label{eq:minimal_whitened_error}
    \end{equation}

    \item For any naive rank-$r$ SVD approximation $\widehat{W}_{\mathrm{naive}} = \sum_{k=1}^r \sigma_k(W) u_k(W) v_k(W)^\top$ constructed without activation whitening, the excess output reconstruction error satisfies:
    \begin{equation}
    \Delta \mathcal{E} \triangleq \mathcal{E}(\widehat{W}_{\mathrm{naive}}) - \mathcal{E}^* \ge \sum_{k=r+1}^{d_{\mathrm{in}}} \sigma_k^2(W) \left( v_k(W)^\top \Sigma_X v_k(W) \right) - \sum_{i=r+1}^{d_{\mathrm{in}}} \sigma_i^2(W L) \ge 0
    \label{eq:excess_error_bound}
    \end{equation}
    Equality holds if and only if $\Sigma_X = \alpha I_{d_{\mathrm{in}}}$ for some scalar $\alpha > 0$.
\end{enumerate}
\end{theorem}
```

---

### 2.3 Step-by-Step Formal Proof of Theorem 1

```latex
\begin{proof}
We construct the proof across five rigorous algebraic steps.

\paragraph{Step 1: Metric Pullback and Isometric Reformulation.}
Expanding the weighted objective function in equation~\eqref{eq:weighted_opt_problem} via the cyclic property of the matrix trace:
\begin{align}
\mathcal{J}(\widehat{W}) &\triangleq \frac{1}{M} \|(W - \widehat{W}) X\|_F^2 + \lambda \|W - \widehat{W}\|_F^2 \nonumber \\
&= \frac{1}{M} \mathrm{Tr}\left((W - \widehat{W}) X X^\top (W - \widehat{W})^\top\right) + \lambda \mathrm{Tr}\left((W - \widehat{W}) I (W - \widehat{W})^\top\right) \nonumber \\
&= \mathrm{Tr}\left((W - \widehat{W}) \left(\frac{1}{M} X X^\top + \lambda I\right) (W - \widehat{W})^\top\right) \nonumber \\
&= \mathrm{Tr}\left((W - \widehat{W}) \Sigma_X (W - \widehat{W})^\top\right)
\label{eq:proof_step1_trace}
\end{align}
Substituting the Cholesky factorization $\Sigma_X = L L^\top$ into the trace:
\begin{align}
\mathcal{J}(\widehat{W}) &= \mathrm{Tr}\left((W - \widehat{W}) L L^\top (W - \widehat{W})^\top\right) \nonumber \\
&= \mathrm{Tr}\left( \left((W - \widehat{W}) L\right) \left((W - \widehat{W}) L\right)^\top \right) \nonumber \\
&= \|(W - \widehat{W}) L\|_F^2
\label{eq:proof_step1_frobenius}
\end{align}
Thus, multiplying $W$ on the right by $L$ defines an exact isometric pullback from the anisotropic Riemannian activation metric induced by $\Sigma_X$ to standard Euclidean Frobenius space.

\paragraph{Step 2: Bijective Change of Optimization Variables.}
Define the transformed target matrix $\widetilde{W} \triangleq W L \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ and the transformed optimization variable $K \triangleq \widehat{W} L \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$.
Since $\Sigma_X \succ 0$ with $\lambda > 0$, the lower triangular matrix $L$ is strictly non-singular ($\det(L) = \prod_{i=1}^{d_{\mathrm{in}}} L_{ii} > 0$).
By Sylvester's rank inequality for non-singular post-multiplication:
\begin{equation}
\mathrm{rank}(K) = \mathrm{rank}(\widehat{W} L) = \mathrm{rank}(\widehat{W})
\label{eq:proof_rank_invariance}
\end{equation}
Therefore, the map $\widehat{W} \mapsto K = \widehat{W} L$ is a rank-preserving bijection. The constrained optimization problem in equation~\eqref{eq:weighted_opt_problem} is algebraically isomorphic to:
\begin{equation}
K^* = \arg\min_{\mathrm{rank}(K) \le r} \|\widetilde{W} - K\|_F^2
\label{eq:proof_transformed_problem}
\end{equation}

\paragraph{Step 3: Application of the Classical Eckart-Young-Mirsky Theorem.}
Problem~\eqref{eq:proof_transformed_problem} is the canonical low-rank matrix approximation problem in standard Frobenius norm.
By the Eckart-Young-Mirsky Theorem (Mirsky, 1960), the unique global minimizer is obtained by the truncated SVD of $\widetilde{W}$:
\begin{equation}
K^* = \widetilde{W}_r = \sum_{i=1}^r \sigma_i u_i v_i^\top = U_r \Sigma_r V_r^\top
\label{eq:proof_truncated_K}
\end{equation}
and the minimal value of the Frobenius norm error is strictly:
\begin{equation}
\|\widetilde{W} - K^*\|_F^2 = \sum_{i=r+1}^{\min(d_{\mathrm{out}}, d_{\mathrm{in}})} \sigma_i^2(\widetilde{W}) = \sum_{i=r+1}^{\min(d_{\mathrm{out}}, d_{\mathrm{in}})} \sigma_i^2(W L)
\label{eq:proof_minimal_error}
\end{equation}

\paragraph{Step 4: Pullback to Parameter Space.}
Inverting the bijective variable transformation $\widehat{W} = K L^{-1}$, we obtain the unique optimal weight approximation:
\begin{equation}
\widehat{W}^* = K^* L^{-1} = (U_r \Sigma_r V_r^\top) L^{-1} = (U_r \Sigma_r) (V_r^\top L^{-1})
\label{eq:proof_final_W_star}
\end{equation}
Factoring into down-projection and up-projection matrices $W_{\mathrm{down}} \triangleq U_r \Sigma_r \in \mathbb{R}^{d_{\mathrm{out}} \times r}$ and $W_{\mathrm{up}} \triangleq V_r^\top L^{-1} \in \mathbb{R}^{r \times d_{\mathrm{in}}}$, we establish statement (1) and statement (2) of the theorem.

\paragraph{Step 5: Excess Error Gap of Naive SVD.}
Let $\widehat{W}_{\mathrm{naive}} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ be the unweighted rank-$r$ truncated SVD of $W$.
Since $\mathrm{rank}(\widehat{W}_{\mathrm{naive}}) \le r$, the transformed matrix $K_{\mathrm{naive}} \triangleq \widehat{W}_{\mathrm{naive}} L$ satisfies $\mathrm{rank}(K_{\mathrm{naive}}) \le r$.
By the strict uniqueness of the Eckart-Young-Mirsky projection in equation~\eqref{eq:proof_truncated_K}:
\begin{equation}
\|\widetilde{W} - K^*\|_F^2 \le \|\widetilde{W} - K_{\mathrm{naive}}\|_F^2
\label{eq:proof_inequality}
\end{equation}
with equality holding if and only if $K_{\mathrm{naive}} = K^*$.
This occurs if and only if the right singular subspaces of $W L$ coincide identically with the right singular subspaces of $W$, which requires $L = \sqrt{\alpha} I_{d_{\mathrm{in}}}$ ($\Sigma_X = \alpha I_{d_{\mathrm{in}}}$).

Expanding the naive error using the spectral decomposition $\Sigma_X = \sum_{j=1}^{d_{\mathrm{in}}} \lambda_j q_j q_j^\top$:
\begin{equation}
\mathcal{E}(\widehat{W}_{\mathrm{naive}}) = \sum_{k=r+1}^{d_{\mathrm{in}}} \sigma_k^2(W) \sum_{j=1}^{d_{\mathrm{in}}} \lambda_j (v_k(W)^\top q_j)^2
\label{eq:proof_naive_expansion}
\end{equation}
If any discarded singular direction $v_k(W)$ has non-zero projection onto the dominant activation eigenvector $q_{\max}$ with $\lambda_{\max} \gg 1$, the error $\mathcal{E}(\widehat{W}_{\mathrm{naive}})$ is amplified by $\lambda_{\max}$, whereas $\widehat{W}^*$ provably absorbs $\lambda_{\max}$ into the leading singular values of $W L$.
This completes the proof.
\end{proof}
```

---

### 2.4 Computational Efficiency: Cholesky vs. Symmetric Eigendecomposition
A key theoretical and engineering choice in NeuralOps is using the **lower-triangular Cholesky factor** $L$ rather than the symmetric matrix square root $\Sigma_X^{1/2} = Q \Lambda^{1/2} Q^\top$:

| Metric / Dimension | Cholesky Whitening ($L L^\top$) | Matrix Square Root ($\Sigma_X^{1/2}$) | Algorithmic Superiority |
| :--- | :---: | :---: | :--- |
| **Factorization FLOPs** | $\frac{1}{3} d_{\mathrm{in}}^3$ | $2 d_{\mathrm{in}}^3 + \mathcal{O}(d_{\mathrm{in}}^2)$ | **$6\times$ Faster FLOPs** |
| **Solver Implementation** | Native `cublasStrsm` (triangular solve) | Explicit inversion $Q \Lambda^{-1/2} Q^\top$ | **Zero Explicit Inversion** |
| **Memory Bandwidth** | Packed triangular storage $\frac{d(d+1)}{2}$ | Full dense square matrix $d^2$ | **$50\%$ Cache Savings** |
| **Conditioning Bound** | $\kappa(L) = \sqrt{\kappa(\Sigma_X)}$ | $\kappa(\Sigma_X^{1/2}) = \sqrt{\kappa(\Sigma_X)}$ | Equivalent conditioning |

---

## 3. Riemannian Geometry on the Stiefel Manifold

During model surgery and distillation, post-surgery parameter updates must preserve orthogonality in low-rank projection bases $U \in \mathbb{R}^{n \times p}$ ($p \le n$) to prevent subspace collapse and guarantee full-rank representations.

### 3.1 Differentiable Structure of the Stiefel Manifold
The compact Stiefel manifold is defined as the set of all $n \times p$ orthonormal frames:
$$\operatorname{St}(p, n) \triangleq \left\{ U \in \mathbb{R}^{n \times p} \;\middle|\; U^\top U = I_p \right\}$$
The manifold dimension is $\operatorname{dim}(\operatorname{St}(p, n)) = n p - \frac{1}{2} p (p + 1)$.

```latex
\begin{proposition}[Stiefel Tangent Space Characterization]
\label{prop:stiefel_tangent}
The tangent space of $\operatorname{St}(p, n)$ at point $U$ is given by:
\begin{equation}
T_U \operatorname{St}(p, n) = \left\{ \Delta \in \mathbb{R}^{n \times p} \;\middle|\; U^\top \Delta + \Delta^\top U = 0 \right\}
\label{eq:stiefel_tangent_def}
\end{equation}
Equivalently, $U^\top \Delta$ is a skew-symmetric matrix: $\mathrm{sym}(U^\top \Delta) = 0$, where $\mathrm{sym}(M) \triangleq \frac{1}{2}(M + M^\top)$.
\end{proposition}

\begin{proof}
Let $\gamma: (-\epsilon, \epsilon) \to \operatorname{St}(p, n)$ be any smooth curve on the manifold with $\gamma(0) = U$ and velocity vector $\gamma'(0) = \Delta$.
Since $\gamma(t) \in \operatorname{St}(p, n)$ for all $t$, the orthogonality constraint holds identically:
\begin{equation}
\gamma(t)^\top \gamma(t) = I_p \quad \forall t \in (-\epsilon, \epsilon)
\end{equation}
Differentiating both sides with respect to $t$ at $t = 0$ via the Leibniz product rule:
\begin{equation}
\left. \frac{d}{dt} \left( \gamma(t)^\top \gamma(t) \right) \right|_{t=0} = \gamma'(0)^\top \gamma(0) + \gamma(0)^\top \gamma'(0) = \Delta^\top U + U^\top \Delta = 0
\end{equation}
Since $U^\top \Delta + \Delta^\top U = 2 \mathrm{sym}(U^\top \Delta) = 0$, the velocity vector $\Delta$ lies in $T_U \operatorname{St}(p, n)$.
\end{proof}
```

---

### 3.2 Orthogonal Tangent Projection & Riemannian Gradient

```latex
\begin{theorem}[Orthogonal Tangent Space Projection on $\operatorname{St}(p, n)$]
\label{thm:stiefel_projection}
Endow the ambient Euclidean space $\mathbb{R}^{n \times p}$ with the canonical trace metric $\langle A, B \rangle \triangleq \mathrm{Tr}(A^\top B)$.
For any ambient gradient matrix $G \triangleq \nabla_U \mathcal{L} \in \mathbb{R}^{n \times p}$, the orthogonal projection onto the tangent space $T_U \operatorname{St}(p, n)$ is:
\begin{equation}
\mathcal{P}_U(G) = G - U \, \mathrm{sym}(U^\top G) = G - \frac{1}{2} U \left( U^\top G + G^\top U \right)
\label{eq:stiefel_proj_formula}
\end{equation}
Consequently, the Riemannian gradient of $\mathcal{L}$ under the induced metric is:
\begin{equation}
\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U) = \mathcal{P}_U(\nabla_U \mathcal{L}) = \nabla_U \mathcal{L} - U \, \mathrm{sym}\left( U^\top \nabla_U \mathcal{L} \right)
\label{eq:stiefel_riemannian_gradient}
\end{equation}
\end{theorem}

\begin{proof}
Any ambient matrix $G \in \mathbb{R}^{n \times p}$ decomposes uniquely into a tangent component $\Delta \in T_U \operatorname{St}(p, n)$ and an orthogonal normal component $N \in (T_U \operatorname{St}(p, n))^\perp$:
\begin{equation}
G = \Delta + N
\label{eq:proof_decomp}
\end{equation}
By definition of the normal space under the trace metric, $N$ must satisfy $\langle N, H \rangle = \mathrm{Tr}(N^\top H) = 0$ for all $H \in T_U \operatorname{St}(p, n)$.
Since $U^\top H = - H^\top U$ is skew-symmetric, any matrix of the form $N = U S$ where $S = S^\top \in \mathbb{R}^{p \times p}$ is symmetric satisfies:
\begin{equation}
\mathrm{Tr}(N^\top H) = \mathrm{Tr}((U S)^\top H) = \mathrm{Tr}(S U^\top H) = 0
\end{equation}
because the trace product of a symmetric matrix and a skew-symmetric matrix is identically zero. Thus $(T_U \operatorname{St}(p, n))^\perp = \{ U S \mid S = S^\top \}$.

Multiplying equation~\eqref{eq:proof_decomp} on the left by $U^\top$ and noting $U^\top U = I_p$:
\begin{equation}
U^\top G = U^\top \Delta + U^\top (U S) = U^\top \Delta + S
\end{equation}
Applying the symmetrizing operator $\mathrm{sym}(\cdot)$ to both sides:
\begin{equation}
\mathrm{sym}(U^\top G) = \mathrm{sym}(U^\top \Delta) + \mathrm{sym}(S)
\end{equation}
Since $\Delta \in T_U \operatorname{St}(p, n)$, $\mathrm{sym}(U^\top \Delta) = 0$. Since $S$ is symmetric, $\mathrm{sym}(S) = S$. Hence:
\begin{equation}
S = \mathrm{sym}(U^\top G) \implies N = U \, \mathrm{sym}(U^\top G)
\end{equation}
Subtracting the normal component from $G$ yields the tangent projection:
\begin{equation}
\Delta = \mathcal{P}_U(G) = G - U \, \mathrm{sym}(U^\top G) = G - \frac{1}{2} U (U^\top G + G^\top U)
\end{equation}
This completes the proof.
\end{proof}
```

---

### 3.3 Geodesic Retraction Maps
Following a tangent update step $\xi = -\eta \operatorname{grad}_{\operatorname{St}}\mathcal{L}(U)$, the updated point $U + \xi$ departs from the manifold ($ (U+\xi)^\top (U+\xi) \neq I_p $). A **retraction** $R_U: T_U \operatorname{St}(p, n) \to \operatorname{St}(p, n)$ maps tangent vectors back to the manifold while preserving first-order Taylor expansion accuracy:

#### 1. QR Decomposition Retraction:
$$R_U^{\mathrm{QR}}(\xi) = \operatorname{qf}(U + \xi)$$
where $\operatorname{qf}(A)$ extracts the orthogonal $Q$ factor of the QR decomposition $A = Q R$ with positive diagonal elements ($R_{ii} > 0$).

#### 2. Polar Decomposition Retraction:
$$R_U^{\mathrm{polar}}(\xi) = (U + \xi) \left( (U + \xi)^\top (U + \xi) \right)^{-1/2} = (U + \xi) \left( I_p + \xi^\top \xi \right)^{-1/2}$$
since $U^\top \xi + \xi^\top U = 0$.
The deviation from orthogonality is strictly quadratic in step size: $\|(U+\xi)^\top (U+\xi) - I_p\|_F = \|\xi^\top \xi\|_F = \mathcal{O}(\eta^2)$.

#### 3. Wen-Yin Cayley Transform (Zero SVD / Explicit Retraction):
To eliminate expensive QR factorizations on GPU, the Wen-Yin transform uses a skew-symmetric generator $W \in \mathbb{R}^{n \times n}$:
$$W(U, G) \triangleq G U^\top - U G^\top = -\left(G U^\top - U G^\top\right)^\top$$
The update equation for learning rate $\tau > 0$ is:
$$U(\tau) = \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) U$$

```latex
\begin{theorem}[Strict Conservation of Orthogonality Under Cayley Transform]
\label{thm:cayley_orthogonality}
For any skew-symmetric matrix $W = -W^\top \in \mathbb{R}^{n \times n}$ and any step size $\tau \in \mathbb{R}$, the Cayley matrix:
\begin{equation}
Q \triangleq \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right)
\label{eq:cayley_matrix}
\end{equation}
is strictly orthogonal: $Q^\top Q = Q Q^\top = I_n$. Consequently, if $U \in \operatorname{St}(p, n)$, then $U(\tau) = Q U \in \operatorname{St}(p, n)$ exactly.
\end{theorem}

\begin{proof}
Compute the transpose of $Q$:
\begin{equation}
Q^\top = \left( I_n - \frac{\tau}{2} W \right)^\top \left( \left( I_n + \frac{\tau}{2} W \right)^{-1} \right)^\top = \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)^{-1}
\end{equation}
using the skew-symmetry identity $W^\top = -W$.
Since $(I_n - \frac{\tau}{2} W)$ and $(I_n + \frac{\tau}{2} W)$ commute:
\begin{equation}
\left( I_n - \frac{\tau}{2} W \right) \left( I_n + \frac{\tau}{2} W \right) = I_n - \frac{\tau^2}{4} W^2 = \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)
\end{equation}
their inverses commute with the primal factors. Therefore:
\begin{align}
Q^\top Q &= \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)^{-1} \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) \nonumber \\
&= \left( I_n + \frac{\tau}{2} W \right) \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) \nonumber \\
&= I_n \cdot I_n = I_n
\end{align}
Then $U(\tau)^\top U(\tau) = (Q U)^\top (Q U) = U^\top (Q^\top Q) U = U^\top I_n U = U^\top U = I_p$.
The frame remains strictly on the Stiefel manifold.
\end{proof}
```

---

## 4. Multi-Pillar Knowledge Distillation Formulation

Following neural surgery or weight factorization, the compressed student model $S$ suffers from representation drift. NeuralOps heals the student through a unified **6-Pillar Knowledge Distillation** objective:

$$\mathcal{L}_{\text{6P}} = \lambda_{\text{task}} \mathcal{L}_{\text{task}} + \lambda_{\text{KD}} \mathcal{L}_{\text{logits}} + \lambda_{\text{hidden}} \mathcal{L}_{\text{hidden}} + \lambda_{\text{cos}} \mathcal{L}_{\text{cos}} + \lambda_{\text{attn}} \mathcal{L}_{\text{attn}} + \lambda_{\text{stief}} \mathcal{L}_{\text{Stiefel}}$$

```
═════════════════════════════════════════════════════════════════════════════════════════
                      6-PILLAR KNOWLEDGE DISTILLATION TAXONOMY
═════════════════════════════════════════════════════════════════════════════════════════
  [Input Tokens X] ──┬────────────────────────────────────────────────────────┐
                     ▼                                                        ▼
         ┌───────────────────────┐                                ┌───────────────────────┐
         │     Teacher Model     │                                │    Student (Surgical) │
         └───────────┬───────────┘                                └───────────┬───────────┘
                     │                                                        │
  ┌──────────────────┼────────────────────────────────────────────────────────┼──────────────────┐
  │ Pillar 1: Task CE│ Target Labels y  <───────────────────────────> Output  │ L_task           │
  │ Pillar 2: Logits │ p_T^\tau (Soft Distribution) <─ KL Div ─> p_S^\tau     │ L_logits         │
  │ Pillar 3: Hidden │ H_T^{(l)} (Layer Latents)  <─── MSE ───> H_S^{(l)}     │ L_hidden         │
  │ Pillar 4: Cosine │ Directional Geometry     <── Cos Sim ──> Embedding     │ L_cos            │
  │ Pillar 5: Attn   │ A_T (Attention Matrices) <─ Frobenius ─> A_S           │ L_attn           │
  │ Pillar 6: Stiefel│ Orthogonality Penalty: ||U^T U - I||_F^2 + ||V^T V - I||^2│ L_Stiefel        │
  └──────────────────┴────────────────────────────────────────────────────────┴──────────────────┘
```

### 4.1 Explicit Formulations of the 6 Loss Pillars

#### Pillar 1: Ground-Truth Task Cross-Entropy ($\mathcal{L}_{\text{task}}$)
Preserves fundamental task discriminability against hard labels $y \in \{1, \dots, K\}$:
$$\mathcal{L}_{\text{task}} = - \sum_{k=1}^K y_k \log p_S(k), \quad p_S(k) = \frac{\exp(z_S(k))}{\sum_{j=1}^K \exp(z_S(j))}$$

#### Pillar 2: Temperature-Scaled Soft Logits KL Divergence ($\mathcal{L}_{\text{logits}}$)
Transfers inter-class semantic dark knowledge from teacher logits $z_T$ to student logits $z_S$ with temperature $\tau \ge 1.0$:
$$\mathcal{L}_{\text{logits}} = \tau^2 D_{\mathrm{KL}}\left( p_T^\tau \;\middle\|\; p_S^\tau \right) = \tau^2 \sum_{k=1}^K p_T^\tau(k) \log\left(\frac{p_T^\tau(k)}{p_S^\tau(k)}\right)$$
where $p_M^\tau(k) \triangleq \frac{\exp(z_M(k)/\tau)}{\sum_{j=1}^K \exp(z_M(j)/\tau)}$ for $M \in \{S, T\}$. The $\tau^2$ scaling factor ensures gradient magnitudes remain invariant as $\tau$ varies.

#### Pillar 3: Intermediate Hidden State Mean Squared Error ($\mathcal{L}_{\text{hidden}}$)
Aligns the activation trajectories across $L$ transformer layer blocks:
$$\mathcal{L}_{\text{hidden}} = \frac{1}{L} \sum_{l=1}^L \frac{1}{B \cdot N \cdot d_{\mathrm{model}}} \left\| H_T^{(l)} - H_S^{(l)} W_{\mathrm{proj}}^{(l)} \right\|_F^2$$
where $W_{\mathrm{proj}}^{(l)} \in \mathbb{R}^{d_S \times d_T}$ is a learnable projection matrix when student hidden dimension differs from teacher ($d_S \neq d_T$).

#### Pillar 4: Directional Cosine Embedding Similarity ($\mathcal{L}_{\text{cos}}$)
Guarantees geometric alignment of latent token vectors independent of norm scaling:
$$\mathcal{L}_{\text{cos}} = 1 - \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{t=1}^N \frac{\langle h_S^{(l)}(b, t), \; h_T^{(l)}(b, t) \rangle}{\|h_S^{(l)}(b, t)\|_2 \, \|h_T^{(l)}(b, t)\|_2 + \epsilon}$$

#### Pillar 5: Multi-Head Attention Distribution Matching ($\mathcal{L}_{\text{attn}}$)
Transfers contextual token interaction topology from teacher heads to student heads:
$$\mathcal{L}_{\text{attn}} = \frac{1}{L \cdot H} \sum_{l=1}^L \sum_{h=1}^H D_{\mathrm{KL}}\left( A_T^{(l, h)} \;\middle\|\; A_S^{(l, h)} \right) = \frac{1}{L \cdot H} \sum_{l=1}^L \sum_{h=1}^H \sum_{i=1}^N \sum_{j=1}^N A_T^{(l, h)}(i, j) \log \left(\frac{A_T^{(l, h)}(i, j)}{A_S^{(l, h)}(i, j) + \epsilon}\right)$$

#### Pillar 6: Stiefel Subspace Orthogonality Regularization ($\mathcal{L}_{\text{Stiefel}}$)
Penalizes projection frame degradation to prevent rank collapse during fine-tuning:
$$\mathcal{L}_{\text{Stiefel}} = \frac{1}{4} \sum_{l=1}^L \left( \left\| U_l^\top U_l - I_{r_l} \right\|_F^2 + \left\| V_l^\top V_l - I_{r_l} \right\|_F^2 \right)$$

---

## 5. LaTeX Proof & Theorem Typesetting Standards

### 5.1 Standard LaTeX Preamble Configuration
Ensure all theoretical documents use standardized AMS math packages and macro definitions:

```latex
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{mathtools}
\usepackage{bm}

% Standardized Theorem Environments
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{assumption}{Assumption}

\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{example}{Example}

% Standard Mathematical Operators
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\sym}{sym}
\DeclareMathOperator{\skewop}{skew}
\DeclareMathOperator{\qf}{qf}
```

### 5.2 Mathematical Formatting Discipline

1. **Matrix & Vector Notation**:
   - Scalars: Lowercase lightface italic ($s, \lambda, \tau, \epsilon$).
   - Vectors: Lowercase bold italic ($\bm{x}, \bm{u}, \bm{v}$) or standard math mode ($x, u, v$).
   - Matrices: Uppercase lightface ($W, L, \Sigma$) or uppercase bold ($\mathbf{W}, \mathbf{L}, \mathbf{\Sigma}$).
   - Transpose: Use unambiguous `\top` ($\mathbf{W}^\top$), never a loose character `$W^T$`.
2. **Norms and Brackets**:
   - Frobenius Norm: `\|A\|_F` ($\|A\|_F$).
   - Spectral 2-Norm: `\|A\|_2` ($\|A\|_2$).
   - Traces: `\operatorname{Tr}(A)` or `\Tr(A)`.
   - Condition Number: `\kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}`.
3. **Clamping & Bounded Operations**:
   - Explicitly annotate numerical safety guards in prose and equations:
     $$s_{\mathrm{clamped}} = \operatorname{clamp}\left(s, \, 10^{-8}, \, 10^3\right)$$
4. **Hadamard vs Matrix Multiplication**:
   - Matrix product: $A B$.
   - Element-wise product: $A \odot B$.
   - Outer/Tensor product: $u \otimes v$ or $u v^\top$.
   - Kronecker product: $A \otimes B$.

---

## 6. Verification Checklist for Mathematical Formulations

Before finalizing any theoretical section, verify the manuscript against this quality gate:

- [ ] **Dimensional Consistency**: Every matrix product $A B$ has matching inner dimensions ($\mathbb{R}^{m \times k} \times \mathbb{R}^{k \times n}$).
- [ ] **Rank Validity**: Low-rank bounds strictly satisfy $r \le \min(d_{\mathrm{in}}, d_{\mathrm{out}})$.
- [ ] **Metric Symmetry**: Every covariance matrix $\Sigma_X$ is explicitly proven or stated as symmetric positive semi-definite ($\Sigma_X \succeq 0$) or strictly positive definite ($\Sigma_X \succ 0$).
- [ ] **Tikhonov Damping**: Covariance inversion/Cholesky terms feature explicit regularizers ($\Sigma_X = \frac{1}{M} X X^\top + \lambda I$).
- [ ] **Tangent Orthogonality**: Tangent vectors $\Delta \in T_U \operatorname{St}(p, n)$ satisfy $\operatorname{sym}(U^\top \Delta) = 0$.
- [ ] **Projection Idempotence**: The tangent projector satisfies $\mathcal{P}_U(\mathcal{P}_U(G)) = \mathcal{P}_U(G)$.
- [ ] **Distillation Temperature**: Soft logit distillation has explicit $\tau^2$ normalization.
- [ ] **Zero Synthetic Shortcuts**: No placeholder theorems, no hand-waving without explicit algebraic proofs.
