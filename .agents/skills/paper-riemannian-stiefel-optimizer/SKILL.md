---
name: paper-riemannian-stiefel-optimizer
description: "Author mathematically rigorous sections on Stiefel manifold Riemannian geometry, orthogonal projection operators, retraction maps, and Riemannian Adam/SGD optimization for long research papers"
version: "1.0.0"
tags: ["neuralops", "paper", "math", "stiefel", "riemannian", "optimization", "geometry"]
---

# Skill: Riemannian Optimization on the Stiefel Manifold for Research Manuscripts

## Overview & Scope
Use this skill when drafting, verifying, formalizing, or reviewing publication-grade mathematical sections, lemmas, theorems, and algorithm descriptions concerning **Riemannian optimization on the Stiefel manifold** $\operatorname{St}(p, n)$ for top-tier AI and systems conferences (NeurIPS, ICLR, ICML, MLSys) and doctoral dissertations.

In deep learning model surgery, low-rank factorization, and structured compression (such as NeuralOps activation-aware SVD and linear attention replacement), parameters frequently factor into semi-orthogonal projection bases $U \in \mathbb{R}^{n \times p}$ ($U^\top U = I_p$) and adaptive core weights. Optimizing these bases using naive Euclidean gradient descent or standard Adam violates the orthogonality constraint after a single gradient step, inducing:
1. **Subspace Collapse**: Loss of linear independence among basis columns, degenerating effective matrix rank.
2. **Singular Value Drift**: Deviation of basis singular values from unity, destroying spectral norm bounds.
3. **Severe Ill-Conditioning**: Condition number degradation from $\kappa(U) = 1$ to $\kappa(U) \gg 10^4$, causing explosive gradient oscillations.

This skill equips agents with the exact mathematical formulation, step-by-step proofs, AMS-LaTeX theorem environments, algorithmic pseudocode, and production PyTorch references to author rigorous papers on Riemannian manifold optimization.

---

## 1. Differential Geometry of the Stiefel Manifold

```
═════════════════════════════════════════════════════════════════════════════════════════
                GEOMETRIC STRUCTURE OF THE COMPACT STIEFEL MANIFOLD
═════════════════════════════════════════════════════════════════════════════════════════

                 Ambient Euclidean Space: R^{n x p}
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
        │                       T_U St(p, n)                          │
        │           ┌───────────────────────────────────┐             │
        │           │  Tangent Space: U^T ξ + ξ^T U = 0 │             │
        │           │  dim = np - 1/2 p(p+1)            │             │
        │           └─────────────────┬─────────────────┘             │
        │                             │                               │
        │                             │ Tangent Projection            │
        │                             │ P_U(G) = G - U sym(U^T G)     │
        │                             ▼                               │
        │                       * U in St(p, n)                       │
        │                      / \                                    │
        │                     /   \                                   │
        │                    /     \   Retraction: R_U(ξ) in St(p, n) │
        │                   /       \  (QR, Polar, Cayley)            │
        │                  /         ▼                                │
        │                 /           * U_+ in St(p, n)               │
        │       ═════════╩═════════════╩════════════════              │
        │             Stiefel Manifold: St(p, n)                      │
        │             { U in R^{n x p} | U^T U = I_p }                │
        └─────────────────────────────────────────────────────────────┘
═════════════════════════════════════════════════════════════════════════════════════════
```

### 1.1 Formal Definition and Topological Structure
Let $n, p \in \mathbb{N}$ with $p \le n$. The **compact Stiefel manifold** $\operatorname{St}(p, n)$ is the embedded Riemannian submanifold of $\mathbb{R}^{n \times p}$ consisting of all $n \times p$ column-orthonormal frames:
$$\operatorname{St}(p, n) \triangleq \left\{ U \in \mathbb{R}^{n \times p} \;\middle|\; U^\top U = I_p \right\}$$

When $p = 1$, $\operatorname{St}(1, n)$ is the unit hypersphere $\mathbb{S}^{n-1} \subset \mathbb{R}^n$.
When $p = n$, $\operatorname{St}(n, n)$ is the orthogonal Lie group $\mathrm{O}(n)$.
For general $1 < p < n$, $\operatorname{St}(p, n)$ is a smooth, compact, connected (for $p < n$), real differential manifold.

#### Dimension Formula:
The constraint $U^\top U = I_p$ is symmetric and imposes $\frac{1}{2} p (p + 1)$ independent algebraic constraints on the $n p$ ambient variables. Consequently:
$$\dim\left(\operatorname{St}(p, n)\right) = n p - \frac{1}{2} p (p + 1) = p \left( n - \frac{p + 1}{2} \right)$$

#### Row-Stiefel Variant ($n < p$):
In dual compression regimes (such as right-singular matrices $V \in \mathbb{R}^{p \times n}$ where $n > p$), the constraint is imposed on rows:
$$\operatorname{St}_{\text{row}}(n, p) \triangleq \left\{ V \in \mathbb{R}^{n \times p} \;\middle|\; V V^\top = I_n \right\}, \quad n \le p$$
where $\dim(\operatorname{St}_{\text{row}}(n, p)) = n p - \frac{1}{2} n (n + 1)$.

---

### 1.2 Metric Structures on $\operatorname{St}(p, n)$
There are two canonical Riemannian metrics applied to $\operatorname{St}(p, n)$:

#### 1. Embedded Euclidean (Trace) Metric:
The standard metric induced directly from the ambient embedding space $\mathbb{R}^{n \times p}$:
$$g_U^{\text{euc}}(\xi_1, \xi_2) \triangleq \langle \xi_1, \xi_2 \rangle = \operatorname{Tr}\left(\xi_1^\top \xi_2\right), \quad \forall \xi_1, \xi_2 \in T_U \operatorname{St}(p, n)$$
This metric is computationally lightweight and is the standard metric utilized in NeuralOps and deep learning surgery.

#### 2. Canonical (Quotient) Metric:
The metric derived from viewing $\operatorname{St}(p, n)$ as the homogeneous quotient space $\mathrm{O}(n) / \mathrm{O}(n - p)$:
$$g_U^{\text{can}}(\xi_1, \xi_2) \triangleq \operatorname{Tr}\left(\xi_1^\top \left( I_n - \frac{1}{2} U U^\top \right) \xi_2\right)$$
The canonical metric penalizes normal motions differently, but the embedded Euclidean metric provides strictly superior throughput on GPU hardware while maintaining identical first-order convergence guarantees.

---

## 2. Tangent Space Derivation & Geometric Decomposition

### 2.1 Tangent Space Characterization

```latex
\begin{proposition}[Stiefel Tangent Space]
\label{prop:stiefel_tangent_space}
Let $U \in \operatorname{St}(p, n)$. The tangent space to $\operatorname{St}(p, n)$ at $U$, denoted $T_U \operatorname{St}(p, n)$, is given explicitly by:
\begin{equation}
T_U \operatorname{St}(p, n) = \left\{ \xi \in \mathbb{R}^{n \times p} \;\middle|\; U^\top \xi + \xi^\top U = 0 \right\}
\label{eq:tangent_space_def}
\end{equation}
Equivalently, $\xi \in T_U \operatorname{St}(p, n)$ if and only if $U^\top \xi$ is a skew-symmetric matrix in the Lie algebra $\mathfrak{so}(p)$:
\begin{equation}
\operatorname{sym}(U^\top \xi) = 0, \quad \text{where } \operatorname{sym}(M) \triangleq \frac{1}{2}\left(M + M^\top\right)
\label{eq:tangent_skew_symmetric}
\end{equation}
\end{proposition}

\begin{proof}
Let $\gamma: (-\epsilon, \epsilon) \to \operatorname{St}(p, n)$ be an arbitrary smooth curve passing through $\gamma(0) = U$ with initial velocity vector $\gamma'(0) = \xi \in T_U \operatorname{St}(p, n)$.
Because the curve remains entirely within the manifold for all $t \in (-\epsilon, \epsilon)$, the defining orthogonality relation holds identically:
\begin{equation}
\gamma(t)^\top \gamma(t) = I_p, \quad \forall t \in (-\epsilon, \epsilon)
\end{equation}
Differentiating both sides with respect to $t$ at $t = 0$ via the matrix Leibniz product rule:
\begin{equation}
\left. \frac{d}{dt} \left( \gamma(t)^\top \gamma(t) \right) \right|_{t=0} = \gamma'(0)^\top \gamma(0) + \gamma(0)^\top \gamma'(0) = \xi^\top U + U^\top \xi = 0
\end{equation}
Rearranging terms yields $U^\top \xi = - (U^\top \xi)^\top$, confirming that $U^\top \xi \in \mathfrak{so}(p)$ is skew-symmetric, which proves equation~\eqref{eq:tangent_space_def}.
\end{proof}
```

### 2.2 Tangent Vector Direct Sum Decomposition
Any tangent vector $\xi \in T_U \operatorname{St}(p, n)$ can be parameterized uniquely by decomposing $\mathbb{R}^n$ into the range space of $U$ and its orthogonal complement:
$$\xi = U \Omega + U_\perp K$$
where:
- $\Omega \triangleq U^\top \xi \in \mathfrak{so}(p)$ is a $p \times p$ skew-symmetric matrix ($\Omega^\top = -\Omega$).
- $U_\perp \in \operatorname{St}(n - p, n)$ satisfies $U^\top U_\perp = 0$ and $U U^\top + U_\perp U_\perp^\top = I_n$.
- $K \triangleq U_\perp^\top \xi \in \mathbb{R}^{(n - p) \times p}$ is an unconstrained arbitrary coefficient matrix.

The degrees of freedom match the dimension exactly:
$$\dim(\mathfrak{so}(p)) + \dim(\mathbb{R}^{(n - p) \times p}) = \frac{p(p - 1)}{2} + (n - p) p = n p - \frac{p(p + 1)}{2} = \dim(\operatorname{St}(p, n))$$

### 2.3 Normal Space Characterization
Under the ambient trace inner product $\langle A, B \rangle = \operatorname{Tr}(A^\top B)$, the normal space $(T_U \operatorname{St}(p, n))^\perp$ is the orthogonal complement of the tangent space in $\mathbb{R}^{n \times p}$:
$$(T_U \operatorname{St}(p, n))^\perp = \left\{ N \in \mathbb{R}^{n \times p} \;\middle|\; \langle N, \xi \rangle = 0, \; \forall \xi \in T_U \operatorname{St}(p, n) \right\}$$

```latex
\begin{proposition}[Stiefel Normal Space]
\label{prop:stiefel_normal_space}
The normal space to $\operatorname{St}(p, n)$ at $U$ under the Euclidean trace metric is:
\begin{equation}
(T_U \operatorname{St}(p, n))^\perp = \left\{ U S \;\middle|\; S = S^\top \in \operatorname{Sym}(p) \right\}
\label{eq:normal_space_def}
\end{equation}
where $\operatorname{Sym}(p)$ denotes the vector space of $p \times p$ symmetric matrices.
\end{proposition}

\begin{proof}
Let $N = U S$ with $S = S^\top$. For any tangent vector $\xi \in T_U \operatorname{St}(p, n)$, compute the inner product:
\begin{equation}
\langle N, \xi \rangle = \operatorname{Tr}\left( (U S)^\top \xi \right) = \operatorname{Tr}\left( S U^\top \xi \right)
\end{equation}
Because $S$ is symmetric ($S^\top = S$) and $U^\top \xi$ is skew-symmetric ($(U^\top \xi)^\top = -U^\top \xi$), their trace product vanishes identically:
\begin{equation}
\operatorname{Tr}\left( S U^\top \xi \right) = \operatorname{Tr}\left( (S U^\top \xi)^\top \right) = \operatorname{Tr}\left( (U^\top \xi)^\top S^\top \right) = -\operatorname{Tr}\left( U^\top \xi S \right) = -\operatorname{Tr}\left( S U^\top \xi \right) = 0
\end{equation}
Thus $\{ U S \mid S \in \operatorname{Sym}(p) \} \subseteq (T_U \operatorname{St}(p, n))^\perp$.
By dimension counting:
\begin{equation}
\dim\left((T_U \operatorname{St}(p, n))^\perp\right) = n p - \left( n p - \frac{p(p+1)}{2} \right) = \frac{p(p+1)}{2} = \dim(\operatorname{Sym}(p))
\end{equation}
Since the dimensions coincide, equation~\eqref{eq:normal_space_def} holds with equality.
\end{proof}
```

---

## 3. Orthogonal Projection & Riemannian Gradient

### 3.1 Closed-Form Tangent Projection Operator
Let $\mathcal{L}: \mathbb{R}^{n \times p} \to \mathbb{R}$ be a smooth loss function, and let $G \triangleq \nabla_U \mathcal{L} \in \mathbb{R}^{n \times p}$ denote the ambient Euclidean gradient. The **Riemannian gradient** $\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U)$ is the unique tangent vector in $T_U \operatorname{St}(p, n)$ satisfying:
$$\langle \operatorname{grad}_{\operatorname{St}} \mathcal{L}(U), \xi \rangle = \mathcal{D}\mathcal{L}(U)[\xi] = \langle G, \xi \rangle, \quad \forall \xi \in T_U \operatorname{St}(p, n)$$

Under the ambient Euclidean metric, the Riemannian gradient is simply the orthogonal projection of $G$ onto $T_U \operatorname{St}(p, n)$.

```latex
\begin{theorem}[Orthogonal Tangent Projection on the Stiefel Manifold]
\label{thm:stiefel_projection}
Endow $\mathbb{R}^{n \times p}$ with the Euclidean trace metric $\langle A, B \rangle = \operatorname{Tr}(A^\top B)$. For any ambient matrix $Z \in \mathbb{R}^{n \times p}$, the orthogonal projection operator $\mathcal{P}_U: \mathbb{R}^{n \times p} \to T_U \operatorname{St}(p, n)$ is given in closed form by:
\begin{equation}
\mathcal{P}_U(Z) = Z - U \, \operatorname{sym}\left(U^\top Z\right) = Z - \frac{1}{2} U \left( U^\top Z + Z^\top U \right)
\label{eq:stiefel_projection_formula}
\end{equation}
Consequently, the Riemannian gradient of a objective $\mathcal{L}$ at $U \in \operatorname{St}(p, n)$ is:
\begin{equation}
\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U) = \mathcal{P}_U(\nabla_U \mathcal{L}) = \nabla_U \mathcal{L} - \frac{1}{2} U \left( U^\top \nabla_U \mathcal{L} + (\nabla_U \mathcal{L})^\top U \right)
\label{eq:riemannian_gradient_formula}
\end{equation}
\end{theorem}

\begin{proof}
By the direct sum decomposition of the ambient space $\mathbb{R}^{n \times p} = T_U \operatorname{St}(p, n) \oplus (T_U \operatorname{St}(p, n))^\perp$, any matrix $Z \in \mathbb{R}^{n \times p}$ decomposes uniquely as:
\begin{equation}
Z = \xi + N
\label{eq:z_decomp}
\end{equation}
where $\xi \in T_U \operatorname{St}(p, n)$ is tangent and $N \in (T_U \operatorname{St}(p, n))^\perp$ is normal.
By Proposition~\ref{prop:stiefel_normal_space}, the normal component must take the form $N = U S$ for some symmetric matrix $S \in \operatorname{Sym}(p)$.
Substituting $N = U S$ into~\eqref{eq:z_decomp} and multiplying both sides on the left by $U^\top$:
\begin{equation}
U^\top Z = U^\top \xi + U^\top (U S) = U^\top \xi + S
\end{equation}
since $U^\top U = I_p$.
Applying the linear symmetrizing operator $\operatorname{sym}(\cdot)$ to both sides:
\begin{equation}
\operatorname{sym}\left(U^\top Z\right) = \operatorname{sym}\left(U^\top \xi\right) + \operatorname{sym}(S)
\end{equation}
Because $\xi \in T_U \operatorname{St}(p, n)$, Proposition~\ref{prop:stiefel_tangent_space} gives $\operatorname{sym}(U^\top \xi) = 0$.
Because $S$ is symmetric, $\operatorname{sym}(S) = S$. Thus:
\begin{equation}
S = \operatorname{sym}\left(U^\top Z\right) \implies N = U \, \operatorname{sym}\left(U^\top Z\right)
\end{equation}
Subtracting the normal component $N$ from $Z$ yields the unique tangent projection:
\begin{equation}
\xi = \mathcal{P}_U(Z) = Z - N = Z - U \, \operatorname{sym}\left(U^\top Z\right) = Z - \frac{1}{2} U \left( U^\top Z + Z^\top U \right)
\end{equation}
This completes the proof.
\end{proof}
```

### 3.2 Dual Row-Stiefel Projection ($n < p$)
When optimizing weight matrices with more columns than rows ($U \in \mathbb{R}^{n \times p}$ with $n < p$ and $U U^\top = I_n$), the orthogonal projection projects onto $T_U \operatorname{St}_{\text{row}}(n, p)$:
$$\mathcal{P}_U^{\text{row}}(Z) = Z - \operatorname{sym}\left(Z U^\top\right) U = Z - \frac{1}{2} \left( Z U^\top + U Z^\top \right) U$$

### 3.3 Projection Properties
1. **Idempotency**: $\mathcal{P}_U(\mathcal{P}_U(Z)) = \mathcal{P}_U(Z)$ for all $Z \in \mathbb{R}^{n \times p}$.
2. **Self-Adjointness**: $\langle \mathcal{P}_U(A), B \rangle = \langle A, \mathcal{P}_U(B) \rangle$.
3. **Non-Expansiveness**: $\|\mathcal{P}_U(Z)\|_F \le \|Z\|_F$ with equality if and only if $Z \in T_U \operatorname{St}(p, n)$.

---

## 4. Retraction Maps onto the Stiefel Manifold

Following a tangent update step $\xi \in T_U \operatorname{St}(p, n)$, moving linearly to $U + \xi$ violates orthogonality:
$$(U + \xi)^\top (U + \xi) = I_p + U^\top \xi + \xi^\top U + \xi^\top \xi = I_p + \xi^\top \xi \neq I_p$$
A **retraction map** $R_U: T_U \operatorname{St}(p, n) \to \operatorname{St}(p, n)$ pulls tangent vectors back onto the manifold surface.

```latex
\begin{definition}[Riemannian Retraction]
\label{def:retraction}
A smooth mapping $R: T \operatorname{St}(p, n) \to \operatorname{St}(p, n)$ is a retraction if for every $U \in \operatorname{St}(p, n)$, the restriction $R_U \triangleq R(U, \cdot): T_U \operatorname{St}(p, n) \to \operatorname{St}(p, n)$ satisfies:
\begin{enumerate}
    \item $R_U(0) = U$ (Zero preservation).
    \item $\left. \frac{d}{dt} R_U(t \xi) \right|_{t=0} = \xi$ for all $\xi \in T_U \operatorname{St}(p, n)$ (Local rigidity / first-order agreement).
\end{enumerate}
\end{definition}
```

---

### 4.1 Retraction 1: QR Decomposition Retraction
The QR retraction extracts the orthonormal factor from the thin QR decomposition of $U + \xi$:
$$R_U^{\text{QR}}(\xi) \triangleq \operatorname{qf}(U + \xi)$$
where $\operatorname{qf}(M)$ denotes the $Q$ factor in the QR decomposition $M = Q R$ normalized such that the diagonal elements of $R$ are strictly positive: $R_{ii} > 0$.

```latex
\begin{proposition}[QR Retraction Properties]
\label{prop:qr_retraction}
Let $U \in \operatorname{St}(p, n)$ and $\xi \in T_U \operatorname{St}(p, n)$ with $\|\xi\|_2 < 1$. Then:
\begin{enumerate}
    \item $U + \xi$ has full column rank $p$, guaranteeing a unique thin QR decomposition.
    \item $R_U^{\mathrm{QR}}(\xi) \in \operatorname{St}(p, n)$ satisfies the first-order retraction conditions.
    \item The second-order Taylor expansion satisfies:
    \begin{equation}
    R_U^{\mathrm{QR}}(\xi) = U + \xi - \frac{1}{2} U \xi^\top \xi + \mathcal{O}\left(\|\xi\|_F^3\right)
    \end{equation}
\end{enumerate}
\end{proposition}
```

*Implementation detail*: PyTorch's `torch.linalg.qr(..., mode="reduced")` may return diagonal entries in $R$ with negative signs. To preserve continuity, phase normalization must be applied:
$$Q_{\text{canonical}} = Q \cdot \operatorname{diag}\left(\operatorname{sgn}(\operatorname{diag}(R))\right)$$

---

### 4.2 Retraction 2: Polar Decomposition Retraction
The polar retraction computes the nearest orthogonal matrix in the Frobenius norm:
$$R_U^{\text{polar}}(\xi) \triangleq (U + \xi) \left( (U + \xi)^\top (U + \xi) \right)^{-1/2}$$

```latex
\begin{proposition}[Polar Retraction Simplification on Tangent Vectors]
\label{prop:polar_retraction}
For any $\xi \in T_U \operatorname{St}(p, n)$ with $\|\xi\|_2 < 1$, because $U^\top \xi + \xi^\top U = 0$, the Gram matrix simplifies to:
\begin{equation}
(U + \xi)^\top (U + \xi) = U^\top U + U^\top \xi + \xi^\top U + \xi^\top \xi = I_p + \xi^\top \xi
\end{equation}
Hence, the polar retraction is:
\begin{equation}
R_U^{\mathrm{polar}}(\xi) = (U + \xi) \left( I_p + \xi^\top \xi \right)^{-1/2}
\label{eq:polar_simplified}
\end{equation}
Furthermore, $R_U^{\mathrm{polar}}$ is a second-order retraction matching the Riemannian geodesic:
\begin{equation}
R_U^{\mathrm{polar}}(\xi) = U + \xi - \frac{1}{2} U \xi^\top \xi + \mathcal{O}\left(\|\xi\|_F^3\right)
\end{equation}
\end{proposition}
```

*Error bound*: The departure from orthogonality before retraction is strictly bounded:
$$\left\| (U + \xi)^\top (U + \xi) - I_p \right\|_F = \|\xi^\top \xi\|_F \le \|\xi\|_F^2 = \mathcal{O}(\eta^2)$$
For learning rates $\eta \le 10^{-3}$, the deviation is bounded by $10^{-6}$, guaranteeing numerical conditioning.

---

### 4.3 Retraction 3: Cayley Transform Retraction (Wen-Yin Scheme)
To bypass explicit matrix square roots or sequential QR factorizations, the Wen-Yin transform (Wen & Yin, 2013) constructs an explicit skew-symmetric operator $W \in \mathfrak{so}(n)$:
$$W(U, G) \triangleq \mathcal{P}_U(G) U^\top - U \mathcal{P}_U(G)^\top = G U^\top - U G^\top \in \mathbb{R}^{n \times n}$$
where $W^\top = -W$.
The Cayley curve for step size $\tau > 0$ is defined as:
$$U(\tau) = \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) U$$

```latex
\begin{theorem}[Exact Orthogonality Preservation Under Cayley Retraction]
\label{thm:cayley_exact_orthogonality}
For any skew-symmetric matrix $W = -W^\top \in \mathbb{R}^{n \times n}$ and any step size $\tau \in \mathbb{R}$, the Cayley matrix:
\begin{equation}
\mathcal{Q}(\tau) \triangleq \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right)
\label{eq:cayley_operator}
\end{equation}
is an element of the special orthogonal group $\mathrm{SO}(n)$: $\mathcal{Q}(\tau)^\top \mathcal{Q}(\tau) = I_n$ and $\det(\mathcal{Q}(\tau)) = +1$.
Consequently, if $U \in \operatorname{St}(p, n)$, then $U(\tau) = \mathcal{Q}(\tau) U \in \operatorname{St}(p, n)$ exactly.
\end{theorem}

\begin{proof}
Taking the transpose of $\mathcal{Q}(\tau)$ using the skew-symmetry property $W^\top = -W$:
\begin{equation}
\mathcal{Q}(\tau)^\top = \left( I_n - \frac{\tau}{2} W \right)^\top \left( \left( I_n + \frac{\tau}{2} W \right)^{-1} \right)^\top = \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)^{-1}
\end{equation}
Notice that the two factors $(I_n - \frac{\tau}{2} W)$ and $(I_n + \frac{\tau}{2} W)$ commute:
\begin{equation}
\left( I_n - \frac{\tau}{2} W \right) \left( I_n + \frac{\tau}{2} W \right) = I_n - \frac{\tau^2}{4} W^2 = \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)
\end{equation}
Therefore, their inverses also commute:
\begin{equation}
\left( I_n - \frac{\tau}{2} W \right)^{-1} \left( I_n + \frac{\tau}{2} W \right)^{-1} = \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right)^{-1}
\end{equation}
Now, evaluate the product $\mathcal{Q}(\tau)^\top \mathcal{Q}(\tau)$:
\begin{align}
\mathcal{Q}(\tau)^\top \mathcal{Q}(\tau) &= \left( I_n + \frac{\tau}{2} W \right) \left( I_n - \frac{\tau}{2} W \right)^{-1} \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) \nonumber \\
&= \left( I_n + \frac{\tau}{2} W \right) \left( I_n + \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right)^{-1} \left( I_n - \frac{\tau}{2} W \right) \nonumber \\
&= I_n \cdot I_n = I_n
\end{align}
Finally, for the updated parameter $U(\tau) = \mathcal{Q}(\tau) U$:
\begin{equation}
U(\tau)^\top U(\tau) = U^\top \mathcal{Q}(\tau)^\top \mathcal{Q}(\tau) U = U^\top I_n U = U^\top U = I_p
\end{equation}
Thus, $U(\tau)$ remains strictly on $\operatorname{St}(p, n)$ without any numerical drift.
\end{proof}
```

---

### 4.4 Low-Rank Woodbury Acceleration for Cayley Retraction
A direct evaluation of $(I_n + \frac{\tau}{2} W)^{-1}$ requires an $\mathcal{O}(n^3)$ inversion, which is prohibitive when $n \gg p$ (e.g., $n = 4096, p = 64$).
Because $W = G U^\top - U G^\top$ has low rank $2p$, we factor $W$ as:
$$W = \begin{bmatrix} G & -U \end{bmatrix} \begin{bmatrix} U & G \end{bmatrix}^\top = V_1 V_2^\top \quad \in \mathbb{R}^{n \times n}$$
where $V_1, V_2 \in \mathbb{R}^{n \times 2p}$.

Applying the **Sherman-Morrison-Woodbury theorem**:
$$\left( I_n + \frac{\tau}{2} V_1 V_2^\top \right)^{-1} = I_n - \frac{\tau}{2} V_1 \left( I_{2p} + \frac{\tau}{2} V_2^\top V_1 \right)^{-1} V_2^\top$$
The matrix to invert is now of size $2p \times 2p$ rather than $n \times n$.

```
Complexity Reduction:
Direct Inversion:      O(n^3)               FLOPs
Woodbury Inversion:    O(n p^2 + (2p)^3)    FLOPs
For n=4096, p=64:      Direct: ~68.7 GFLOPs  vs  Woodbury: ~33.5 MFLOPs  (2050x Speedup)
```

---

### 4.5 Comparative Retraction Taxonomy

| Retraction Scheme | Formula | Order | FLOP Complexity | Stability Bound | GPU Hardware Fit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Thin QR ($\operatorname{qf}$)** | $\operatorname{qf}(U - \eta \mathcal{P}_U(G))$ | 1st | $\mathcal{O}(n p^2)$ | Machine $\epsilon$ | Highly optimized via `torch.linalg.qr` |
| **Polar Factor** | $(U+\xi)(I_p + \xi^\top \xi)^{-1/2}$ | 2nd | $\mathcal{O}(n p^2 + p^3)$ | $\mathcal{O}(\eta^2)$ deviation | Excellent for $p \le 128$ (Cholesky solve) |
| **Cayley (Woodbury)** | $(I - \frac{\tau}{2} W)^{-1}(I + \frac{\tau}{2} W) U$ | 2nd | $\mathcal{O}(n p^2 + 8 p^3)$ | Strictly 0 deviation | Best for strict Lie algebra constraints |
| **Geodesic (Exp map)**| $U M \cos(\Sigma) + Q \sin(\Sigma)$ | Exact | $\mathcal{O}(n p^2 + p^3)$ (SVD) | Exact geodesic | Slow on GPU due to thin SVD latency |

---

## 5. Vector Transport & Riemannian Adam/SGD Algorithms

```
═════════════════════════════════════════════════════════════════════════════════════════
                       RIEMANNIAN VECTOR TRANSPORT ON St(p, n)
═════════════════════════════════════════════════════════════════════════════════════════

                 T_U St(p, n)                           T_{U_+} St(p, n)
          ┌─────────────────────────┐             ┌─────────────────────────┐
          │ Momentum m_k            │             │ Transported Momentum    │
          │ Tangent to U_k          │             │ T_{U -> U_+}(m_k)       │
          └────────────┬────────────┘             └────────────▲────────────┘
                       │                                       │
                       │             Retraction R_U(ξ)         │  Orthogonal Projection
                       ├───────────────────────────────────────┤  P_{U_+}(m_k)
                       ▼                                       │
                    * U_k ────────────────────────────────► * U_+
                      \                                       /
                       \                                     /
                        \═══════════════════════════════════/
                                    St(p, n)
═════════════════════════════════════════════════════════════════════════════════════════
```

### 5.1 Vector Transport Operator
In Euclidean space, gradient vectors from different iterations can be added directly: $m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k$.
On a curved Riemannian manifold, the tangent spaces $T_{U_{k-1}} \operatorname{St}(p, n)$ and $T_{U_k} \operatorname{St}(p, n)$ are distinct linear subspaces. Past momentum vectors cannot be added without transporting them into the current tangent space.

A **vector transport** $\mathcal{T}_{U \to U_+}: T_U \operatorname{St}(p, n) \to T_{U_+} \operatorname{St}(p, n)$ linearly maps tangent vectors while preserving tangent space consistency. Under the embedded metric, the canonical choice is the **orthogonal projection vector transport**:
$$\mathcal{T}_{U \to U_+}(\xi) \triangleq \mathcal{P}_{U_+}(\xi) = \xi - U_+ \operatorname{sym}\left(U_+^\top \xi\right)$$

---

### 5.2 Riemannian Stochastic Gradient Descent (R-SGD)
At iteration $k$, given step size $\eta_k$ and momentum parameter $\mu \in [0, 1)$:

```latex
\begin{algorithm}[H]
\caption{Riemannian SGD on the Stiefel Manifold with Momentum}
\label{alg:rsgd_stiefel}
\begin{algorithmic}[1]
\Require Initial point $U_0 \in \operatorname{St}(p, n)$, learning rate $\eta_k$, momentum $\mu \in [0, 1)$.
\State Initialize momentum buffer $M_0 = \mathbf{0} \in T_{U_0} \operatorname{St}(p, n)$.
\For{$k = 0, 1, 2, \dots, K-1$}
    \State Compute ambient Euclidean gradient $G_k = \nabla_U \mathcal{L}(U_k)$.
    \State Project onto tangent space: $g_k = \mathcal{P}_{U_k}(G_k) = G_k - \frac{1}{2} U_k \left(U_k^\top G_k + G_k^\top U_k\right)$.
    \If{$\mu > 0$ and $k > 0$}
        \State Transport momentum: $\widetilde{M}_{k-1} = \mathcal{P}_{U_k}(M_{k-1})$.
        \State Update momentum: $M_k = \mu \widetilde{M}_{k-1} + (1 - \mu) g_k$.
        \State Tangent search direction: $\xi_k = -\eta_k M_k$.
    \Else
        \State Tangent search direction: $\xi_k = -\eta_k g_k$.
    \EndIf
    \State Retract to manifold: $U_{k+1} = R_{U_k}(\xi_k) \in \operatorname{St}(p, n)$.
\EndFor
\State \Return $U_K$.
\end{algorithmic}
\end{algorithm}
```

---

### 5.3 Riemannian Adaptive Moment Estimation (R-Adam / RAMAN)
Riemannian Adam adapts per-coordinate learning rates along the manifold surface:

```latex
\begin{algorithm}[H]
\caption{Riemannian AdamW on the Stiefel Manifold}
\label{alg:radam_stiefel}
\begin{algorithmic}[1]
\Require Initial point $U_0 \in \operatorname{St}(p, n)$, learning rate $\eta$, hyperparameters $\beta_1, \beta_2 \in [0, 1)$, $\epsilon > 0$.
\State Initialize first moment $M_0 = \mathbf{0}_{n \times p}$ and second moment $V_0 = \mathbf{0}_{n \times p}$.
\For{$k = 1, 2, \dots, K$}
    \State Compute ambient gradient $G_k = \nabla_U \mathcal{L}(U_{k-1})$.
    \State Compute Riemannian gradient: $g_k = \mathcal{P}_{U_{k-1}}(G_k)$.
    \State Transport previous first moment: $\widetilde{M}_{k-1} = \mathcal{P}_{U_{k-1}}(M_{k-1})$.
    \State Update biased first moment: $M_k = \beta_1 \widetilde{M}_{k-1} + (1 - \beta_1) g_k$.
    \State Update biased second moment: $V_k = \beta_2 V_{k-1} + (1 - \beta_2) (g_k \odot g_k)$.
    \State Correct first moment bias: $\widehat{M}_k = \frac{M_k}{1 - \beta_1^k}$.
    \State Correct second moment bias: $\widehat{V}_k = \frac{V_k}{1 - \beta_2^k}$.
    \State Compute raw preconditioned step: $\Xi_k = -\frac{\eta}{\sqrt{\widehat{V}_k} + \epsilon} \odot \widehat{M}_k$.
    \State Project preconditioned step into tangent space: $\xi_k = \mathcal{P}_{U_{k-1}}(\Xi_k)$.
    \State Apply retraction: $U_k = R_{U_{k-1}}(\xi_k) \in \operatorname{St}(p, n)$.
\EndFor
\State \Return $U_K$.
\end{algorithmic}
\end{algorithm}
```

> [!IMPORTANT]
> **Preconditioner Tangent Space Preservation**: The Hadamard division $\Xi_k = -\frac{\eta}{\sqrt{\widehat{V}_k} + \epsilon} \odot \widehat{M}_k$ performs non-uniform coordinate rescaling that moves the vector slightly out of $T_{U_{k-1}} \operatorname{St}(p, n)$. Step 11 explicitly reapplies $\mathcal{P}_{U_{k-1}}(\Xi_k)$ to re-orthogonalize the search direction before retraction.

---

## 6. Post-Surgery Orthogonality Recovery Proofs for SVD Basis Weights

When a foundation model undergoes neural surgery (e.g., activation-aware SVD $W \approx U_{\text{down}} V_{\text{up}}$), fine-tuning under standard Euclidean Adam rapidly destroys the singular value spectrum.

### 6.1 Subspace Preservation and Spectral Bounds

```latex
\begin{theorem}[Condition Number Invariance Under Stiefel Optimization]
\label{thm:condition_number_invariance}
Let $U \in \operatorname{St}(p, n)$ be an orthogonal projection basis parameterizing a low-rank layer $h(x) = U \Sigma V^\top x$. If $U_k$ is updated via any valid Riemannian retraction $U_{k+1} = R_{U_k}(\xi_k)$, then for all $k \ge 0$:
\begin{enumerate}
    \item Every singular value of $U_k$ satisfies $\sigma_i(U_k) = 1$ for all $i \in \{1, \dots, p\}$.
    \item The spectral condition number is strictly invariant:
    \begin{equation}
    \kappa(U_k) \triangleq \frac{\sigma_{\max}(U_k)}{\sigma_{\min}(U_k)} \equiv 1.0, \quad \forall k \ge 0
    \label{eq:condition_number_one}
    \end{equation}
    \item The linear subspace rank is preserved without rank collapse: $\operatorname{rank}(U_k) = p$.
\end{enumerate}
\end{theorem}

\begin{proof}
By Definition~\ref{def:retraction}, for any tangent direction $\xi_k \in T_{U_k} \operatorname{St}(p, n)$, the image of the retraction lies strictly on the manifold:
\begin{equation}
U_{k+1} = R_{U_k}(\xi_k) \in \operatorname{St}(p, n) \implies U_{k+1}^\top U_{k+1} = I_p
\end{equation}
Let $\lambda_i(M)$ denote the eigenvalues of a symmetric matrix $M$. Then:
\begin{equation}
\sigma_i^2(U_{k+1}) = \lambda_i\left(U_{k+1}^\top U_{k+1}\right) = \lambda_i(I_p) = 1, \quad \forall i \in \{1, \dots, p\}
\end{equation}
Taking the non-negative square root gives $\sigma_i(U_{k+1}) = 1$ for all $i$.
The condition number is:
\begin{equation}
\kappa(U_{k+1}) = \frac{\max_i \sigma_i(U_{k+1})}{\min_i \sigma_i(U_{k+1})} = \frac{1}{1} = 1.0
\end{equation}
Finally, $\operatorname{rank}(U_{k+1}) = \operatorname{Tr}(I_p) = p$. The basis columns remain strictly linearly independent.
\end{proof}
```

---

### 6.2 Convergence of Riemannian Gradient Descent on $\operatorname{St}(p, n)$

```latex
\begin{theorem}[Linear-Sublinear Convergence on Compact Stiefel Manifolds]
\label{thm:riemannian_convergence}
Let $\mathcal{L}: \operatorname{St}(p, n) \to \mathbb{R}$ be continuously differentiable with $L_{\operatorname{R}}$-Lipschitz continuous Riemannian gradient on $\operatorname{St}(p, n)$:
\begin{equation}
\|\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U) - \mathcal{T}_{V \to U}(\operatorname{grad}_{\operatorname{St}} \mathcal{L}(V))\|_F \le L_{\operatorname{R}} \, \operatorname{dist}_{\operatorname{St}}(U, V)
\end{equation}
Let $U_{k+1} = R_{U_k}(-\eta \operatorname{grad}_{\operatorname{St}} \mathcal{L}(U_k))$ where $R$ is a retraction with second-order boundedness $M_R \ge 0$, and step size satisfies $\eta \in \left(0, \frac{2}{L_{\operatorname{R}} + M_R \sup \|\nabla \mathcal{L}\|}\right)$.
Then:
\begin{enumerate}
    \item The sequence of loss values $\mathcal{L}(U_k)$ is monotonically non-increasing:
    \begin{equation}
    \mathcal{L}(U_{k+1}) - \mathcal{L}(U_k) \le -c \eta \|\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U_k)\|_F^2
    \end{equation}
    for some constant $c > 0$.
    \item The Riemannian gradient norm converges to zero at the optimal sublinear rate:
    \begin{equation}
    \min_{0 \le k \le K} \|\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U_k)\|_F \le \frac{\sqrt{\mathcal{L}(U_0) - \mathcal{L}^*_{\operatorname{St}}}}{\sqrt{c \eta (K+1)}} = \mathcal{O}\left(\frac{1}{\sqrt{K}}\right)
    \end{equation}
    where $\mathcal{L}^*_{\operatorname{St}} \triangleq \min_{U \in \operatorname{St}(p, n)} \mathcal{L}(U)$ exists by compactness of $\operatorname{St}(p, n)$ (Weierstrass extreme value theorem).
\end{enumerate}
\end{theorem}
```

---

## 7. Production PyTorch Reference Implementation

This reference matches and extends `neuralops/trainer/optimizers/stiefel.py` and `riemannian.py`, implementing both QR retraction and low-rank Woodbury Cayley retraction with automatic parameter routing.

```python
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Production Reference: Stiefel Manifold Riemannian Optimization Engine
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def project_stiefel_gradient(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Project Euclidean gradient G onto the tangent space T_W St(p, n).
    
    Handles both column-stiefel (n >= p) and row-stiefel (n < p) matrices,
    as well as batched tensors (dim > 2).
    """
    if W.dim() == 2:
        n, p = W.shape
        if n >= p:
            # Column-Stiefel: W^T W = I_p
            # P_W(G) = G - W sym(W^T G)
            wt_g = torch.matmul(W.t(), G)
            sym = 0.5 * (wt_g + wt_g.t())
            return G - torch.matmul(W, sym)
        else:
            # Row-Stiefel: W W^T = I_n
            # P_W(G) = G - sym(G W^T) W
            g_wt = torch.matmul(G, W.t())
            sym = 0.5 * (g_wt + g_wt.t())
            return G - torch.matmul(sym, W)
    elif W.dim() > 2:
        n, p = W.shape[-2], W.shape[-1]
        wt = W.transpose(-2, -1)
        gt = G.transpose(-2, -1)
        if n >= p:
            wt_g = torch.matmul(wt, G)
            gt_w = torch.matmul(gt, W)
            sym = 0.5 * (wt_g + gt_w)
            return G - torch.matmul(W, sym)
        else:
            g_wt = torch.matmul(G, wt)
            w_gt = torch.matmul(W, gt)
            sym = 0.5 * (g_wt + w_gt)
            return G - torch.matmul(sym, W)
    else:
        return G


def stiefel_qr_retraction(W: torch.Tensor, grad_R: torch.Tensor, lr: float) -> torch.Tensor:
    """Apply thin QR retraction: R_W(xi) = qf(W - lr * grad_R)."""
    w_step = W - lr * grad_R
    
    if W.dim() == 2:
        n, p = W.shape
        if n >= p:
            q, r = torch.linalg.qr(w_step, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            return q * ph.unsqueeze(-2)
        else:
            q, r = torch.linalg.qr(w_step.t(), mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            return (q * ph.unsqueeze(-2)).t()
    elif W.dim() > 2:
        n, p = W.shape[-2], W.shape[-1]
        if n >= p:
            q, r = torch.linalg.qr(w_step, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            return q * ph.unsqueeze(-2)
        else:
            w_step_t = w_step.transpose(-2, -1)
            q, r = torch.linalg.qr(w_step_t, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            return (q * ph.unsqueeze(-2)).transpose(-2, -1)
    else:
        norm = torch.linalg.norm(w_step)
        return w_step / (norm + 1e-12)


def stiefel_cayley_retraction_woodbury(
    W: torch.Tensor,
    grad_R: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Apply Cayley transform retraction with Sherman-Morrison-Woodbury inversion.
    
    W_new = (I - 0.5 * lr * A)^{-1} (I + 0.5 * lr * A) W
    where A = grad_R W^T - W grad_R^T is skew-symmetric with rank 2p.
    """
    n, p = W.shape[-2], W.shape[-1]
    if n < p:
        # Dual formulation for row-stiefel
        return stiefel_cayley_retraction_woodbury(W.t(), grad_R.t(), lr).t()
    
    # Low-rank factorization: A = U V^T where U = [grad_R, -W], V = [W, grad_R]
    U_mat = torch.cat([grad_R, -W], dim=-1)  # (n, 2p)
    V_mat = torch.cat([W, grad_R], dim=-1)  # (n, 2p)
    
    # Woodbury identity:
    # (I_n - 0.5 * lr * U V^T)^{-1} = I_n + 0.5 * lr * U (I_{2p} - 0.5 * lr * V^T U)^{-1} V^T
    tau = 0.5 * lr
    VT_U = torch.matmul(V_mat.t(), U_mat)  # (2p, 2p)
    core = torch.eye(2 * p, device=W.device, dtype=W.dtype) - tau * VT_U
    core_inv = torch.linalg.inv(core)
    
    # Compute numerator: (I_n + tau * U V^T) W = W + tau * U (V^T W)
    vt_w = torch.matmul(V_mat.t(), W)
    num = W + tau * torch.matmul(U_mat, vt_w)
    
    # Apply Woodbury inverse to numerator:
    vt_num = torch.matmul(V_mat.t(), num)
    sol = num + tau * torch.matmul(U_mat, torch.matmul(core_inv, vt_num))
    return sol


class RiemannianStiefelAdamW(Optimizer):
    """Production Hybrid Riemannian-Stiefel + Euclidean AdamW Optimizer.
    
    Automatically routes parameters with Stiefel orthogonal constraints through
    tangent space projection, vector transport, and retraction, while optimizing
    remaining parameters under decoupled weight decay AdamW.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        lr_stiefel: float | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        retraction: str = "qr",  # 'qr' or 'cayley'
    ) -> None:
        effective_lr_stiefel = lr_stiefel if lr_stiefel is not None else lr
        defaults = dict(
            lr=lr,
            lr_stiefel=effective_lr_stiefel,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            retraction=retraction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            manifold = group.get("manifold", "euclidean")
            lr = group["lr"]
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            eps = group.get("eps", 1e-8)
            retraction_type = group.get("retraction", "qr")

            if manifold == "stiefel":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        grad = grad.to_dense()

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1

                    # 1. Project ambient Euclidean gradient to tangent space
                    grad_r = project_stiefel_gradient(p.data, grad)

                    # 2. Vector transport prior first moment: P_{p.data}(exp_avg)
                    exp_avg_transported = project_stiefel_gradient(p.data, exp_avg)

                    # 3. Update moments
                    exp_avg.copy_(exp_avg_transported.mul_(beta1).add_(grad_r, alpha=1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad_r, grad_r, value=1.0 - beta2)

                    # 4. Bias correction
                    bias_corr1 = 1.0 - beta1 ** state["step"]
                    bias_corr2 = 1.0 - beta2 ** state["step"]
                    m_hat = exp_avg / bias_corr1
                    v_hat = exp_avg_sq / bias_corr2

                    # 5. Preconditioned tangent step with tangent re-projection
                    step_dir = m_hat / (v_hat.sqrt() + eps)
                    tangent_step = project_stiefel_gradient(p.data, step_dir)

                    # 6. Manifold Retraction onto Stiefel manifold
                    if retraction_type == "cayley":
                        w_next = stiefel_cayley_retraction_woodbury(p.data, tangent_step, lr)
                    else:
                        w_next = stiefel_qr_retraction(p.data, tangent_step, lr)

                    p.data.copy_(w_next)

            else:
                # Standard Euclidean AdamW execution path
                weight_decay = group.get("weight_decay", 0.0)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1

                    if weight_decay != 0.0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** state["step"]
                    bias_corr2 = 1.0 - beta2 ** state["step"]
                    step_size = lr / bias_corr1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def compute_orthogonality_error(self) -> dict[str, float]:
        """Compute Frobenius norm deviation ||W^T W - I_p||_F across all Stiefel parameters."""
        errors: dict[str, float] = {}
        for group in self.param_groups:
            if group.get("manifold") == "stiefel":
                for i, p in enumerate(group["params"]):
                    w = p.detach()
                    n, p_dim = w.shape[-2], w.shape[-1]
                    if n >= p_dim:
                        gram = torch.matmul(w.transpose(-2, -1), w)
                        eye = torch.eye(p_dim, device=w.device, dtype=w.dtype)
                    else:
                        gram = torch.matmul(w, w.transpose(-2, -1))
                        eye = torch.eye(n, device=w.device, dtype=w.dtype)
                    err = float(torch.linalg.norm(gram - eye).item())
                    errors[f"stiefel_param_{i}"] = err
        return errors
```

---

## 8. Publication-Grade LaTeX Drafting Templates

When generating camera-ready LaTeX code for manuscripts, use this self-contained snippet compatible with AMS packages (`amsmath`, `amssymb`, `amsthm`, `booktabs`):

```latex
% Preamble requirements:
% \usepackage{amsmath, amssymb, amsthm, mathtools}
% \newtheorem{theorem}{Theorem}
% \newtheorem{proposition}{Proposition}
% \newtheorem{definition}{Definition}

\subsection{Riemannian Stiefel Optimization for Low-Rank Bases}
\label{sec:riemannian_optimization}

To guarantee that post-surgery low-rank factorizations $W \approx U_{\mathrm{down}} V_{\mathrm{up}}$ do not suffer from directional subspace collapse or singular value decay, we constrain the projection bases to the compact Stiefel manifold:
\begin{equation}
\operatorname{St}(p, n) \triangleq \left\{ U \in \mathbb{R}^{n \times p} \;\middle|\; U^\top U = I_p \right\}, \quad p \le n
\label{eq:stiefel_manifold_def}
\end{equation}
The tangent space $T_U \operatorname{St}(p, n)$ is the subspace of first-order orthogonal perturbations:
\begin{equation}
T_U \operatorname{St}(p, n) = \left\{ \xi \in \mathbb{R}^{n \times p} \;\middle|\; U^\top \xi + \xi^\top U = 0 \right\} = \left\{ \xi \;\middle|\; \operatorname{sym}\left(U^\top \xi\right) = 0 \right\}
\label{eq:stiefel_tangent_space}
\end{equation}
Given an ambient Euclidean gradient $G = \nabla_U \mathcal{L} \in \mathbb{R}^{n \times p}$, the Riemannian gradient under the induced canonical trace metric $\langle A, B \rangle = \operatorname{Tr}(A^\top B)$ is given by the unique orthogonal projection:
\begin{equation}
\operatorname{grad}_{\operatorname{St}} \mathcal{L}(U) = \mathcal{P}_U(G) = G - U \, \operatorname{sym}\left(U^\top G\right) = G - \frac{1}{2} U \left( U^\top G + G^\top U \right)
\label{eq:stiefel_proj_eval}
\end{equation}
Following a tangent momentum step $\xi \in T_U \operatorname{St}(p, n)$, the updated parameters are mapped back onto the manifold via the thin QR retraction:
\begin{equation}
R_U^{\mathrm{QR}}(\xi) = \operatorname{qf}\left( U + \xi \right)
\label{eq:qr_retraction_eval}
\end{equation}
where $\operatorname{qf}(M)$ extracts the orthogonal $Q$ factor of the reduced QR decomposition with positive diagonal elements ($R_{ii} > 0$).
By Theorem~\ref{thm:condition_number_invariance}, this transformation preserves the condition number $\kappa(U_k) \equiv 1.0$ unconditionally across all optimization epochs, eliminating gradient vanishing along singular directions.
```

---

## 9. Verification & Pre-Flight Quality Gate

Before submitting any manuscript text or PR incorporating Stiefel manifold equations:
1. [ ] **Coordinate Transpose Guard**: Ensure $U^\top \xi + \xi^\top U = 0$ is used for column-Stiefel ($n \ge p$) and $U \xi^\top + \xi U^\top = 0$ for row-Stiefel ($n < p$).
2. [ ] **Symmetrizer Factor**: Verify $\operatorname{sym}(M) = \frac{1}{2}(M + M^\top)$; check that the $\frac{1}{2}$ factor is not omitted.
3. [ ] **Trace Annihilation Proof**: Verify the orthogonality proof uses $\operatorname{Tr}(S \cdot \Omega) = 0$ for symmetric $S$ and skew-symmetric $\Omega$.
4. [ ] **Woodbury Dimension Match**: Check that Cayley Woodbury inversion explicitly inverts a $(2p \times 2p)$ matrix, not an $(n \times n)$ matrix.
5. [ ] **Phase Normalization in QR**: Verify that the sign of diagonal entries $R_{ii}$ is checked to prevent sign-flip discontinuities in auto-differentiation graphs.
6. [ ] **No Fake Artifacts**: Ensure numerical fallbacks and placeholder values are strictly avoided per repository guidelines.
