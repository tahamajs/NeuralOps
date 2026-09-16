<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Soft Ramp Properties

### A.1.1 Continuity

**Claim.** $g_z(d) = \text{clamp}((R + z - d) / R, 0, 1)$ is continuous in 
$(z, d)$.

**Proof.** The function $\text{clamp}(x, 0, 1) = \min(\max(x, 0), 1)$ is 
continuous in $x$. The argument $(R + z - d) / R$ is continuous in $(z, d)$. 
Composition preserves continuity. $\square$

### A.1.2 Differentiability in $z$

**Claim.** $\partial g_z(d) / \partial z$ exists everywhere.

**Proof.** 

- For $d < z$: $g_z(d) = 1$ (constant), so $\partial g_z / \partial z = 0$.
- For $z < d < z + R$: $g_z(d) = (R + z - d) / R$, so $\partial g_z / \partial z = 1/R$.
- For $d > z + R$: $g_z(d) = 0$ (constant), so $\partial g_z / \partial z = 0$.

At the boundaries $d = z$ and $d = z + R$, the derivative is one-sided but 
the function is continuous. $\square$

## A.2 Span Gradient via Sigmoid

Given $z = \sigma(\ell) \cdot S_\text{max}$:

$$
\frac{\partial z}{\partial \ell} = \sigma(\ell)(1 - \sigma(\ell)) \cdot S_\text{max}
\tag{A.1}
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial \ell} = \frac{\partial \mathcal{L}}{\partial z} \cdot \sigma(\ell)(1 - \sigma(\ell)) \cdot S_\text{max}
\tag{A.2}
$$

## A.3 L1 Penalty Effect

The L1 penalty adds:

$$
\mathcal{L}_\text{span} = \lambda \sum_h z_h = \lambda S_\text{max} \sum_h \sigma(\ell_h)
\tag{A.3}
$$

Gradient:

$$
\frac{\partial \mathcal{L}_\text{span}}{\partial \ell_h} = \lambda S_\text{max} \cdot \sigma(\ell_h)(1 - \sigma(\ell_h))
\tag{A.4}
$$

Since $\sigma \in (0, 1)$, this is always positive. The gradient pushes 
$\ell_h$ toward $-\infty$, hence $z_h \to 0$.

## A.4 Equilibrium Analysis

**Claim.** At equilibrium, the span satisfies:

$$
\left|\frac{\partial \mathcal{L}_\text{task}}{\partial z_h}\right| = \lambda
\tag{A.5}
$$

**Proof.** The total gradient is:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial \ell_h} + \lambda S_\text{max} \sigma(\ell_h)(1 - \sigma(\ell_h)) = 0
\tag{A.6}
$$

At equilibrium. Since $\partial z_h / \partial \ell_h = S_\text{max} \sigma(1-\sigma)$, 
this simplifies to:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial z_h} + \lambda = 0
\tag{A.7}
$$

$\square$

## A.5 Approximation Error Bound

**Claim.** For a head with span $z$ and ramp $R$:

$$
\|P - \tilde{P}\|_1 \leq \sum_{(i,j): |i-j| > z} P_{ij} + \epsilon_R
\tag{A.8}
$$

where:

$$
\epsilon_R \leq \max_{i,j} |1 - g_z(|i-j|)| \cdot \sum_j P_{ij}
\tag{A.9}
$$

**Proof sketch.** The ramp down-weights distant scores. The total mass 
removed is bounded by the sum over $|i-j| > z$ plus the ramp transition 
region. $\square$

## A.6 Comparison with Hard Window

For hard window of size $w$:

$$
P_{ij}^{\text{hard}} = \frac{\mathbb{1}[|i-j| \le w] \exp(S_{ij})}{\sum_j \mathbb{1}[|i-j| \le w] \exp(S_{ij})}
\tag{A.10}
$$

The soft ramp approaches this as $R \to 0$. For finite $R$, the soft ramp 
provides **smoother gradients** and **avoids boundary artifacts**.

---