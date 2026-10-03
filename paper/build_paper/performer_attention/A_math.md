<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Math derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Kernel Approximation

$$
\text{softmax}(q^\top k / \sqrt{d}) \approx \frac{\phi(q)^\top \phi(k)}{\sum_j \phi(q)^\top \phi(k_j)}
\tag{A.1}
$$

## A.2 FAVOR+ Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(\omega^\top x - \frac{\|x\|^2}{2}\right)
\tag{A.2}
$$

**Proof of positivity:** $\exp(\cdot) > 0$ always.

**Proof of unbiasedness:** For $\omega \sim \mathcal{N}(0, I)$:

$$
\mathbb{E}_\omega[\phi(q)^\top \phi(k)] = \exp(q^\top k)
\tag{A.3}
$$

## A.3 Orthogonal Initialization

The matrix $\omega \in \mathbb{R}^{d \times F}$ is built from orthogonal 
blocks. Each block $B_i \in \mathbb{R}^{d \times d}$ satisfies 
$B_i^\top B_i = I$.

**Why orthogonal?** Reduces variance of the estimator:

$$
\text{Var}[\phi(q)^\top \phi(k)]_{\text{orthogonal}} < \text{Var}[\phi(q)^\top \phi(k)]_{\text{i.i.d.}}
\tag{A.4}
$$

## A.4 Numerical Stabilization

**Problem.** For large $\|x\|$, $\exp(\omega^\top x - \|x\|^2/2)$ overflows.

**Solution.**

$$
\phi(x) = \exp\!\left(\omega^\top x - \max(\omega^\top x) - \frac{\|x\|^2}{2}\right)
\tag{A.5}
$$

Since $\omega^\top x - \max(\omega^\top x) \leq 0$, the exponent is bounded 
above by $-\|x\|^2/2$. For $\|x\| \leq \sqrt{60}$, the exponent stays in 
the FP16 range.

## A.5 Causal Prefix Scan

For chunk $c$, start position $s_c$:

$$
\text{intra}_i = \sum_{j \in \text{chunk}_c, j \leq i} \phi(q_i)^\top \phi(k_j) v_j
\tag{A.6}
$$

$$
\text{inter}_i = \phi(q_i)^\top S_{c-1}
\tag{A.7}
$$

where $S_{c-1} = \sum_{j < s_c} \phi(k_j) v_j^\top$.

## A.6 Complexity

For $F$ features, $d$ head dim, $L$ sequence:

| Operation | Cost |
|---|---|
| Feature map | $\mathcal{O}(L F d)$ |
| Bidirectional KV | $\mathcal{O}(L F d)$ |
| Causal intra | $\mathcal{O}(L C F d)$ per chunk |
| Causal inter | $\mathcal{O}(L F d)$ |
| **Total** | $\mathcal{O}(L F d)$ |

Linear in $L$.

## A.7 Approximation Error

Choromanski et al. (2021) bound:

$$
\left|\text{Attn}_{\text{FAVOR+}} - \text{Attn}_{\text{softmax}}\right| \leq C \cdot \frac{\|V\|_\infty \cdot e^{\|\omega\|}}{\sqrt{F}}
\tag{A.8}
$$

For $F = 64$: error $\sim 0.125 \cdot \|V\|_\infty \cdot e^{\|\omega\|}$.

---