<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 The Cosformer Identity

**Claim.**

$$
\tilde{q}_i^\top \tilde{k}_j + \tilde{q}_i'^\top \tilde{k}_j' = \phi(q_i)^\top \phi(k_j) \cdot \cos(\theta_i - \theta_j)
\tag{A.1}
$$

where:
- $\tilde{q}_i = \phi(q_i) \cos\theta_i$
- $\tilde{q}_i' = \phi(q_i) \sin\theta_i$
- $\tilde{k}_j = \phi(k_j) \cos\theta_j$
- $\tilde{k}_j' = \phi(k_j) \sin\theta_j$

**Proof.**

$$
\begin{aligned}
\tilde{q}_i^\top \tilde{k}_j &= (\phi(q_i) \cos\theta_i)^\top (\phi(k_j) \cos\theta_j) \\
&= \cos\theta_i \cos\theta_j \cdot \phi(q_i)^\top \phi(k_j)
\end{aligned}
\tag{A.2}
$$

Similarly:

$$
\tilde{q}_i'^\top \tilde{k}_j' = \sin\theta_i \sin\theta_j \cdot \phi(q_i)^\top \phi(k_j)
\tag{A.3}
$$

Sum:

$$
\tilde{q}_i^\top \tilde{k}_j + \tilde{q}_i'^\top \tilde{k}_j' = (\cos\theta_i \cos\theta_j + \sin\theta_i \sin\theta_j) \cdot \phi(q_i)^\top \phi(k_j) = \cos(\theta_i - \theta_j) \cdot \phi(q_i)^\top \phi(k_j)
\tag{A.4}
$$

$\square$

## A.2 Locality Analysis

With $\theta_i = \pi i / (2 M)$, we have $\theta_i - \theta_j = \pi (i - j) / (2 M)$.

$$
\cos(\theta_i - \theta_j) = \cos\!\left(\frac{\pi (i-j)}{2M}\right)
\tag{A.5}
$$

**Decay rate.** For $|i - j| = k$:

$$
\cos\!\left(\frac{\pi k}{2M}\right) \approx 1 - \frac{\pi^2 k^2}{8M^2}
\tag{A.6}
$$

For $M = 512$ and $k = 128$: $\cos \approx 0.87$. **Slow decay.**

**Rapid decay region.** For $k = M$: $\cos = 0$. So the reweighting 
essentially **zeros out** attention beyond $M$ tokens.

## A.3 Complexity of Encoder Path

**Time.** For each head:
- $K_c^\top V$: $\mathcal{O}(L d^2)$
- $K_s^\top V$: $\mathcal{O}(L d^2)$
- $Q_c (K_c^\top V)$: $\mathcal{O}(L d^2)$
- $Q_s (K_s^\top V)$: $\mathcal{O}(L d^2)$
- Denominators: $\mathcal{O}(L d)$

**Total:** $\mathcal{O}(L d^2)$ per head. **Linear in $L$.**

## A.4 Complexity of Decoder Path

**Time.** For each head, per chunk:
- Intra-chunk: $\mathcal{O}(C^2 d)$ per chunk, $\mathcal{O}(L C d)$ total
- Inter-chunk: $\mathcal{O}(C d^2)$ per chunk, $\mathcal{O}((L/C) \cdot C d^2) = \mathcal{O}(L d^2)$ total

**Total:** $\mathcal{O}(L d^2 + L C d)$.

For $C \ll d$: $\mathcal{O}(L d^2)$.

## A.5 Approximation Error

**ELU + 1 is not an RFF.** The approximation:

$$
\text{elu}(q)^\top \text{elu}(k) + \|q\|_1 + \|k\|_1 + d \approx \exp(q^\top k)
\tag{A.7}
$$

is only approximate. The error grows with $\|q\|$ and $\|k\|$.

**Bounds.** For $\|q\|, \|k\| \leq R$:

$$
\left|(\text{elu}(q)+1)^\top(\text{elu}(k)+1) - e^{q^\top k}\right| \leq C_R
\tag{A.8}
$$

where $C_R$ grows exponentially in $R$.

## A.6 Causality

For the causal path:

$$
\frac{\partial Y_i}{\partial X_j} = 0 \text{ for } j > i
\tag{A.9}
$$

**Proof sketch.** Within a chunk, the causal mask zeros future contributions. 
Across chunks, the state $(S, Z)$ only accumulates from processed chunks. 
So $Y_i$ depends only on $X_{\leq i}$. $\square$

---