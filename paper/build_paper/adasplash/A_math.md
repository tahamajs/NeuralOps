<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Derivation of α-Entmax Closed Form

**Problem.** Solve:

$$
p^* = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H_\alpha^T(p)
\tag{A.1}
$$

where $H_\alpha^T(p) = \frac{1}{\alpha(\alpha-1)} \sum_i (p_i - p_i^\alpha)$.

**Lagrangian.**

$$
\mathcal{L}(p, \tau) = \langle p, s \rangle + \frac{1}{\alpha(\alpha-1)} \sum_i (p_i - p_i^\alpha) - \tau \left(\sum_i p_i - 1\right)
\tag{A.2}
$$

**Stationarity.** $\partial \mathcal{L} / \partial p_i = 0$:

$$
s_i + \frac{1 - \alpha p_i^{\alpha-1}}{\alpha(\alpha-1)} - \tau = 0
\tag{A.3}
$$

Rearranging:

$$
p_i^{\alpha - 1} = (\alpha - 1) s_i - \tau + \frac{1}{\alpha}
\tag{A.4}
$$

Let $\tilde\tau = \tau - 1/\alpha$. Then:

$$
p_i = \max\left(0, (\alpha - 1) s_i - \tilde\tau\right)^{1/(\alpha - 1)}
\tag{A.5}
$$

with $\tilde\tau$ chosen so that $\sum_i p_i = 1$. $\square$

## A.2 Bisection Convergence

**Claim.** Bisection on $[\tau_\text{lo}, \tau_\text{hi}]$ converges to 
precision $\epsilon$ in $O(\log((\tau_\text{hi} - \tau_\text{lo})/\epsilon))$ 
iterations.

**Proof.** Standard bisection result. With initial range $O(1)$ and 25 
iterations, precision is $O(2^{-25}) = O(10^{-7.5})$. $\square$

## A.3 Support Sparsity Bound

**Claim.** At convergence, the support of α-entmax is:
$$
S = \{i : (\alpha - 1) s_i > \tau^*\}
$$

**Proof.** Direct from the closed form: $p_i > 0$ iff $(\alpha - 1) s_i > \tau^*$. $\square$

## A.4 Attention Sink Preservation

**Claim.** For causal decoders with splash masking, preserving first $S$ 
tokens does not violate causality.

**Proof.** For query $i$, attention mask includes key $j$ if:
- $j < S$ (sink) OR
- $0 \leq i - j \leq W$ (local window)

Case 1: Sink. If $j < S \leq i$ (assuming $i \geq S$), then $j \leq i$, 
so causality is preserved. For $i < S$, only $j \leq i$ are included 
because $i - j \geq 0$ requires $j \leq i$.

Case 2: Local. $0 \leq i - j \leq W$ implies $j \leq i$. Causality holds.

Therefore, all attended keys satisfy $j \leq i$. $\square$

## A.5 Parameter Count Analysis

For square $d \times d$ projection:

$$
\text{Original params} = d^2
\tag{A.6}
$$

$$
\text{AdaSplash params} = d^2 + r_L (d + d) = d^2 + 2 r_L d
\tag{A.7}
$$

**Change:** $+2 r_L d$ per projection.

Across 4 projections: $+8 r_L d$ per layer.

Across $N_l$ layers: $+8 r_L d N_l$.

For DistilBERT ($d = 768$, $r_L = 16$, $N_l = 6$): $+589{,}824$ params.

## A.6 Entmax vs Softmax Sparsity

**Claim.** For scores with high variance, α-entmax produces more zeros 
than softmax.

**Proof sketch.** Softmax assigns $p_i > 0$ for all $i$. Entmax assigns 
$p_i = 0$ when $(\alpha - 1) s_i < \tau^*$. The threshold $\tau^*$ scales 
with the score distribution; for high variance, fewer scores exceed 
$\tau^*$, giving more sparsity. $\square$

## A.7 Softmax Fallback

**Claim.** When support is empty, the softmax fallback is a valid probability 
distribution.

**Proof.** Softmax always produces valid probabilities. $\square$

## A.8 Causality of Entmax

**Claim.** α-entmax preserves causality under causal masking.

**Proof.** Entmax is a row-wise operation on the score matrix. Masking with 
$-\infty$ ensures that masked entries receive zero weight in the output. 
Row-wise independence preserves row-specific causality. $\square$
