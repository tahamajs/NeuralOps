<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 LoRA Initialization

**Claim.** With $A$ initialized from Kaiming uniform and $B = 0$, the 
effective weight $W + \frac{\alpha}{r} B A = W$ at initialization.

**Proof.** Substituting $B = 0$:

$$
W + \frac{\alpha}{r} \cdot 0 \cdot A = W \qquad \square
$$

## A.2 Attention Complexity

**Claim.** Standard MHSA is $\mathcal{O}(L^2 d)$ in time and $\mathcal{O}(L^2)$ 
in memory.

**Proof.** The attention matrix $S = Q K^\top$ has shape $B \times H \times L \times L$. 
Computing it takes $B H L^2 d_h$ operations. Softmax over $L$ entries per 
row takes $\mathcal{O}(B H L^2)$. The output $P V$ takes $B H L^2 d_h$. 
Summing: $\mathcal{O}(B H L^2 d_h) = \mathcal{O}(L^2 d)$ since $d = H d_h$. $\square$

## A.3 LoRA Overhead

**Claim.** LoRA adds $\mathcal{O}(L d r)$ compute per forward pass.

**Proof.** For each of the four projections, LoRA computes:

$$
\text{LoRA}(X) = (X A^\top) B^\top
$$

Two matmuls: $X \in \mathbb{R}^{B \times L \times d}$ times $A^\top \in \mathbb{R}^{d \times r}$ 
gives $\mathcal{O}(L d r)$; the result times $B^\top \in \mathbb{R}^{r \times d_\text{out}}$ 
gives $\mathcal{O}(L r d_\text{out})$. Total $\mathcal{O}(L d r)$. $\square$

## A.4 Parameter Increase

**Claim.** EAT adds $r(d_\text{in} + d_\text{out})$ parameters per projection.

**Proof.** LoRA has $A \in \mathbb{R}^{r \times d_\text{in}}$ and 
$B \in \mathbb{R}^{d_\text{out} \times r}$. Total: $r d_\text{in} + d_\text{out} r = r(d_\text{in} + d_\text{out})$. $\square$

For DistilBERT (6 layers, 4 projections of shape $768 \times 768$, $r = 16$):

$$
6 \times 4 \times 16 \times (768 + 768) = 589{,}824
$$

matching the empirical increase.

## A.5 Pruning Complexity

**Claim.** Top-$k$ pruning is $\mathcal{O}(L \log L)$.

**Proof.** Sorting $L$ scores takes $\mathcal{O}(L \log L)$. Selecting top-$k$ 
from a heap takes $\mathcal{O}(L \log k)$. Dominant: $\mathcal{O}(L \log L)$. $\square$

## A.6 Causality

**Claim.** EAT preserves causality for decoder models.

**Proof.** Each token's output depends only on:

1. Its own $q_i$ (via $Q = W_Q X + \text{LoRA}_Q(X)$, position-wise).
2. Keys $k_j$ and values $v_j$ for $j \leq i$ (via the causal mask).

The causal mask sets $S_{ij} = -\infty$ for $j > i$, so softmax assigns zero 
weight to future tokens. $\square$

---