<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Complexity Analysis

**Claim.** OHA is $\mathcal{O}(L(W + d_c) d)$ in time.

**Proof.** 
- Local attention: $\mathcal{O}(L W d_h H)$.
- Compression: $X W_k^\text{comp}$ is $\mathcal{O}(L d d_c)$; softmax over $L$ is $\mathcal{O}(L d_c)$; matmul with $K$ is $\mathcal{O}(L d_c d_h H)$.
- Global attention: $Q C_K^\top$ is $\mathcal{O}(L d_c d_h H)$; softmax over $d_c$; matmul with $C_V$ is $\mathcal{O}(L d_c d_h H)$.
- Sum: $\mathcal{O}(L(W + d_c) H d_h) = \mathcal{O}(L(W + d_c) d)$.

## A.2 Compression as Soft Weighted Average

**Claim.** When $d_c = 1$ and $W_k^\text{comp} = \mathbf{1}$ (all-ones), the 
compressed key is a uniform average of K.

**Proof.** $P_K = \text{softmax}(\mathbf{1} X)$ has equal weights $1/L$ over 
the sequence. So $C_K = \frac{1}{L} \sum_i K_i$, a uniform average. $\square$

## A.3 Gate Initialization

**Claim.** With bias $2.0$, $\gamma \approx 0.88$ at start.

**Proof.** $\sigma(2.0) = \frac{1}{1 + e^{-2}} = \frac{e^2}{e^2 + 1} \approx 0.8808$. $\square$

## A.4 Causality

**Claim.** OHA preserves causality for decoder models.

**Proof.** 
1. Local mask enforces $j \leq i$.
2. Global mask uses `tril` on $d_c$ slots.
3. Compression softmax is over positions $1..i$.

Therefore output at position $i$ depends only on inputs $1..i$. $\square$

## A.5 Parameter Count

**Claim.** OHA adds $2 d d_c + d H + H + 4 r(d + H d_h)$ parameters per layer.

**Proof.**
- Compression: $2 d d_c$.
- Gate weight: $d H$.
- Gate bias: $H$.
- LoRA (4 projections): $4 r(d + H d_h)$.
- Total: $2 d d_c + d H + H + 4 r(d + H d_h)$.

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $d_c = 32$, $r = 16$):
- $2 \cdot 768 \cdot 32 = 49,152$
- $768 \cdot 12 = 9,216$
- $12$
- $4 \cdot 16 \cdot 1536 = 98,304$
- Total per layer: $156,684$ ✓

---
