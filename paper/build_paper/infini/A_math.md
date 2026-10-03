<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Complexity Analysis

**Claim.** Infini-Attention is $\mathcal{O}(L(W + d_c) d)$ in time.

**Proof.** 
- Local attention: $Q_i$ attends to $W$ keys → $\mathcal{O}(W)$ per query. 
  Total: $\mathcal{O}(L W d_h)$.
- Memory attention: $Q_i$ attends to $d_c$ memory slots → $\mathcal{O}(d_c)$ 
  per query. Total: $\mathcal{O}(L d_c d_h)$.
- Sum over $H$ heads: $\mathcal{O}(L(W + d_c) H d_h) = \mathcal{O}(L(W + d_c) d)$.

## A.2 Gate Initialization Effect

**Claim.** At $\beta = -5$, memory contributes <1%.

**Proof.** $\sigma(-5) = \frac{1}{1 + e^5} \approx 0.0067$. If $\|M\|_F = O(0.001 \sqrt{d_c d_h})$, 
then $\sigma(\beta) \|M\|_F = O(0.007 \cdot 0.001 \sqrt{d_c d_h})$. This is 
negligible compared to local attention output (typically $O(1)$). $\square$

## A.3 Parameter Count

**Claim.** Infini-Attention adds $H d_c d_h + 1 + 4 r(d + H d_h)$ parameters 
per layer (assuming $H = H_{kv}$).

**Proof.** 
- Memory: $H d_c d_h$.
- Gate: 1.
- LoRA Q: $r(d + H d_h)$.
- LoRA K: $r(d + H d_h)$.
- LoRA V: $r(d + H d_h)$.
- LoRA O: $r(H d_h + d) = r(d + H d_h)$.
- Total LoRA: $4 r(d + H d_h)$.

For DistilBERT ($H = 12$, $d_h = 64$, $d_c = 128$, $d = 768$, $r = 16$):
- Memory: $12 \cdot 128 \cdot 64 = 98,304$.
- LoRA: $4 \cdot 16 \cdot (768 + 768) = 98,304$.
- Total per layer: $196,609$.

## A.4 Causality

**Claim.** Infini-Attention preserves causality for decoder models.

**Proof.** The local mask enforces $j \leq i$. The memory branch uses only 
$Q_i$ (position-aware) and $M$ (position-independent). Therefore, output 
at position $i$ depends only on inputs $1..i$. $\square$

## A.5 Fusion Preserves Scale

**Claim.** Additive fusion with $\sigma(\beta) \approx 0$ preserves the 
local branch's scale.

**Proof.** $\|\text{output}\|_2 = \|\text{local} + \sigma(\beta) \text{mem}\|_2 \leq \|\text{local}\|_2 + \sigma(\beta)\|\text{mem}\|_2$. 
For $\sigma(\beta) \to 0$, this is $\|\text{local}\|_2$. $\square$

---