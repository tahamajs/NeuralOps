# Appendix A. Mathematical Derivations

## A.1 Short Path Exactness

**Theorem.** At $L \leq L_\text{threshold}$, with zero LoRA, FUSIONATTN 
output equals the pretrained attention output.

**Proof.** The short path computes standard softmax attention:

$$
\text{Out} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{A.1}
$$

with $Q, K, V$ projected using copied base weights (LoRA = 0). The 
computation is identical to the pretrained attention. $\square$

## A.2 Long Path Complexity

**Claim.** Long path attention costs $\mathcal{O}(K \cdot L \cdot B \cdot d_h)$.

**Proof.** For each of $T = L/B$ query blocks, we select top-$K$ key 
blocks, compute scores between $B$ queries and $K \cdot B$ keys, softmax, 
and context. Total cost:

$$
T \times B \times K B \times d_h = L \times K B \times d_h = \mathcal{O}(K L B d_h)
\tag{A.2}
$$

$\square$

## A.3 Gate Score Formulation

The gate score for block pair $(i, j)$:

$$
g(i, j) = \frac{\phi(\bar{Q}_i)^\top \phi(\bar{K}_j)}{\tau}
\tag{A.3}
$$

where $\phi$ is a 2-layer MLP. This is a **bilinear form** with learnable 
projections.

## A.4 Top-K Selection

For query block $i$, we select:

$$
\text{TopK}_i = \arg\text{top-}K_j (g(i, j))
\tag{A.4}
$$

Block 0 is always included (attention sink). This is a **hard, discrete** 
selection. Gradient flows through the selected blocks only.

## A.5 Causality in Block Attention

For a query block $i$ with selected key block $j$:

- If $j < i$: all keys are causally valid.
- If $j = i$: within-block causal masking applied.
- If $j > i$: no keys are valid (but block 0 always selected if $i > 0$, 
  and $j = 0$ for $i > 0$ is always causally valid).

Causality is enforced by:

$$
\text{mask}(q, k) = -\infty \text{ if } q < k
\tag{A.5}
$$

## A.6 Parameter Count

For layer with $d$ hidden, $d_h$ head_dim, $r_L$ LoRA rank:

| Component | Parameters |
|---|---|
| LoRA Q | $2 r_L d$ |
| LoRA K | $2 r_L d$ |
| LoRA V | $2 r_L d$ |
| LoRA O | $2 r_L d$ |
| Gate MLP $W_1$ | $4 d_h^2$ |
| Gate MLP $W_2$ | $4 d_h$ |
| Gate bias | $4 d_h + 1$ |
| Gate temp | 1 |
| **Total per layer** | $8 r_L d + 4 d_h^2 + O(d_h)$ |

For $d = 768$, $r_L = 16$, $d_h = 64$: **~131K params per layer**.

---