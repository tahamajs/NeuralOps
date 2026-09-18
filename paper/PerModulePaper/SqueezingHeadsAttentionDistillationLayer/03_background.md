<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.2}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.3}
$$

**Parameter count.** $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$, 
$W_O \in \mathbb{R}^{d \times d}$. Total: $4 d^2$.

## 3.2 Head Redundancy

Empirical findings:

- Most heads in BERT can be pruned (Voita et al., 2019).
- Many layers need only one head (Michel et al., 2019).
- Head patterns are dominated by a few categories (Clark et al., 2019).

**Implication.** Reducing head capacity is a viable compression strategy.

## 3.3 LoRA

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.4}
$$

where $A \in \mathbb{R}^{r_L \times d}$, $B \in \mathbb{R}^{d \times r_L}$, 
$B = 0$ at initialization. Only $A, B$ are trained.

**Benefits.**
- **Small trainable footprint:** $2 r_L d$ vs $d^2$.
- **Preserves pretrained:** Base weights unchanged.
- **Composable:** Multiple LoRA adapters can be combined.

## 3.4 Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) uses $H_{kv} < H$ KV heads. When the source 
attention has more heads than the target, we pool:

$$
W_K^{(\text{target})}_g = \frac{1}{r} \sum_{h \in g} W_K^{(\text{source})}_h
\tag{3.5}
$$

where $r = H_\text{source} / H_\text{target}$.

## 3.5 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) applies position-dependent rotation:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.6}
$$

Squeezing Heads preserves RoPE through the wrapper.

## 3.6 Head Dimension Reduction

When reducing head dimension from $d_h$ to $d_h / s$:

$$
W_Q^{(\text{new})} = W_Q^{(\text{old})}_{:d_h/s, :}
\tag{3.7}
$$

We preserve the **leading** $d_h / s$ rows, corresponding to the principal 
components of the head.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Per-head dimension |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $s$ | Head-dim reduction factor |
| $W_Q, W_K, W_V, W_O$ | Attention projections |

---
