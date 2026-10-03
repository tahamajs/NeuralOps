<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Attention

Standard attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

## 3.2 Sliding-Window Attention

A sliding-window mask restricts attention to nearby tokens:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W/2 \\ 0 & \text{otherwise} \end{cases}
\tag{3.2}
$$

Then:

$$
\text{Attn}_\text{window}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \log M\right)V
\tag{3.3}
$$

**Complexity.** $\mathcal{O}(L W)$ time, $\mathcal{O}(L W)$ memory. Linear 
in $L$ for fixed $W$.

## 3.3 Gating

An input-dependent gate:

$$
g_t = \sigma(W_g x_t + b_g)
\tag{3.4}
$$

where $W_g \in \mathbb{R}^{d \times k}$ and $k$ is the gate dimension. 
The gate can be per-token, per-head, or scalar.

**Initialization.** Bias $b_g$ controls the initial gate value. Common choices:
- $b_g = 0$: $\sigma(0) = 0.5$ (equal mixture)
- $b_g = 2$: $\sigma(2) \approx 0.88$ (favor first branch)
- $b_g = -2$: $\sigma(-2) \approx 0.12$ (favor second branch)

## 3.4 Preservation Gate (Residual with Learnable Coefficient)

A preservation gate blends the block output with the block input:

$$
Y = \alpha \cdot F(X) + (1 - \alpha) \cdot X
\tag{3.5}
$$

where $\alpha = \sigma(W_\alpha)$ is learnable.

**Special cases:**
- $\alpha = 1$: pure residual (standard transformer)
- $\alpha = 0$: pure identity (block ignored)
- $\alpha = 0.5$: half residual

**ReZero** (Bachlechner et al., 2021). Uses $\alpha = 0$ at initialization, 
i.e., identity at start. Our choice of $\alpha = \sigma(2) \approx 0.88$ is 
different: we start with residual-dominant but not identity.

## 3.5 LoRA

LoRA (Hu et al., 2021):

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha_L}{r} BA
\tag{3.6}
$$

with $B = 0$ at initialization.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Number of heads |
| $W$ | Window size |
| $g$ | Mixing gate |
| $\alpha$ | Preservation coefficient |
| $r$ | LoRA rank |
</parameter>

---