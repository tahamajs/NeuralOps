# 3. Background

## 3.1 Standard Attention

For $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

Complexity: $\mathcal{O}(L^2 d_h)$.

## 3.2 Block-Sparse Attention

Partition the sequence into $T = L / B$ blocks. For block $i$ of queries 
and block $j$ of keys, attention is computed only if $(i, j) \in \mathcal{S}$ 
for some sparsity mask $\mathcal{S}$.

**Fixed mask.** Longformer: $(i, j) \in \mathcal{S}$ iff $|i - j| \leq W/B$ 
or $j = 0$.

**Content-based mask.** MoBA/Routing: $(i, j) \in \mathcal{S}$ iff $j$ 
is among the top-$K$ for $i$ by some score.

## 3.3 MoBA-Style Block Selection

MoBA (Lu et al., 2024) computes a **block gate score**:

$$
g(i, j) = \langle \phi(\bar{Q}_i), \phi(\bar{K}_j) \rangle / \tau
\tag{3.2}
$$

where $\bar{Q}_i$ is the mean of query block $i$, $\phi$ is a learned 
projection, and $\tau$ is a temperature.

For each query block $i$, the top-$K$ key blocks by $g(i, j)$ are selected.

## 3.4 Grouped Query Attention

GQA uses $H_{kv} < H$ KV heads, repeated to match query heads:

$$
\text{repeat\_kv}(K, n)_{b, h, i, d} = K_{b, h \bmod H_{kv}, i, d}
\tag{3.3}
$$

## 3.5 Rotary Position Embeddings

RoPE rotates Q, K by position-dependent angles. Partial RoPE (Qwen3.5) 
rotates only the first `rotary_dim` coordinates.

## 3.6 LoRA

LoRA (Hu et al., 2021) parameterizes:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.4}
$$

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Block size |
| $T$ | Number of blocks |
| $K$ | Top-K blocks |
| $\bar{Q}_i$ | Query block mean |
| $\bar{K}_j$ | Key block mean |
| $g(i, j)$ | Gate score |
| $\tau$ | Gate temperature |

---