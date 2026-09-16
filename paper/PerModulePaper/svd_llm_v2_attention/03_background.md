<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^H
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
unitary, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.** Keeping the top-$r$:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^H
\tag{3.2}
$$

**Optimality.** By Eckart–Young, $W_r$ minimizes $\|W - W_r\|_F$. Error:

$$
\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{3.3}
$$

**Symmetric form (common).** Many implementations store $U\Sigma^{1/2}$ and 
$\Sigma^{1/2}V^H$ for FP16 stability:

$$
W_r = \underbrace{U_{:, :r}\Sigma_{:r, :r}^{1/2}}_{\text{factor 1}} \cdot \underbrace{\Sigma_{:r, :r}^{1/2}V_{:r, :}^H}_{\text{factor 2}}
\tag{3.4}
$$

**Explicit form (SVD-LLM V2).** SVD-LLM V2 stores $U$, $\Sigma$, $V^H$ 
**separately**:

$$
W_r = U_{:, :r} \cdot \Sigma_{:r, :r} \cdot V_{:r, :}^H
\tag{3.5}
$$

This is less numerically stable but **interpretable**.

## 3.2 Cumulative Spectral Energy

$$
E(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.6}
$$

$E(r)$ is monotonically increasing from $0$ to $1$.

**Energy threshold.** Given $\eta$:

$$
r_\text{eff} = \min\{r : E(r) \geq \eta\}
\tag{3.7}
$$

## 3.3 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.8}
$$

## 3.4 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads. Repeated $H / H_{kv}$ times. SVD-LLM V2 factorizes at 
natural dimensions and applies `repeat_kv` at forward.

## 3.5 GQA Weight Pooling

When the source model has more KV heads than the target dimension, we 
**mean-pool** per group:

$$
W_K^\text{pooled}_{h, :} = \frac{1}{|G_h|} \sum_{g \in G_h} W_{g, :}
\tag{3.9}
$$

where $G_h$ is the group of source heads mapped to target head $h$. This is 
used when loading a checkpoint trained with a different head count.

## 3.6 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.10}
$$

SVD-LLM V2 applies RoPE after projection, preserving structure.

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ at initialization. In SVD-LLM V2:

$$
y = ((x V^H{}^\top) \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.12}
$$

## 3.8 Nested Encoder Structure

In BERT-style models:

```python
BertAttention
├── self       (BertSelfAttention)
│   ├── query
│   ├── key
│   └── value
└── output     (BertSelfOutput)
    └── dense  # ← Output projection OUTSIDE
```

SVD-LLM V2 sets `is_nested_encoder = True` and skips `out_proj` in forward.

## 3.9 Conv1D in GPT-2

GPT-2 uses HuggingFace `Conv1D`, storing `(in, out)`:

```python
self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

SVD-LLM V2 transposes via `_unpack_weight_bias`.

## 3.10 Notation Summary

| Symbol | Meaning |
|---|---|
| $W, U, \Sigma, V^H$ | Weight and factors |
| $\sigma_i$ | Singular values |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $d, H, H_{kv}, d_h$ | Model dims |
| $r_L, \alpha$ | LoRA rank, scaling |

---