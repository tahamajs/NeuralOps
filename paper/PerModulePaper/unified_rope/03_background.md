<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \mathrm{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.1}
$$

## 3.2 Rotary Position Embedding (RoPE)

RoPE encodes position by rotating each 2D subspace of $Q, K$:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \mathrm{rot}(q_i) \sin(\theta_i)
\tag{3.2}
$$

where the rotation operator acts on the first half vs second half:

$$
\mathrm{rot}(x) = \begin{pmatrix} -x_{d_h/2:} \\ x_{:d_h/2} \end{pmatrix}
\tag{3.3}
$$

and $\theta_i$ is a position-dependent angle:

$$
\theta_i^{(k)} = i \cdot \omega_k, \qquad
\omega_k = \mathrm{base}^{-2k/d_h}, \quad k = 0, \dots, d_h/2 - 1
\tag{3.4}
$$

**Effect on attention.** The score between $q_i$ and $k_j$ depends only on 
$i - j$:

$$
\tilde{q}_i^\top \tilde{k}_j = f(q_i, k_j, i - j)
\tag{3.5}
$$

## 3.3 Partial-Dimension RoPE

Not all models apply RoPE to the full head dimension. For models with 
partial rotation (e.g., Qwen with `rotary_dim < head_dim`):

$$
\tilde{q}_i = \begin{pmatrix} q_i^{[:r]} \cos(\theta_i) + \mathrm{rot}(q_i^{[:r]}) \sin(\theta_i) \\ q_i^{[r:]} \end{pmatrix}
\tag{3.6}
$$

where $r$ is the rotary dimension. UnifiedRoPE supports this via 
`rotary_dim = 2 * (head_dim // 2)`.

## 3.4 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times before attention:

$$
K_\text{expanded} = \mathrm{repeat\_kv}(K, H / H_{kv})
\tag{3.7}
$$

UnifiedRoPE applies `repeat_kv` after RoPE.

## 3.5 Nested Encoder Structure

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

The output projection `output.dense` is **outside** the inner attention. 
When replacing the inner attention, the outer `output.dense` must still 
apply. UnifiedRoPE returns raw context and lets the outer layer project.

## 3.6 Conv1D in GPT-2

GPT-2 uses HuggingFace's `Conv1D`, which stores weights as `(in, out)`:

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

This is **transposed** relative to `nn.Linear`. UnifiedRoPE detects Conv1D 
and transposes during weight transfer.

## 3.7 LoRA Adapters

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.8}
$$

with $B = 0$ at initialization. UnifiedRoPE's `LoRALinear` wraps this.

## 3.8 Precomputed Caches

RoPE angles $\theta_i$ are precomputed once:

```python
inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
t = arange(seq_len)
freqs = einsum("i,j->ij", t, inv_freq)
emb = cat([freqs, freqs], dim=-1)
cos_cached = emb.cos()
sin_cached = emb.sin()
```

UnifiedRoPE stores these as **non-persistent buffers** (`persistent=False`), 
so they are not saved with the model state dict.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $d, d_h, H, H_{kv}$ | Model dims |
| $Q, K, V$ | Query, Key, Value |
| $L$ | Sequence length |
| $B$ | Batch size |
| $\theta_i$ | RoPE angle at position $i$ |
| $\mathrm{rot}(\cdot)$ | Half-rotation operator |
| $\omega_k$ | Rotation frequency |
| $r$ | Rotary dimension |
| $r_L, \alpha$ | LoRA rank, scaling |

---