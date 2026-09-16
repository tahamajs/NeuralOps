<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Truncated SVD

For any matrix $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma$ is diagonal with non-negative entries 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD** keeps the top-$r$ singular values:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

**Optimality.** By Eckart–Young, $W_r$ minimizes $\|W - W_r\|_F$ over all 
rank-$r$ matrices. The error is:

$$
\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{3.3}
$$

**Symmetric factorization.** We distribute $\Sigma^{1/2}$ between the two 
factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{3.4}
$$

so that $W_r = U_r V_r$ with balanced magnitudes.

## 3.2 Spectral Energy

Define the **cumulative spectral energy** of $W$:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2}
\tag{3.5}
$$

$E(r)$ is monotonically increasing from $E(0) = 0$ to $E(\min(m, n)) = 1$.

**Effective rank.** Given threshold $\eta$, the energy-adaptive rank is:

$$
r_\text{eff}(\eta) = \min\{r : E(r) \geq \eta\}
\tag{3.6}
$$

For rapidly-decaying spectra (e.g., power-law $\sigma_i \propto i^{-2}$), 
$r_\text{eff}$ is small. For uniform spectra, $r_\text{eff}$ approaches 
$\min(m, n)$.

**Empirical observations.** Transformer weight matrices typically have 
**power-law or exponential spectral decay**. Empirically, energy thresholds 
of $\eta \in [0.90, 0.99]$ capture the bulk of the spectrum at moderate rank.

## 3.3 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$, MHSA computes:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.7}
$$

MinimalNOVA factorizes each of $W_Q, W_K, W_V, W_O$.

## 3.4 Grouped-Query Attention (GQA)

Modern models use $H_{kv} < H$ KV heads:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h}
\tag{3.8}
$$

The KV heads are repeated $H/H_{kv}$ times before attention. MinimalNOVA 
supports GQA by factorizing the KV projections at their natural 
$H_{kv} d_h$ output dimension, then applying `repeat_kv` during forward.

## 3.5 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.9}
$$

MinimalNOVA applies RoPE **after** projection, preserving the rotational 
structure.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, 
$B = 0$ at initialization. In MinimalNOVA:

$$
y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.11}
$$

## 3.7 Nested Encoder Structure

In BERT-style models (RoBERTa, BERT), attention is **nested** inside 
`BertAttention`:

```python
BertAttention
├── self       (BertSelfAttention)
│   ├── query  (Linear)
│   ├── key    (Linear)
│   └── value  (Linear)
└── output     (BertSelfOutput)
    └── dense  (Linear)   # ← output projection
```

The output projection is **outside** the inner attention module. When 
replacing the inner attention, the outer `output.dense` must still apply. 
MinimalNOVA handles this via the `is_nested_encoder` flag.

## 3.8 Conv1D in GPT-2

GPT-2 uses HuggingFace's `Conv1D` (not `nn.Conv1d`), which stores weights 
as `(in_features, out_features)`:

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

This is **transposed** relative to `nn.Linear`. MinimalNOVA detects Conv1D 
via `hasattr(module, "nf")` and transposes weights before SVD.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $W, U, V$ | Weight and factors |
| $\sigma_i$ | Singular values |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $r_\text{max}$ | Rank cap |
| $d, d_h, H, H_{kv}$ | Model dims |
| $r_L, \alpha$ | LoRA rank, scaling |

---