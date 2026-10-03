# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

where $W_Q \in \mathbb{R}^{d \times H \cdot d_h}$, $W_K, W_V \in \mathbb{R}^{d \times H_{kv} \cdot d_h}$.

The output projection:

$$
\text{Out} = \text{Attn}(Q, K, V) W_O
\tag{3.2}
$$

where $W_O \in \mathbb{R}^{H \cdot d_h \times d}$.

## 3.2 Multi-Head Attention Parameters

| Projection | Shape | Params |
|---|---|---|
| $W_Q$ | $d \times H d_h$ | $d H d_h$ |
| $W_K$ | $d \times H_{kv} d_h$ | $d H_{kv} d_h$ |
| $W_V$ | $d \times H_{kv} d_h$ | $d H_{kv} d_h$ |
| $W_O$ | $H d_h \times d$ | $d H d_h$ |

For standard MHA ($H = H_{kv}$):
$$
\text{Total} = 4 d^2
\tag{3.3}
$$

## 3.3 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.4}
$$

where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$ diagonal, 
$V \in \mathbb{R}^{n \times n}$ orthogonal.

**Truncated SVD.** Rank-$r$ approximation:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.5}
$$

**Error.** $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

## 3.4 Symmetric Factorization

For numerical stability, distribute $\Sigma^{1/2}$:

$$
W_r = \underbrace{U_{:, :r} \Sigma_{:r, :r}^{1/2}}_{U_r'} \underbrace{\Sigma_{:r, :r}^{1/2} V_{:r, :}^\top}_{V_r'}
\tag{3.6}
$$

Both factors have balanced magnitude.

## 3.5 Joint QKV Stacking

Instead of separate SVD per projection, we stack:

$$
W_{QKV} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} \in \mathbb{R}^{(H d_h + 2 H_{kv} d_h) \times d}
\tag{3.7}
$$

Then:

$$
W_{QKV} = U_{QKV} \Sigma_{QKV} V_{QKV}^\top
\tag{3.8}
$$

**Key property.** The right factor $V_{QKV}$ is a **single basis** for the 
input space, shared across Q, K, V.

**Why this works.** Q, K, V in pretrained transformers are correlated — the 
same input features drive all three. A joint SVD captures this correlation.

## 3.6 LoRA

LoRA (Hu et al., 2021) adds trainable low-rank deltas:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.9}
$$

with $B = 0$ initially. Applied to both QKV and output branches.

## 3.7 Grouped-Query Attention (GQA)

For $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}
\tag{3.10}
$$

Expanded to $H$ heads via `repeat_kv`.

## 3.8 Rotary Position Embeddings (RoPE)

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.11}
$$

Applied to $Q, K$ after projection.

## 3.9 Nested vs Flat Encoders

| Model | Structure | Output Projection |
|---|---|---|
| DistilBERT | Flat | Inside attention (`out_lin`) |
| RoBERTa | Nested | Outside attention (`output.dense`) |
| GPT-2 | Flat | Inside attention (`c_proj`) |

**Critical.** For nested encoders, our attention module must **not** apply 
its own output projection. For flat encoders, it **must**. Getting this 
wrong causes double projection and collapse.

---