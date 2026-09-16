<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
\tag{3.1}
$$

with $W^Q, W^K, W^V \in \mathbb{R}^{H d_h \times d}$. Attention:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.2}
$$

The KV cache stores $K, V \in \mathbb{R}^{B \times H \times L \times d_h}$ — 
a total of $2 \cdot H \cdot d_h \cdot L$ values per layer.

## 3.2 Multi-Head Latent Attention (MLA)

MLA replaces $W^K, W^V$ with a **shared low-rank latent**:

$$
c_t = W^{DKV} h_t \in \mathbb{R}^c
\tag{3.3}
$$

Then:

$$
k_t = W^{UK} c_t, \quad v_t = W^{UV} c_t
\tag{3.4}
$$

where $W^{UK}, W^{UV} \in \mathbb{R}^{H d_h \times c}$.

The KV cache now stores $c_t \in \mathbb{R}^c$ instead of $(k_t, v_t) \in \mathbb{R}^{2 H d_h}$.

**Compression ratio:**

$$
\rho = \frac{2 H d_h}{c}
\tag{3.5}
$$

For DeepSeek-V2: $2 H d_h = 2 \cdot 128 \cdot 128 = 32{,}768$, $c = 512$, 
$\rho = 64$.

## 3.3 Low-Rank Initialization

To initialize $W^{DKV}, W^{UK}, W^{UV}$ from a pretrained attention, we 
concatenate the existing $W^K$ and $W^V$:

$$
W^{KV}_\text{fused} = \begin{pmatrix} W^K \\ W^V \end{pmatrix} \in \mathbb{R}^{2 H d_h \times d}
\tag{3.6}
$$

Then compute SVD:

$$
W^{KV}_\text{fused} \approx U \Sigma V^\top
\tag{3.7}
$$

and set:

$$
W^{DKV} = V_{:, :c}^\top, \quad
W^{UK} = U_{:, :c} \Sigma_{:c, :c}
\tag{3.8}
$$

with the corresponding slice of $U \Sigma$ for $W^{UV}$.

## 3.4 Singular Value Normalization

To preserve the reconstruction scale, we compute a scalar correction:

$$
\gamma = \frac{\|W^{KV}_\text{fused}\|_F}{\|\Sigma_{:c}\|_F}
\tag{3.9}
$$

and multiply the up-projection by $\gamma$:

$$
W^{UK} \leftarrow \gamma \cdot U_{:H d_h, :c} \Sigma_{:c}
\tag{3.10}
$$

This preserves the total Frobenius norm of the effective $W^K, W^V$ 
reconstruction.

## 3.5 GQA Compatibility

MLA is compatible with GQA. When $H_{kv} < H$, the up-projections have 
output dimension $H_{kv} d_h$, and `repeat_kv` expands them to $H d_h$ 
during forward.

## 3.6 RoPE Compatibility

RoPE (Su et al., 2021) applies rotations to $Q, K$. MLA preserves this by 
applying RoPE **after** the up-projection.

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ at initialization. In MLA, we apply LoRA to the Q and O 
projections.

## 3.8 Output Scale Calibration

MLA introduces a trainable scalar `output_scale` that multiplies the 
output. This is **not** in the original DeepSeek-V2 paper but is used in 
our reproduction to absorb distributional shift from the K/V factorization.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $W^Q, W^K, W^V, W^O$ | Standard attention weights |
| $W^{DKV}$ | Latent down-projection |
| $W^{UK}, W^{UV}$ | Up-projections |
| $c_t$ | Latent vector at position $t$ |
| $c$ | Latent dimension (rank) |
| $H, H_{kv}, d_h$ | Head counts and dimension |
| $\gamma$ | Normalization scale |
| $\rho$ | Compression ratio |

---