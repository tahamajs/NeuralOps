# 3. Background

## 3.1 Standard Multi-Head Attention

For input $X \in \mathbb{R}^{B \times N \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

Complexity: $\mathcal{O}(N^2 d_h)$.

## 3.2 Mixture-of-Block Attention (MoBA)

MoBA partitions the sequence into $N_p = \lceil N / P \rceil$ pages of 
size $P$. For each query page $i$, a router selects the top-$k_\text{exp}$ 
key pages:

$$
\mathcal{S}_i = \text{TopK}(\{\ell_{ij}\}_j, k_\text{exp})
\tag{3.3}
$$

Attention is computed only over the selected pages:

$$
\text{Out}_i = \text{softmax}\!\left(\frac{Q_i K_{\mathcal{S}_i}^\top}{\sqrt{d_h}}\right) V_{\mathcal{S}_i}
\tag{3.4}
$$

Complexity: $\mathcal{O}(N_p \cdot k_\text{exp} \cdot P^2 d_h)$.

## 3.3 The MoBA Router

The router logit between query page $i$ and key page $j$:

$$
\ell_{ij} = \frac{(W_\text{rg} \bar{q}_i) \cdot \bar{k}_j}{\sqrt{d_h}}
\tag{3.5}
$$

where:

$$
\bar{q}_i = \frac{1}{P} \sum_{t \in \text{page}_i} q_t, \quad
\bar{k}_j = \frac{1}{P} \sum_{t \in \text{page}_j} k_t
\tag{3.6}
$$

$W_\text{rg} \in \mathbb{R}^{d_h \times d_h}$ is a learnable routing matrix, 
initialized to identity.

## 3.4 Delta-KV Compression

Given consecutive key vectors $k_t, k_{t-1}$, the temporal delta is:

$$
\Delta k_t = k_t - k_{t-1}
\tag{3.7}
$$

For smooth sequences (natural language often is), $\|\Delta k_t\| \ll \|k_t\|$.

**Compression.** Reshape deltas to $\Delta \in \mathbb{R}^{(B H (N-1)) \times d_h}$, 
then compute SVD:

$$
\Delta \approx U_r \Sigma_r V_r^\top
\tag{3.8}
$$

Reconstruct:

$$
\hat{\Delta} = U_r \Sigma_r V_r^\top
\tag{3.9}
$$

Then reconstruct $K$:

$$
\hat{k}_t = k_0 + \sum_{\tau=1}^t \hat{\Delta}_\tau
\tag{3.10}
$$

## 3.5 Straight-Through Estimator (STE)

For differentiable quantization:

$$
\text{STE}(x) = x + (Q(x) - x).\text{detach}()
\tag{3.11}
$$

**Forward:** $Q(x)$ (quantized).
**Backward:** $\nabla \text{STE}(x) = \nabla x$ (identity).

## 3.6 4-Bit Uniform Quantization

Given $x$, compute min/max per row:

$$
x_\text{min} = \min(x), \quad x_\text{max} = \max(x)
\tag{3.12}
$$

Scale:

$$
s = \frac{x_\text{max} - x_\text{min}}{15}
\tag{3.13}
$$

Quantize:

$$
x_q = \text{round}\!\left(\frac{x - x_\text{min}}{s}\right) \in \{0, 1, \dots, 15\}
\tag{3.14}
$$

Dequantize:

$$
\hat{x} = x_q \cdot s + x_\text{min}
\tag{3.15}
$$

## 3.7 Load Balancing Loss

For MoE routers, a common auxiliary loss:

$$
\mathcal{L}_\text{aux} = N_p \cdot \sum_{j=1}^{N_p} \rho_j \cdot \pi_j
\tag{3.16}
$$

where:

- $\rho_j$ = fraction of queries that selected page $j$
- $\pi_j$ = mean router probability for page $j$

**Effect.** Encourages uniform page selection, preventing router collapse.

## 3.8 CLS Token Preservation

In classification models (BERT family), the `[CLS]` token at position 0 
aggregates sequence information. We ensure page 0 is always selected:

$$
\ell_{i, 0} \leftarrow \ell_{i, 0} + 10^4
\tag{3.17}
$$

This boost is large enough that page 0 always appears in the top-$k$.

## 3.9 Grouped-Query Attention (GQA)

For $H_{kv} < H$, K/V are shared across query heads. Our implementation 
expands via `repeat_kv` before attention.

## 3.10 Rotary Position Embeddings (RoPE)

Applied to $Q, K$ after projection, before delta compression.

## 3.11 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $P$ | Page size |
| $N_p$ | Number of pages |
| $k_\text{exp}$ | Experts per query |
| $r$ | Delta SVD rank |
| $d_h$ | Head dim |
| $W_\text{rg}$ | Router weight |

---