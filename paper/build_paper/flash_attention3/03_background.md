# 3. Background

## 3.1 Standard Attention

For $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

**Naive implementation.** Materialize $QK^\top \in \mathbb{R}^{L \times L}$, 
apply softmax, multiply by $V$. Memory: $\mathcal{O}(L^2)$.

## 3.2 FlashAttention Tiling

FlashAttention partitions $Q, K, V$ into blocks:

$$
Q = [Q_1, Q_2, \dots, Q_T], \quad K = [K_1, \dots, K_T], \quad V = [V_1, \dots, V_T]
\tag{3.2}
$$

For each $Q$-block, iterate over $K, V$-blocks, computing partial attention 
in shared memory. Uses **online softmax** to avoid materializing the full 
attention matrix:

$$
m_i^{(t)} = \max(m_i^{(t-1)}, \max_j (Q_i K_j^\top / \sqrt{d_h}))
\tag{3.3}
$$

$$
\ell_i^{(t)} = e^{m_i^{(t-1)} - m_i^{(t)}} \ell_i^{(t-1)} + \sum_j e^{Q_i K_j^\top / \sqrt{d_h} - m_i^{(t)}}
\tag{3.4}
$$

$$
O_i^{(t)} = \text{diag}(e^{m_i^{(t-1)} - m_i^{(t)}}) O_i^{(t-1)} + \sum_j e^{Q_i K_j^\top / \sqrt{d_h} - m_i^{(t)}} V_j
\tag{3.5}
$$

The final output is $O_i = O_i^{(T)} / \ell_i^{(T)}$.

**Memory.** $\mathcal{O}(B \cdot L \cdot d_h)$ for the output plus 
$\mathcal{O}(B^2)$ for the block buffers. Total: $\mathcal{O}(L)$.

## 3.3 FlashAttention-3 Improvements

Over FlashAttention-2, v3 adds:

1. **TMA (Tensor Memory Accelerator).** Asynchronous bulk memory copies 
   from HBM to shared memory, freeing the SM for computation.
2. **Warp specialization.** Producer-consumer model where some warps handle 
   memory while others compute.
3. **FP8 support.** Tensor cores can process FP8 (e4m3) at 2× throughput.

**Result.** ~2× throughput over v2 on H100.

## 3.4 PyTorch SDPA

`F.scaled_dot_product_attention` auto-selects one of three backends:

- **Math path:** naive implementation.
- **Memory-efficient path (xFormers-style):** tiled computation similar 
  to FlashAttention-1.
- **FlashAttention-2 (if available):** uses the `flash_attn` library.

The wrapper dispatches to SDPA when FlashAttention-3 is unavailable.

## 3.5 GQA and Repeat_KV

For GQA, $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times L \times d_h}
\tag{3.6}
$$

Before attention, repeat each KV head $H/H_{kv}$ times:

$$
\text{repeat\_kv}(K, n) = K[:, :, \text{None}, :, :].\text{expand}(B, H_{kv}, n, L, d_h).\text{reshape}(B, H, L, d_h)
\tag{3.7}
$$

## 3.6 Rotary Position Embeddings

RoPE rotates $Q, K$ by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.8}
$$

Partial RoPE rotates only the first `rotary_dim` coordinates.

## 3.7 LoRA

LoRA (Hu et al., 2021) parameterizes a weight update as $BA$ with $B = 0$.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Block size (FlashAttention) |
| $T$ | Number of blocks |
| $m_i$ | Running max (online softmax) |
| $\ell_i$ | Running sum (online softmax) |
| $O_i$ | Running output |
| $H_{kv}$ | KV head count |

---