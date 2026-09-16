# 3. Background

## 3.1 Standard Scaled Dot-Product Attention

$$
S = Q K^\top / \sqrt{d_h}, \quad P = \text{softmax}(S), \quad O = P V
\tag{3.1}
$$

with $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$.

**Memory.** $S \in \mathbb{R}^{B \times H \times L \times L}$, giving 
$\mathcal{O}(L^2)$.

## 3.2 FlashAttention Tiling

Partition $Q$ into tiles $Q_1, \ldots, Q_T$ and $K, V$ into tiles 
$K_1, \ldots, K_T$.

**Per-query-tile, per-key-tile computation:**

For each $(i, t)$:

$$
S_{it} = Q_i K_t^\top / \sqrt{d_h}
\tag{3.2}
$$

Update running statistics:

$$
m_i^{(t)} = \max(m_i^{(t-1)}, \max_j S_{it, j})
\tag{3.3}
$$

$$
\ell_i^{(t)} = \ell_i^{(t-1)} e^{m_i^{(t-1)} - m_i^{(t)}} + \sum_j e^{S_{it, j} - m_i^{(t)}}
\tag{3.4}
$$

$$
O_i^{(t)} = O_i^{(t-1)} \cdot \frac{\ell_i^{(t-1)} e^{m_i^{(t-1)} - m_i^{(t)}}}{\ell_i^{(t)}} + \frac{\sum_j e^{S_{it, j} - m_i^{(t)}} V_t}{\ell_i^{(t)}}
\tag{3.5}
$$

Only $Q_i, K_t, V_t, m_i, \ell_i, O_i$ are kept in SRAM. No full $S$ tensor.

## 3.3 FlashAttention-3 on Hopper

### 3.3.1 TMA (Tensor Memory Accelerator)

Hopper introduces **TMA**, a hardware unit for bulk tensor copy between HBM 
and SRAM. FA3 issues TMA loads for $K_t, V_t$ tiles in the background.

### 3.3.2 Warp Specialization

- **Producer warps:** issue TMA loads.
- **Consumer warps:** perform matmul and softmax.
- **Epilogue warps:** store outputs.

These run concurrently, hiding latency.

### 3.3.3 FP8 Tensor Cores

Hopper Tensor Cores support FP8 with `e4m3fn` (4 exp, 3 mantissa) and 
`e5m2` (5 exp, 2 mantissa). FA3 uses `e4m3fn` for Q, K, and `e5m2` 
for V.

## 3.4 PyTorch SDPA

PyTorch `F.scaled_dot_product_attention` dispatches to one of:

1. **FlashAttention-2** backend (when available).
2. **Memory-efficient attention** backend.
3. **Math** fallback (slowest).

For our fallback, we use SDPA with `is_causal` when appropriate.

## 3.5 Grouped-Query Attention (GQA)

For GQA models (Qwen, LLaMA):

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times L \times d_h}, \quad
K_\text{expanded} = \text{repeat\_kv}(K, H / H_{kv})
\tag{3.6}
$$

FlashAttention-3 has native GQA support via `flash_attn_func` with different 
$H$ and $H_{kv}$.

## 3.6 Rotary Position Embeddings (RoPE)

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.7}
$$

Applied before attention. Compatible with FA3.

## 3.7 LoRA

$$
\Delta W = \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.8}
$$

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $Q, K, V$ | Query, Key, Value |
| $S$ | Attention scores |
| $P$ | Softmax probs |
| $m_i$ | Running max |
| $\ell_i$ | Running sum |
| $B, H, L, d_h$ | Batch, Heads, Length, Head dim |
| $H_{kv}$ | KV head count |
```

---