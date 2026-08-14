# Grouped-Query Attention (GQA): Subspace-Grouped Key-Value Sharing for High-Throughput Inference

## 1. Abstract & Theoretical Motivation

Grouped-Query Attention (GQA), introduced by Ainslie et al. (2023), presents a structural paradigm that bridges Multi-Head Attention (MHA) and Multi-Query Attention (MQA). During autoregressive language model inference, fetching the Key-Value (KV) cache from High-Bandwidth Memory (HBM) to on-chip SRAM creates an acute memory bandwidth bottleneck. While MQA reduces the KV cache footprint by sharing a single KV head across all query heads, it can degrade representation capacity and model quality. 

GQA resolves this trade-off by partitioning the $H_Q$ query heads into $H_{KV}$ distinct groups ($1 < H_{KV} < H_Q$), where each group shares a dedicated Key-Value head pair. The compression ratio $\gamma = H_Q / H_{KV}$ balances computational expressiveness and memory traffic. GQA achieves quality comparable to full MHA while substantially diminishing KV cache size, improving batch processing capability and inference decoding throughput.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ denote the input hidden state tensor, where $B$ is batch size, $L$ is sequence length, and $D$ is model dimension. The model features $H_Q$ query heads and $H_{KV}$ key-value heads, with head dimension $d_k = D / H_Q$. The key-value group ratio is defined as:

\[
\gamma = \frac{H_Q}{H_{KV}} \in \mathbb{Z}^+
\]

The linear projections map hidden states into query, key, and value spaces:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times (H_Q \cdot d_k)}, \quad W_Q \in \mathbb{R}^{D \times (H_Q \cdot d_k)}
\]
\[
K = X W_K \in \mathbb{R}^{B \times L \times (H_{KV} \cdot d_k)}, \quad W_K \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\]
\[
V = X W_V \in \mathbb{R}^{B \times L \times (H_{KV} \cdot d_k)}, \quad W_V \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\]

Reshaping into multi-head layouts yields:

\[
Q \in \mathbb{R}^{B \times H_Q \times L \times d_k}, \quad K \in \mathbb{R}^{B \times H_{KV} \times L \times d_k}, \quad V \in \mathbb{R}^{B \times H_{KV} \times L \times d_k}
\]

To compute attention scores across all query heads, key and value tensors are expanded along the head dimension using standard repeat-interleave operations:

\[
K_{\text{shared}} = \text{RepeatInterleave}(K, \text{repeats}=\gamma, \text{dim}=1) \in \mathbb{R}^{B \times H_Q \times L \times d_k}
\]
\[
V_{\text{shared}} = \text{RepeatInterleave}(V, \text{repeats}=\gamma, \text{dim}=1) \in \mathbb{R}^{B \times H_Q \times L \times d_k}
\]

Specifically, for a query head $h \in \{0, 1, \dots, H_Q - 1\}$, its corresponding Key-Value index $g(h)$ is given by:

\[
g(h) = \left\lfloor \frac{h}{\gamma} \right\rfloor \in \{0, 1, \dots, H_{KV} - 1\}
\]

The scaled dot-product attention matrix $S \in \mathbb{R}^{B \times H_Q \times L \times L}$ per head $h$ is derived as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot K_{b, g(h), j, m}
\]

Incorporating the universal attention mask $M \in \mathbb{R}^{B \times 1 \times L \times L}$:

\[
\widetilde{S}_{b, h, i, j} = S_{b, h, i, j} + M_{b, 1, i, j}
\]

Applying row-wise Softmax over key sequence positions $j$:

\[
A_{b, h, i, j} = \text{Softmax}_j (\widetilde{S}_{b, h, i, j}) = \frac{\exp\left(\widetilde{S}_{b, h, i, j} - \max_k \widetilde{S}_{b, h, i, k}\right)}{\sum_{l=1}^L \exp\left(\widetilde{S}_{b, h, i, l} - \max_k \widetilde{S}_{b, h, i, k}\right)}
\]

After applying dropout and optional head masking, the context representation $O \in \mathbb{R}^{B \times H_Q \times L \times d_k}$ is aggregated:

\[
O_{b, h, i, m} = \sum_{j=1}^L A_{b, h, i, j} \cdot V_{b, g(h), j, m}
\]

Finally, context heads are concatenated and projected to the output space via $W_O \in \mathbb{R}^{(H_Q \cdot d_k) \times D}$:

\[
Y = \text{Concat}(O_1, O_2, \dots, O_{H_Q}) \cdot W_O \in \mathbb{R}^{B \times L \times D}
\]

### Group-Averaged Weight Initialisation:
When initializing GQA from a pretrained MHA checkpoint, the key and value projection weights are mean-pooled across query head groups:

\[
W_K^{\text{GQA}}[g, :, :] = \frac{1}{\gamma} \sum_{r=0}^{\gamma - 1} W_K^{\text{MHA}}[g \cdot \gamma + r, :, :]
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, H_Q * d_k) [W_q] ──> Q: [B, L, H_Q * d_k] ──> view & transpose ──> Q: [B, H_Q, L, d_k]
  ├──> Linear(D, H_KV * d_k) [W_k] ──> K: [B, L, H_KV * d_k] ──> view & transpose ──> K: [B, H_KV, L, d_k]
  └──> Linear(D, H_KV * d_k) [W_v] ──> V: [B, L, H_KV * d_k] ──> view & transpose ──> V: [B, H_KV, L, d_k]
                                │
                                ├──> repeat_interleave(γ, dim=1) ──> K_shared: [B, H_Q, L, d_k]
                                ├──> repeat_interleave(γ, dim=1) ──> V_shared: [B, H_Q, L, d_k]
                                │
                                ├──> MatMul: Q @ K_shared.transpose(-2, -1) ──> [B, H_Q, L, L]
                                ├──> Scale: * (d_k ** -0.5) ──> [B, H_Q, L, L]
                                ├──> apply_attention_mask(mask) ──> [B, H_Q, L, L]
                                ├──> Softmax(dim=-1) ──> attn: [B, H_Q, L, L]
                                ├──> Dropout(p) & Head Mask ──> [B, H_Q, L, L]
                                │
                                └──> MatMul: attn @ V_shared ──> context: [B, H_Q, L, d_k]
                                        │
                                        └──> transpose(1, 2) ──> [B, L, H_Q, d_k]
                                        └──> contiguous().view([B, L, H_Q * d_k])
                                        └──> Linear(H_Q * d_k, D) [W_o] ──> Output: [B, L, D]
```

### Shape Tracing Table:
1. `hidden_states`: $[B, L, D]$
2. `self.query(hidden_states)`: $[B, L, H_Q \cdot d_k]$
3. `self.key(hidden_states)`: $[B, L, H_{KV} \cdot d_k]$
4. `self.value(hidden_states)`: $[B, L, H_{KV} \cdot d_k]$
5. `q`: $[B, H_Q, L, d_k]$
6. `k`, `v`: $[B, H_{KV}, L, d_k]$
7. `k_shared`, `v_shared` (after repeat): $[B, H_Q, L, d_k]$
8. `scores = Q @ K_shared.T`: $[B, H_Q, L, L]$
9. `attn = Softmax(scores)`: $[B, H_Q, L, L]$
10. `context = attn @ V_shared`: $[B, H_Q, L, d_k]$
11. `out_proj(context)`: $[B, L, D]$

---

## 4. Analytical Complexity Profiling

### Computational Workload (FLOPs & MACs)

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H_Q$ query heads, $H_{KV}$ KV heads, $\gamma = H_Q / H_{KV}$, and $d_k = D / H_Q$.

1. **Projection Computations**:
   - $W_Q$: $[B, L, D] \times [D, D] \implies 2 B L D^2$ FLOPs ($B L D^2$ MACs).
   - $W_K, W_V$: $2 \times [B, L, D] \times [D, \frac{D}{\gamma}] \implies 2 \times 2 B L \frac{D^2}{\gamma} = \frac{4 B L D^2}{\gamma}$ FLOPs ($\frac{2 B L D^2}{\gamma}$ MACs).

2. **Attention Score Matrix & Context ($Q K^\top$ & $A V$)**:
   - $Q K^\top$: $[B, H_Q, L, d_k] \times [B, H_Q, d_k, L] \implies 2 B L^2 D$ FLOPs.
   - Softmax overhead: $\approx 5 B H_Q L^2$ FLOPs.
   - $A V$: $[B, H_Q, L, L] \times [B, H_Q, L, d_k] \implies 2 B L^2 D$ FLOPs.

3. **Output Projection ($W_O$)**:
   - $[B, L, D] \times [D, D] \implies 2 B L D^2$ FLOPs ($B L D^2$ MACs).

### Total Compute Totals:
- **Total MACs**: $2 B L D^2 \left(1 + \frac{1}{\gamma}\right) + 2 B L^2 D$
- **Total FLOPs**: $4 B L D^2 \left(1 + \frac{1}{\gamma}\right) + 4 B L^2 D$

### KV Cache Memory Footprint (Autoregressive Generation $L=1$):
- **Full MHA Cache Size**: $2 \times B \times L_{\text{past}} \times H_Q \times d_k \times \text{bytes} = 2 B L_{\text{past}} D \times \text{bytes}$
- **GQA Cache Size**: $2 \times B \times L_{\text{past}} \times H_{KV} \times d_k \times \text{bytes} = \frac{2 B L_{\text{past}} D}{\gamma} \times \text{bytes}$
- **KV Cache Reduction Factor**: Exact $\gamma = \frac{H_Q}{H_{KV}}$ savings in memory capacity and bandwidth usage.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Explicit Repetition Memory Overhead
In standard naive PyTorch implementations, calling `k.repeat_interleave(repeat, dim=1)` explicitly materializes a tensor of size $[B, H_Q, L, d_k]$. This destroys the memory savings during the forward pass of training, adding unnecessary memory allocation latency and SRAM-to-HBM write overhead.

### Subspace Capacity Saturation
When the grouping factor $\gamma$ is large (e.g., $\gamma = 16$ or $H_{KV} = 1$, corresponding to MQA), a single Key-Value subspace must serve multiple distinct query projections. If query heads attempt to extract orthogonal contextual relationships, the shared KV head suffers from capacity saturation, leading to rank collapse in key representations and reduced model accuracy on long-context tasks.

### Gradient Conflicts in Group-Averaged Adaptations
When converting a pretrained standard MHA model to GQA via group averaging (`copy_weights`), the loss surface can exhibit sharp gradient spikes during early fine-tuning steps due to the sudden collapse of independent KV directions into a single group mean.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Broadcasted Stride Indexing in FlashAttention Kernels
To maximize hardware utilization, Triton fused attention kernels must operate directly on the un-expanded $K, V$ tensors $[B, H_{KV}, L, d_k]$, eliminating the need for `repeat_interleave`.

```
      Logical Query Head Index (h)
     ┌────────────────────────────┐
     │ h ∈ {0, 1, ..., H_Q - 1}   │
     └──────────────┬─────────────┘
                    │ Integer Division: g = h // γ
                    ▼
     ┌────────────────────────────┐
     │ g ∈ {0, 1, ..., H_KV - 1}  │
     └──────────────┬─────────────┘
                    │
   ┌────────────────┴────────────────┐
   │ Loads Q Tile [BLOCK_M, d_k]     │  From Q Head Pointer (h)
   │ Loads K Tile [BLOCK_N, d_k]     │  From K Head Pointer (g = h // γ)
   │ Loads V Tile [BLOCK_N, d_k]     │  From V Head Pointer (g = h // γ)
   └────────────────┬────────────────┘
                    │ Fused Online Softmax & Accumulation
                    ▼
   ┌─────────────────────────────────┐
   │ Writes O Tile [BLOCK_M, d_k]    │  To Output Head Pointer (h)
   └─────────────────────────────────┘
```

### Triton Kernel Implementation Design:
1. **Strided Memory Addressing**:
   Define key and value tensor strides such that query head `h` reads key head `h // γ`. The pointer arithmetic for the key tile is:
   ```python
   k_ptr = K + batch_id * stride_k_b + (head_id // GROUP_SIZE) * stride_k_h + tile_n * stride_k_n
   ```
2. **SRAM Register Reuse**:
   Because $\gamma$ query warps share the exact same key/value tiles, the key tile loaded into L1 SRAM / Shared Memory can be reused across warp groups, maximizing cache hit rates and eliminating redundant HBM reads.
3. **Decoded Inference Kernel**:
   For single-step autoregressive generation ($L_{\text{query}}=1$), write a specialized split-KV Triton decoding kernel that parallelizes sequence length tiles $L_{\text{key}}$ across Thread Blocks, performing a parallel reduction across blocks to achieve peak DRAM bandwidth performance.