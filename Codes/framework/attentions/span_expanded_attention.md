# Span-Expanded Attention: External Memory Span Expansion with Parallel HyLoRA Adaptation

## 1. Abstract & Theoretical Motivation

Extending the receptive field of Transformer architectures beyond the physical training sequence length typically incurs severe compute and memory activation costs. Span-Expanded Attention addresses this challenge by combining external Key-Value (KV) memory span expansion with parallel Hybrid Low-Rank Adaptation (HyLoRA).

Rather than recomputing representations over long context windows, queries attend directly to pre-computed, external KV memory buffers of length $L_{\text{span}}$. To maintain domain adaptability and prevent capacity degradation when attending to external memory, a parallel low-rank residual pathway (HyLoRA) modulates hidden states concurrently with attention aggregation. This decoupled dual-pathway architecture enables zero-shot context expansion alongside parameter-efficient fine-tuning.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ denote the input hidden activation tensor, where $B$ is batch size, $L$ is input sequence length, and $D$ is model dimension. Let $H$ be the number of attention heads, $d_k = D / H$ the head dimension, and $r$ the rank of the HyLoRA adaptation bottleneck.

### Multi-Head Query Projection:
\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad W_Q \in \mathbb{R}^{D \times D}
\]
Reshaped into multi-head tensor format: $Q \in \mathbb{R}^{B \times H \times L \times d_k}$.

### Dual-Mode Key-Value Resolution:
1. **Cached External Memory Span Mode**:
   If an external cached memory tuple $(K_{\text{cache}}, V_{\text{cache}}) \in \mathbb{R}^{B \times L_{\text{span}} \times d_k} \times \mathbb{R}^{B \times L_{\text{span}} \times d_k}$ is provided:
   \[
   K = \text{Expand}(K_{\text{cache}}, \text{shape}=[B, H, L_{\text{span}}, d_k])
   \]
   \[
   V = \text{Expand}(V_{\text{cache}}, \text{shape}=[B, H, L_{\text{span}}, d_k])
   \]
2. **Local Self-Attention Mode**:
   If no cache is provided ($L_{\text{span}} = L$):
   \[
   K = X W_K \in \mathbb{R}^{B \times H \times L \times d_k}, \quad V = X W_V \in \mathbb{R}^{B \times H \times L \times d_k}
   \]
   where $W_K, W_V \in \mathbb{R}^{D \times D}$.

### Span-Expanded Attention Matrix Calculation:
The attention score matrix $S \in \mathbb{R}^{B \times H \times L \times L_{\text{span}}}$ is computed as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot K_{b, h, j, m}
\]

Incorporating additive mask $M \in \mathbb{R}^{B \times 1 \times L \times L_{\text{span}}}$ and row-wise Softmax:

\[
\widetilde{S}_{b, h, i, j} = S_{b, h, i, j} + M_{b, 1, i, j}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (\widetilde{S}_{b, h, i, j}) = \frac{\exp(\widetilde{S}_{b, h, i, j})}{\sum_{l=1}^{L_{\text{span}}} \exp(\widetilde{S}_{b, h, i, l})}
\]

If head mask $H_{\text{mask}} \in \mathbb{R}^{B \times H \times L \times L_{\text{span}}}$ is provided: $\widehat{A} = A \odot H_{\text{mask}}$.

The attention context aggregation yields:

\[
O_{\text{attn}, b, h, i, m} = \sum_{j=1}^{L_{\text{span}}} \widehat{A}_{b, h, i, j} \cdot V_{b, h, j, m} \implies O_{\text{attn}} \in \mathbb{R}^{B \times L \times D}
\]

### Parallel HyLoRA Residual Adaptation:
Concurrently with attention, the input activation $X$ passes through a low-rank bottleneck:

\[
O_{\text{LoRA}} = (X W_A) W_B \in \mathbb{R}^{B \times L \times D}
\]

where $W_A \in \mathbb{R}^{D \times r}$ and $W_B \in \mathbb{R}^{r \times D}$.

### Fused Output Combination:
The contextual output and LoRA adaptation residuals are combined before output projection:

\[
O_{\text{fused}} = O_{\text{attn}} + O_{\text{LoRA}} \in \mathbb{R}^{B \times L \times D}
\]
\[
Y = O_{\text{fused}} W_O + b_O \in \mathbb{R}^{B \times L \times D}, \quad W_O \in \mathbb{R}^{D \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, D, bias=False) [W_q] ──> Q: [B, L, D] ──> view & transpose ──> Q: [B, H, L, d_k]
  │
  ├──> IF kv_cache IS NOT NONE:
  │      k_cache, v_cache [B, L_span, d_k] ──> unsqueeze & expand ──> K, V: [B, H, L_span, d_k]
  │    ELSE:
  │      Linear(D, D) [W_k, W_v] ──> view & transpose ──> K, V: [B, H, L, d_k]
  │
  ├──> MatMul: Q @ K.transpose(-2, -1) / sqrt(d_k) ──> scores: [B, H, L, L_span]
  ├──> Add attention_mask ──> [B, H, L, L_span]
  ├──> Softmax(dim=-1) & Head Mask ──> attn: [B, H, L, L_span]
  ├──> MatMul: attn @ V ──> context: [B, H, L, d_k]
  ├──> Transpose & Contiguous View ──> O_attn: [B, L, D]
  │
  ├──> IF use_lora IS TRUE:
  │      Linear(D, r, bias=False) [lora_A] ──> [B, L, r]
  │      Linear(r, D, bias=False) [lora_B] ──> O_LoRA: [B, L, D]
  │      Add Residual: O_fused = O_attn + O_LoRA ──> [B, L, D]
  │
  └──> Linear(D, D) [out_proj] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Computational Overhead & Workload (FLOPs & MACs)

Let $B$ be batch size, $L$ sequence length, $L_{\text{span}}$ memory span length, $D$ model dimension, $H$ heads, $d_k = D / H$, and $r$ LoRA rank.

1. **Linear Projections**:
   - $W_Q$: $2 B L D^2$ FLOPs ($B L D^2$ MACs).
   - Local $W_K, W_V$ (when no cache): $4 B L D^2$ FLOPs ($2 B L D^2$ MACs).
   - Output Projection ($W_O$): $2 B L D^2$ FLOPs ($B L D^2$ MACs).

2. **Parallel HyLoRA Branch ($W_A, W_B$)**:
   - $W_A$: $[B, L, D] \times [D, r] \implies 2 B L D r$ FLOPs ($B L D r$ MACs).
   - $W_B$: $[B, L, r] \times [r, D] \implies 2 B L D r$ FLOPs ($B L D r$ MACs).
   - Total HyLoRA FLOPs: $4 B L D r$.

3. **Core Span Attention Operations ($Q K^\top$ & $A V$)**:
   - $Q K^\top$: $[B, H, L, d_k] \times [B, H, d_k, L_{\text{span}}] \implies 2 B H L L_{\text{span}} d_k = 2 B L L_{\text{span}} D$ FLOPs.
   - Softmax overhead: $\approx 5 B H L L_{\text{span}}$ FLOPs.
   - $A V$: $[B, H, L, L_{\text{span}}] \times [B, H, L_{\text{span}}, d_k] \implies 2 B L L_{\text{span}} D$ FLOPs.

### Total FLOPs & MACs Summary:
- **Total MACs (Cached Mode)**: $2 B L D^2 + 2 B L D r + 2 B L L_{\text{span}} D$
- **Total FLOPs (Cached Mode)**: $4 B L D^2 + 4 B L D r + 4 B L L_{\text{span}} D + 5 B H L L_{\text{span}}$
- **Parameter Footprint**: $4 D^2 + 2 D r$ (Adding $2 D r$ parameters for HyLoRA, e.g., for $D=768, r=8$, adding $12,288$ parameters).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Unprojected Cache Dimension Mismatch
In `SpanExpandedAttention`, external cache expansion relies on `k_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)`. If `k_cache` is passed with shape $[B, L_{\text{span}}, d_k]$ instead of $[B, L_{\text{span}}, D]$, broadcasting fails. If $K_{\text{cache}}$ was generated with a different projection matrix, domain misalignment occurs.

### HyLoRA Rank Bottleneck
Because $O_{\text{LoRA}} = X W_A W_B$, the rank of the parallel adaptation stream is bounded by $r$:

\[
\text{rank}(O_{\text{LoRA}}) \le \min(L, r) = r
\]

If $r \ll D$ (e.g., $r=4$), the adaptation pathway cannot represent high-frequency feature transformations, leading to underfitting during fine-tuning on complex out-of-distribution targets.

### High Memory Bandwidth Pressure under Long Spans
When $L_{\text{span}} \gg 4096$, storing and fetching uncompressed external cache tensors $K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B \times H \times L_{\text{span}} \times d_k}$ from HBM saturates GPU memory bandwidth lanes.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Dual-Stream Triton Attention-LoRA Kernel Architecture
To maximize execution speedups, Triton kernels fuse the parallel HyLoRA GEMM computations directly with the FlashAttention score calculation.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Q [B, H, L, d_k]                                       │
      │ K_cache [B, H, L_span, d_k], V_cache [B, H, L_span, d_k]│
      │ X [B, L, D], W_A [D, r], W_B [r, D]                    │
      └───────────────────────────┬────────────────────────────┘
                                  │ Concurrent Tile Loading
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Compute LoRA Stream: O_lora = X_tile @ W_A @ W_B              │
 │ 2. Compute Span Attention Stream:                                │
 │    S_tile = Q_tile @ K_cache_tile.T / sqrt(d_k)                  │
 │    P_tile = Softmax(S_tile)                                      │
 │    O_attn = P_tile @ V_cache_tile                                │
 │ 3. Fuse In-Register Residual: O_fused = O_attn + O_lora          │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Write Directly to Output Projection
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Output O [B, L, D]                                     │
      └────────────────────────────────────────────────────────┘
```

### Triton Implementation Guidelines:
1. **Asynchronous Memory Cache Streaming**:
   Utilize NVIDIA Hopper / Ampere Tensor Memory Accelerator (TMA) asynchronous copy instructions to prefetch external memory tiles $K_{\text{cache}}, V_{\text{cache}}$ into shared memory, hiding DRAM latency behind warp compute pipelines.
2. **Double-Buffered LoRA GEMM Fusion**:
   Compute the low-rank projection $X W_A W_B$ during the initial SRAM tile load of $Q$, overlapping matrix multiplications across SM warp groups.
3. **KV Cache Compression Interface**:
   Quantize $K_{\text{cache}}$ and $V_{\text{cache}}$ to FP8 (`fp8_e4m3fn`) or INT8 in HBM, executing dynamic de-quantization inside Triton SRAM registers prior to matrix multiplication.