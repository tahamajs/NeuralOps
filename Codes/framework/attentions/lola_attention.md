# LoLA Attention: Factorized Low-Rank Projection with Triple-Memory Cache

## 1. Abstract & Theoretical Motivation
Standard Transformer self-attention computes query-key affinity over full history sequences, scaling quadratically $\mathcal{O}(N^2)$ in time and memory. Low-Rank Linear Attention with Sparse Caching (LoLA Attention) combines low-rank linear projection factorizations with a hybrid triple-memory state caching system.

LoLA factorizes high-dimensional projection weight matrices $\mathbf{W} \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ into rank-constrained pairs $\mathbf{W} \approx \mathbf{U} \mathbf{V}$ where $\mathbf{U} \in \mathbb{R}^{D_{\text{out}} \times r}, \mathbf{V} \in \mathbb{R}^{r \times D_{\text{in}}}$. To maintain long-context memory without quadratic explosion, LoLA maintains three persistent key-value memory partitions: (1) Recent Sliding Window Cache, (2) Subsampled Sparse Historical Cache, and (3) SVD Truncated Low-Rank Buffer. This hybrid mechanism maintains bounded memory consumption while capturing long-range multi-scale dependencies.

---

## 2. Mathematical Derivations & Formulations

Given input hidden state $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, low-rank linear transformations compute query, key, value, and output projections with rank $r \ll D$:

\[
\mathbf{Q} = (\mathbf{X} \mathbf{V}_Q^\top) \mathbf{U}_Q^\top + \mathbf{b}_Q \in \mathbb{R}^{B \times H \times N \times d}
\]
\[
\mathbf{K} = (\mathbf{X} \mathbf{V}_K^\top) \mathbf{U}_K^\top + \mathbf{b}_K \in \mathbb{R}^{B \times H \times N \times d}
\]
\[
\mathbf{V} = (\mathbf{X} \mathbf{V}_V^\top) \mathbf{U}_V^\top + \mathbf{b}_V \in \mathbb{R}^{B \times H \times N \times d}
\]

where $\mathbf{V}_{(\cdot)} \in \mathbb{R}^{r \times D}$ and $\mathbf{U}_{(\cdot)} \in \mathbb{R}^{D \times r}$.

### Triple-Memory System Caching ($\mathcal{C}$)
LoLA aggregates historical key-value states into a unified memory context $\mathbf{K}_{\text{comb}} = [\mathbf{K} \;\|\; \mathbf{K}_{\text{cache}}]$ and $\mathbf{V}_{\text{comb}} = [\mathbf{V} \;\|\; \mathbf{V}_{\text{cache}}]$ across three memory layers:
1. **Recent Cache ($\mathcal{C}_{\text{recent}}$)**: Sliding window of size $W_{\text{max}}$ storing raw recent keys/values:
   \[
   \mathcal{C}_{\text{recent}} = \{ \mathbf{k}_t, \mathbf{v}_t \}_{t=N - W_{\text{max}} + 1}^N
   \]
2. **Subsampled Sparse Cache ($\mathcal{C}_{\text{sparse}}$)**: Uniformly subsampled stride-4 tokens from older history:
   \[
   \mathcal{C}_{\text{sparse}} = \{ \mathbf{k}_{4i}, \mathbf{v}_{4i} \}_{i=1}^{N/4}
   \]
3. **SVD Low-Rank Compressed Buffer ($\mathcal{C}_{\text{lr}}$)**: Truncated SVD top-$k_{\text{rank}}$ reconstruction of past states:
   \[
   \mathbf{K}_{\text{lr}} = \mathbf{U}_{:k_{\text{rank}}} \boldsymbol{\Sigma}_{:k_{\text{rank}}} \mathbf{V}_{:k_{\text{rank}}}^\top \in \mathbb{R}^{k_{\text{rank}} \times d}
   \]

### Scaled Multi-Memory Attention Scores
Scores are computed separately over current inputs and retrieved memory partitions, weighted by dynamic scaling parameters $\gamma_{\text{lr}}$ and $\gamma_{\text{cache}}$:

\[
\mathbf{S}_{\text{current}} = \gamma_{\text{lr}} \cdot \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}, \quad \mathbf{S}_{\text{cache}} = \gamma_{\text{cache}} \cdot \frac{\mathbf{Q} \mathbf{K}_{\text{cache}}^\top}{\sqrt{d}}
\]
\[
\mathbf{S}_{\text{concat}} = [\mathbf{S}_{\text{current}} \;\|\; \mathbf{S}_{\text{cache}}] \in \mathbb{R}^{B \times H \times N \times (N + C)}
\]

After applying softmax over concatenated sequence dimension $(N + C)$:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}_{\text{concat}} + \mathbf{M}_{\text{mask}}, \text{dim}=-1) \in \mathbb{R}^{B \times H \times N \times (N + C)}
\]
\[
\mathbf{O} = (\mathbf{A} \mathbf{V}_{\text{comb}}) \mathbf{W}_{\text{out}} \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Factorized Low-Rank Projections (U, V):
  │    ├──> Q = (X @ V_q^T) @ U_q^T + b_q ─────────────> [B, H, N, d]
  │    ├──> K = (X @ V_k^T) @ U_k^T + b_k ─────────────> [B, H, N, d]
  │    └──> V = (X @ V_v^T) @ U_v^T + b_v ─────────────> [B, H, N, d]
  │
  ├──> Retrieve Memory Buffers from Cache:
  │    K_cache, V_cache = Cache.get_cached() ──────────> [C, d]
  │    Expand to Batch & Heads ────────────────────────> [B, H, C, d]
  │
  ├──> Dual Score Multiplication:
  │    ├──> S_current = gamma_lr * (Q @ K^T) / sqrt(d) ─> [B, H, N, N]
  │    ├──> S_cache = gamma_cache * (Q @ K_c^T)/sqrt(d)-> [B, H, N, C]
  │    └──> S_concat = Cat([S_current, S_cache]) ──────> [B, H, N, N + C]
  │
  ├──> Softmax & Value Aggregation:
  │    A = Softmax(S_concat) ──────────────────────────> [B, H, N, N + C]
  │    V_comb = Cat([V, V_cache]) ─────────────────────> [B, H, N + C, d]
  │    Context = A @ V_comb ───────────────────────────> [B, H, N, d]
  │
  └──> Update Cache with SVD & Low-Rank Out Projection:
       Cache.update(K, V)
       Y = (Context @ V_out^T) @ U_out^T ──────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Low-Rank Inner Product ($x V_Q^\top$) | `X: [B, N, D]`, `V_q: [r, D]` | `Vx: [B, N, r]` |
| 2 | Low-Rank Outer Product ($Vx U_Q^\top$) | `Vx: [B, N, r]`, `U_q: [D, r]` | `Q: [B, N, D]` -> `[B, H, N, d]` |
| 3 | Memory Cache Retrieval | Memory buffers | `K_cache, V_cache: [B, H, C, d]` |
| 4 | Current Sequence Scores | `Q: [B, H, N, d]`, `K^T: [B, H, d, N]` | `S_lr: [B, H, N, N]` |
| 5 | Cache Sequence Scores | `Q: [B, H, N, d]`, `K_cache^T: [B, H, d, C]` | `S_cache: [B, H, N, C]` |
| 6 | Concatenate Attention Scores | `S_lr: [B, H, N, N]`, `S_cache: [B, H, N, C]` | `S_concat: [B, H, N, N + C]` |
| 7 | Softmax Normalization | `S_concat: [B, H, N, N + C]` | `A: [B, H, N, N + C]` |
| 8 | Value Memory Product | `A: [B, H, N, N + C]`, `V_comb: [B, H, N + C, d]` | `Context: [B, H, N, d]` |
| 9 | Output Low-Rank Linear | `Context: [B, N, D]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Low-Rank Linear Projections ($W_Q, W_K, W_V, W_O$)**:
   - Standard linear requires $2 B N D^2$ per projection.
   - Low-Rank factorized requires $2 \times (2 B N D r) = 4 B N D r$ FLOPs.
   - Total for 4 projections: $16 B N D r$ FLOPs.
2. **Current Sequence Score Product ($\mathbf{Q} \mathbf{K}^\top$)**: $2 B H N^2 d = 2 B N^2 D$ FLOPs.
3. **Cache Sequence Score Product ($\mathbf{Q} \mathbf{K}_{\text{cache}}^\top$)**: $2 B H N C d = 2 B N C D$ FLOPs.
4. **Value Memory Product ($\mathbf{A} \mathbf{V}_{\text{comb}}$)**: $2 B H N (N + C) d = 2 B N (N + C) D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 16 B N D r + 4 B N^2 D + 4 B N C D
\]

When $r \ll D$ (e.g., $r=32, D=768$), projection compute drops significantly compared to standard attention's $8 B N D^2$.

### Memory Footprint & Tensor Store
- **Parameters**: Factorized tensors $\mathbf{U}, \mathbf{V}$ require $4 \times (D r + D r) = 8 D r$ parameters (vs $4 D^2$).
- **Activation Memory**:
  - Combined Keys/Values: $2 B H (N + C) d = 2 B (N + C) D$ floats.
  - Concatenated Scores: $B H N (N + C)$ floats.
  - Cache Buffers ($\mathcal{C}_{\text{recent}}, \mathcal{C}_{\text{sparse}}, \mathcal{C}_{\text{lr}}$): $(W_{\text{max}} + \frac{W_{\text{max}}}{2} + k_{\text{rank}}) D$ floats.
  - Peak Memory: $\mathcal{O}(B N D + B H N (N + C))$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **CPU/GPU SVD Inversion Latency**: Truncated SVD calls `torch.linalg.svd` inside `LowRankCache.update` run on CPU/GPU tensors synchronously during training/inference steps. SVD operations disrupt CUDA stream parallelism and introduce significant latency.
2. **Cache Memory Leak in Single-Batch Ingestion**: The current cache implementation indexes hardcoded batch slice `K[0, 0]`, limiting multi-batch inference and leading to cross-batch memory bleed when $B > 1$.
3. **Low-Rank Expressivity Ceiling**: Highly factorized projection rank $r \ll D$ risks under-fitting fine-grained spatial token representations in early model layers.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
Naive PyTorch executes `torch.cat` to concatenate current sequence tensors with historical cache buffers $K_{\text{cache}}$, causing dynamic DRAM allocations and memory fragmentation.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused Dual-Stream Memory Flash Kernel**: Implement a Triton kernel that loads current query blocks $[BLOCK\_N, d]$ into SRAM, while concurrently streaming current keys and persistent cache buffers from separate GPU DRAM pointers without materializing concatenated tensors.

```python
# Triton Pseudocode for Dual-Stream Cache LoLA Kernel
@triton.jit
def _lola_dual_stream_kernel(
    Q_ptr, K_ptr, V_ptr, K_cache_ptr, V_cache_ptr, Out_ptr,
    N, C, D, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr
):
    # Load Query tile [BLOCK_N, d]
    q = tl.load(Q_ptr + offsets_q)
    
    # Stream current keys K [BLOCK_N, d] -> compute S_lr in SRAM
    k = tl.load(K_ptr + offsets_k)
    s_lr = tl.dot(q, tl.trans(k)) * scale * gamma_lr
    
    # Stream cache keys K_cache [BLOCK_C, d] -> compute S_cache in SRAM
    k_c = tl.load(K_cache_ptr + offsets_kc)
    s_cache = tl.dot(q, tl.trans(k_c)) * scale * gamma_cache
    
    # Online Softmax over concatenated logits [S_lr, S_cache] inside registers
```

2. **Asynchronous GPU SVD Power Iteration**: Replace exact `torch.linalg.svd` calls with 2 iterations of Power Method matrix decomposition running inside GPU registers during backward execution to update the low-rank cache asynchronously.