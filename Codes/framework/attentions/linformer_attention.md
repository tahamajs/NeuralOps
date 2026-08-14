# Linformer Attention: Linear-Complexity Low-Rank Sequence Projection

## 1. Abstract & Theoretical Motivation
The computational bottleneck of standard multi-head self-attention lies in the quadratic scaling of time and space complexities \(\mathcal{O}(N^2)\) with respect to sequence length \(N\). Linformer (Wang et al., 2020) circumvents this bottleneck by leveraging a fundamental theoretical insight: the attention weight matrix \(\mathbf{A} \in \mathbb{R}^{N \times N}\) is typically low-rank in empirical deep models. 

Mathematically, Linformer projects the sequence length dimensions of the Key (\(\mathbf{K}\)) and Value (\(\mathbf{V}\)) matrices from \(N\) down to a fixed, hyperparameterized low-rank dimension \(k \ll N\) using learnable linear projection operators \(\mathbf{E}, \mathbf{F} \in \mathbb{R}^{k \times N_{\text{max}}}\). By reducing the key-value sequence length prior to computing scaled dot-product attention, Linformer converts the quadratic \(\mathcal{O}(N^2 d)\) complexity into a linear relationship \(\mathcal{O}(N k d)\), permitting scalable inference and training across extended context windows.

---

## 2. Mathematical Derivations & Formulations

Given an input representation tensor \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), where \(B\) is batch size, \(N\) is sequence length, and \(D\) is the model hidden dimension, standard projections yield query, key, and value matrices for head \(h \in \{1, \dots, H\}\) with head dimension \(d = D / H\):

\[
\mathbf{Q}_h = \mathbf{X} \mathbf{W}_Q^{(h)} \in \mathbb{R}^{B \times N \times d}, \quad \mathbf{K}_h = \mathbf{X} \mathbf{W}_K^{(h)} \in \mathbb{R}^{B \times N \times d}, \quad \mathbf{V}_h = \mathbf{X} \mathbf{W}_V^{(h)} \in \mathbb{R}^{B \times N \times d}
\]

Linformer defines two sequence projection parameters \(\mathbf{E}, \mathbf{F} \in \mathbb{R}^{k \times N_{\text{max}}}\). For an active sequence length \(N \le N_{\text{max}}\), active slices \(\mathbf{E}_{\text{act}} = \mathbf{E}_{:, :N}\) and \(\mathbf{F}_{\text{act}} = \mathbf{F}_{:, :N}\) are extracted. First, token-level zero-masking is applied to prevent padding contamination:

\[
\widetilde{\mathbf{K}}_h = \mathbf{K}_h \odot \mathbf{M}_{\text{valid}}, \quad \widetilde{\mathbf{V}}_h = \mathbf{V}_h \odot \mathbf{M}_{\text{valid}}
\]

where \(\mathbf{M}_{\text{valid}} \in \{0, 1\}^{B \times 1 \times N \times 1}\). The sequence dimensions of keys and values are then linearly projected:

\[
\mathbf{K}_{\text{proj}, h} = \mathbf{E}_{\text{act}} \widetilde{\mathbf{K}}_h \in \mathbb{R}^{B \times k \times d}, \quad \mathbf{V}_{\text{proj}, h} = \mathbf{F}_{\text{act}} \widetilde{\mathbf{V}}_h \in \mathbb{R}^{B \times k \times d}
\]

Specifically, for each batch instance \(b\) and head \(h\):

\[
(\mathbf{K}_{\text{proj}, h})_{b, i, m} = \sum_{j=1}^N (\mathbf{E}_{\text{act}})_{i, j} (\widetilde{\mathbf{K}}_h)_{b, j, m}, \quad \forall i \in \{1, \dots, k\}, m \in \{1, \dots, d\}
\]

The scaled dot-product attention score tensor \(\mathbf{S}_h \in \mathbb{R}^{B \times N \times k}\) and attention weights \(\mathbf{A}_h \in \mathbb{R}^{B \times N \times k}\) are derived via:

\[
\mathbf{S}_h = \frac{\mathbf{Q}_h \mathbf{K}_{\text{proj}, h}^\top}{\sqrt{d}}, \quad \mathbf{A}_h = \operatorname{softmax}\left(\mathbf{S}_h, \text{dim}=-1\right)
\]

Contextual aggregation produces the head output \(\mathbf{O}_h \in \mathbb{R}^{B \times N \times d}\):

\[
\mathbf{O}_h = \left( \mathbf{A}_h \mathbf{V}_{\text{proj}, h} \right) \odot \mathbf{M}_{\text{valid}}
\]

Finally, multi-head contexts are concatenated and linearly projected back to hidden dimension \(D\):

\[
\mathbf{Y} = \operatorname{Concat}(\mathbf{O}_1, \dots, \mathbf{O}_H) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Tensor: X [B, N, D]
  │
  ├──> Linear Projections (W_Q, W_K, W_V)
  │    │
  │    ├──> Q [B, H, N, d]
  │    ├──> K [B, H, N, d] ──> Mask Padding ──> [B, H, N, d]
  │    └──> V [B, H, N, d] ──> Mask Padding ──> [B, H, N, d]
  │
  ├──> Slice Projections E_act, F_act [k, N]
  │    │
  │    ├──> K_proj = matmul(E_act, K) ──> [B, H, k, d]
  │    └──> V_proj = matmul(F_act, V) ──> [B, H, k, d]
  │
  ├──> Attention Score Computation
  │    S = Q @ K_proj^T / sqrt(d) ───────> [B, H, N, k]
  │    A = Softmax(S, dim=-1) ───────────> [B, H, N, k]
  │    A_drop = Dropout(A) ──────────────> [B, H, N, k]
  │
  ├──> Context Aggregation & Out Projection
  │    Context = A_drop @ V_proj ────────> [B, H, N, d]
  │    Mask Queries ─────────────────────> [B, H, N, d]
  │    Transpose & Flatten ──────────────> [B, N, H * d = D]
  │    Y = Context @ W_O ────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Query/Key/Value Projections | `X: [B, N, D]` | `Q, K, V: [B, N, D]` |
| 2 | Multi-Head Reshape | `Q, K, V: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 3 | Token Masking | `K, V: [B, H, N, d]`, `Mask: [B, 1, N, 1]` | `K_masked, V_masked: [B, H, N, d]` |
| 4 | Projection Matrix Slicing | `E, F: [k, N_max]` | `E_act, F_act: [k, N]` |
| 5 | Key Sequence Compression | `E_act: [k, N]`, `K_masked: [B, H, N, d]` | `K_proj: [B, H, k, d]` |
| 6 | Value Sequence Compression | `F_act: [k, N]`, `V_masked: [B, H, N, d]` | `V_proj: [B, H, k, d]` |
| 7 | Query-Key MatMul | `Q: [B, H, N, d]`, `K_proj^T: [B, H, d, k]` | `S: [B, H, N, k]` |
| 8 | Softmax & Dropout | `S: [B, H, N, k]` | `A: [B, H, N, k]` |
| 9 | Context Matrix Product | `A: [B, H, N, k]`, `V_proj: [B, H, k, d]` | `Context: [B, H, N, d]` |
| 10 | Query Padding Re-masking | `Context: [B, H, N, d]`, `Mask: [B, 1, N, 1]` | `Context_masked: [B, H, N, d]` |
| 11 | Concatenation & Out Projection | `Context_masked: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Projection Linears (\(W_Q, W_K, W_V\))**: \(3 \times 2 B N D^2 = 6 B N D^2\) FLOPs (3 MACs per element).
2. **Sequence Compression MatMuls (\(\mathbf{E}_{\text{act}} \mathbf{K}, \mathbf{F}_{\text{act}} \mathbf{V}\))**: 
   - \(2 \times (2 B H k N d) = 4 B N k D\) FLOPs.
3. **Attention Score MatMul (\(\mathbf{Q} \mathbf{K}_{\text{proj}}^\top\))**:
   - \(2 B H N k d = 2 B N k D\) FLOPs.
4. **Context Weighted Sum (\(\mathbf{A} \mathbf{V}_{\text{proj}}\))**:
   - \(2 B H N k d = 2 B N k D\) FLOPs.
5. **Output Projection (\(W_O\))**: \(2 B N D^2\) FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 8 B N k D
\]

Compared to standard attention's \(8 B N D^2 + 4 B N^2 D\), the sequence-dependent compute transitions from quadratic \(\mathcal{O}(N^2)\) to strictly linear \(\mathcal{O}(N k)\).

### Memory Footprint & Tensor Store
- **Parameters**: \(4 D^2 + 2 k N_{\text{max}}\) floats.
- **Activation Memory**:
  - Q, K, V Tensors: \(3 \times B \cdot H \cdot N \cdot d = 3 B N D\)
  - Projected Key/Value Tensors: \(2 \times B \cdot H \cdot k \cdot d = 2 B k D\)
  - Attention Map: \(B \cdot H \cdot N \cdot k\)
  - Peak Memory: \(\mathcal{O}(B N D + B H N k)\), maintaining a minimal peak compared to \(\mathcal{O}(B H N^2)\).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Fixed Spatial Grid & Dynamic Length Generalization**: The projection parameters \(\mathbf{E}, \mathbf{F} \in \mathbb{R}^{k \times N_{\text{max}}}\) rely on fixed sequence index coordinates. If inputs exceed \(N_{\text{max}}\), fallback zero-padding is triggered (`F.pad`), leading to uninitialized kernel weights and loss of positional coherence.
2. **Global Parameter Sharing & Position Instability**: Linformer uses non-position-aware linear parameters \(\mathbf{E}, \mathbf{F}\). For tasks requiring fine-grained token alignment or causal autoregressive modeling, static global projection compresses sequence histories indiscriminately across time, preventing strictly causal masking.
3. **Low-Rank Collapse**: When \(k\) is selected such that \(k < \text{rank}(\mathbf{A})\), information bottlenecking occurs. Spectral energy distributions in deep layers often demand higher effective ranks, causing information loss for long-tail token dependencies.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
In standard PyTorch, computing \(\mathbf{E}_{\text{act}} \mathbf{K}\) requires materializing intermediate tensors in DRAM. Linformer requires dual contraction steps: sequence projection followed by standard head dot-product.

```
DRAM <──> Read K [B, H, N, d] & E [k, N] ──> Compute K_proj ──> Write DRAM [B, H, k, d]
DRAM <──> Read Q [B, H, N, d] & K_proj ──> Compute Scores ──> Write DRAM [B, H, N, k]
```

### High-Performance Triton Kernel Fusion Strategy
To reach peak hardware efficiency on NVIDIA H100/A100 GPUs:
1. **Fused Sequence Projection & Q-K Reduction Kernel**: Fuse the multiplication \(\mathbf{E} \mathbf{K}\) directly into the FlashAttention tile loop. Load block-tiles of \(\mathbf{K}\) of size `BLOCK_N`, multiply on-chip by sub-blocks of \(\mathbf{E}\), accumulate in SRAM registers for `K_proj`, and immediately compute dot-products with \(\mathbf{Q}\).
2. **Static Tensor Cores Layout**: Allocate layout tiles of `BLOCK_N = 64` and `BLOCK_K = 32`. Keep \(\mathbf{E}_{\text{act}}\) resident in L1 Caches across sequence blocks to maximize SRAM throughput.
3. **Causal Compatibility via Chunking**: Extend the kernel to handle block-diagonal causal projection matrices to enable causal autoregressive inference.