# Progressive Token Pruning: Learnable Saliency Scoring and Monotonic Sequence Compaction

## 1. Abstract & Theoretical Motivation
Dense Multi-Head Self-Attention (MHSA) in Transformer architectures scales quadratically $\mathcal{O}(L^2 d)$ with sequence length $L$. In deep models, sequence representations contain substantial spatial redundancy, as background tokens, stop-words, and repetitive punctuation contribute minimal utility to final task predictions. Processing these non-essential tokens across all layers consumes massive memory bandwidth and FLOPs.

**Progressive Token Pruning (`ProgressiveTokenPruning`)** provides an adaptive, data-driven sequence compaction module that dynamically prunes uninformative token vectors along sequence length $L$. The layer applies a lightweight, learnable scoring projection $W_s \in \mathbb{R}^{D \times 1}$ to map token embeddings to continuous saliency scores $s \in (0, 1)$. Given a retention ratio $\rho \in (0, 1]$, the top-$k$ ($k = \lfloor \rho \cdot L \rfloor$) most salient tokens are extracted. Crucially, the extracted token indices are sorted monotonically prior to gathering, preserving original relative sequence ordering and relative positional embedding alignment.

By reducing sequence length from $L$ to $k \le L$ with minimal parameter overhead ($\mathcal{O}(D)$ parameters), Progressive Token Pruning compounds exponential compute and memory savings down Transformer hierarchies.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Learnable Token Saliency Scoring
Given hidden sequence representation $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, a linear scoring layer parameterizes token-level salience weights $\mathbf{W}_s \in \mathbb{R}^{D \times 1}$ and bias $b_s \in \mathbb{R}$:
$$\mathbf{z} = \mathbf{X} \mathbf{W}_s + b_s \quad \in \mathbb{R}^{B \times L \times 1}$$

Sigmoid activation maps logit activations to normalized saliency probabilities $\mathbf{s} \in (0, 1)^{B \times L}$:
$$\mathbf{s}_{b, i} = \sigma\left( \mathbf{z}_{b, i, 1} \right) = \frac{1}{1 + \exp\left( -\mathbf{z}_{b, i, 1} \right)}, \quad \forall i \in \{1, \dots, L\}$$

### 2.2 Top-$k$ Selection & Monotonic Order Preservation
Given retention ratio $\rho \in (0, 1]$, the number of tokens retained per sequence is:
$$k = \max\left( 1, \lfloor \rho \cdot L \rfloor \right)$$

1. **Top-$k$ Extraction**:
   For each batch sequence $b \in \{1, \dots, B\}$, the indices $\mathcal{U}_b$ corresponding to the $k$ highest saliency scores are extracted:
   $$\mathcal{U}_b = \operatorname{TopK}\left( \mathbf{s}_{b, :}, k \right) \quad \in \{1, \dots, L\}^k$$

2. **Monotonic Index Sorting**:
   Because top-$k$ selection re-orders index entries by descending score magnitude, direct gathering would corrupt sequential relative positions. Monotonic ascending sorting resolves this:
   $$\mathcal{I}_b = \operatorname{Sort}_{\text{ascending}}\left( \mathcal{U}_b \right) \quad \in \{1, \dots, L\}^k$$
   such that $\mathcal{I}_{b, 1} < \mathcal{I}_{b, 2} < \dots < \mathcal{I}_{b, k}$.

### 2.3 Index Gathering & Sequence Compaction
The hidden activation tensor $\mathbf{X}$ is gathered along the sequence axis using the sorted index matrix $\mathcal{I} \in \mathbb{N}^{B \times k}$:
$$\mathbf{X}_{\text{pruned}, b, j, :} = \mathbf{X}_{b, \mathcal{I}_{b, j}, :} \quad \forall j \in \{1, \dots, k\}$$
$$\mathbf{X}_{\text{pruned}} = \operatorname{Gather}\left( \mathbf{X}, \text{dim}=1, \text{index}=\mathcal{I} \right) \quad \in \mathbb{R}^{B \times k \times D}$$

The module outputs the compacted tensor $\mathbf{X}_{\text{pruned}} \in \mathbb{R}^{B \times k \times D}$ and the monotonically sorted index map $\mathcal{I} \in \mathbb{N}^{B \times k}$.

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Input Hidden States | - | `[B, L, D]` | Input activation sequence of length $L$ |
| 2 | Saliency Score Logits| `[B, L, D]` | `[B, L, 1]` | Linear map via $W_s \in \mathbb{R}^{D \times 1}$ |
| 3 | Sigmoid Normalization | `[B, L, 1]` | `[B, L]` | Squeeze last dim & apply Sigmoid activation |
| 4 | Top-$k$ Extraction | `[B, L]` | `[B, k]` | Extract top-$k$ indices ($k = \lfloor \rho L \rfloor$) |
| 5 | Monotonic Sorting | `[B, k]` | `[B, k]` | Sort indices ascending along sequence axis |
| 6 | Index Expansion | `[B, k]` | `[B, k, D]` | `unsqueeze(-1).expand(-1, -1, D)` |
| 7 | Activation Gather | `[B, L, D]`, `Indices`| `[B, k, D]` | Compact tensor along sequence dimension |

```
                       [B, L, D] Input Tensor
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     Saliency Score Network      Input Activations X
     Linear(D -> 1)              [B, L, D]
             │                           │
             ▼                           │
     Sigmoid Normalization               │
     Scores s [B, L]                     │
             │                           │
             ▼                           │
     Top-k Selection (k = rho * L)       │
     Indices U [B, k]                    │
             │                           │
             ▼                           │
     Monotonic Ascending Sort            │
     Sorted Indices I [B, k]             │
             │                           │
             └─────────────┬─────────────┘
                           ▼
              Gather Compacted Activations
              X_pruned [B, k, D]
                           │
                           ▼
          Tuple: (X_pruned, Sorted_Indices)
```

---

## 4. Analytical Complexity Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ input sequence length, $D$ hidden dimension, $\rho$ retention ratio, and $k = \lfloor \rho \cdot L \rfloor$.

1. **Saliency Score Projection**:
   $$\operatorname{FLOPs}_{\text{score}} = 2 B L D$$
2. **Top-$k$ Extraction & Sorting**:
   - Top-$k$ heap search: $B \cdot L \log k$ FLOPs.
   - Bitonic monotonic sort: $B \cdot k \log k$ FLOPs.
   $$\operatorname{FLOPs}_{\text{topk\_sort}} = B L \log k + B k \log k$$
3. **Tensor Gathering Operation**:
   $$\operatorname{FLOPs}_{\text{gather}} = B k D \quad (\text{Memory copy operation})$$

$$\text{Total FLOPs} = 2 B L D + B k D + B L \log k + B k \log k$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} \approx B L D$$
- **Peak Activation Memory**:
  - Score Tensor: $B \cdot L \times 4 \text{ bytes}$
  - Kept Index Tensor: $B \cdot k \times 8 \text{ bytes}$
  - Compacted Output Activation Buffer: $B \cdot k \cdot D \times 2 \text{ bytes}$ (FP16/BF16).
  - Memory Footprint drops from $2 B L D$ to $2 B k D$ bytes.

### 4.3 Asymptotic Scalability Comparison
- **Standard Sequence Layer**: $\mathcal{O}(L D)$ compute, $\mathcal{O}(L D)$ memory footprint.
- **Progressive Token Pruning**: $\mathcal{O}(L D + k D)$ compute, $\mathcal{O}(k D)$ downstream memory footprint.
- Downstream layer attention FLOPs drop by $\left(\frac{k}{L}\right)^2 = \rho^2 \times$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Straight-Through Non-Differentiability & Gradient Disconnect
The index extraction $\mathcal{U} = \operatorname{TopK}(\mathbf{s}, k)$ and sorting operations are discrete step functions.
- **Zero Gradient Derivative**: $\frac{\partial \mathcal{I}}{\partial \mathbf{s}} = 0$ everywhere.
- **Pruning Scorer Disconnect**: Gradients flowing from task loss $\mathcal{L}$ pass through gathered features $\mathbf{X}_{\text{pruned}}$, but **do not backpropagate through $\operatorname{TopK}$ to optimize the scoring weights $\mathbf{W}_s$**.
- Without soft relaxation (e.g., Gumbel-Softmax or straight-through estimator scaling $\mathbf{X} \cdot \mathbf{s}$), scoring weights $\mathbf{W}_s$ remain un-optimized or rely purely on heuristic initialization.

### 5.2 Dynamic Length Ragged Tensor Stalls in Multi-GPU Training
If batch size $B > 1$ and token pruning evaluates dynamic scores:
- In variable sequence contexts, sorting indices $\mathcal{I}_b$ for sample 1 might pick different semantic boundaries than sample 2.
- If $k$ varies per sample ($k_1 \neq k_2$), rectangular tensor format `[B, k, D]` collapses, producing ragged arrays that force costly CPU host fallback and re-padding overhead.

### 5.3 Eviction Cascades & Premature Feature Destruction
If $\rho = 0.5$ across 6 consecutive pruning layers:
- Layer 1: $L \to 0.5 L$
- Layer 2: $0.5 L \to 0.25 L$
- Layer 6: $L \to 0.015 L$ ($1.5\%$ of tokens remaining!).
- **Severe Context Starvation**: Exponential compounding reduces sequence length to a tiny fragment, destroying essential multi-token syntactic structures before deep layers can process them.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

To achieve full hardware acceleration and prevent dynamic memory fragmentation, Progressive Token Pruning requires a custom Triton GPU kernel executing fused scoring, warp bitonic top-$k$ sorting, and contiguous SRAM token gathering.

### 6.1 Triton Hardware Acceleration Pipeline Layout

```
+-----------------------------------------------------------------------------------+
|                        TRITON FUSED PTP PIPELINE                                  |
+-----------------------------------------------------------------------------------+
|  1. FUSED SALIENCY SCORE & BITONIC TOP-K KERNEL                                   |
|     - Loads X [B, L, D] into SRAM.                                                |
|     - Computes s_i = sigmoid(X @ W_s + b_s) in register files.                    |
|     - Executes Bitonic Top-k sort across L elements in warp registers.            |
|     - Sorts top-k indices monotonically ascending: I_sorted = sort(topk_idx).      |
|                                                                                   |
|  2. CONTIGUOUS SRAM GATHER & RE-PACKING KERNEL                                    |
|     - Indirect reads X[b, I_sorted[b, j], :] using hardware pointers.             |
|     - Writes packed, aligned tensor X_pruned [B, k, D] directly to device memory.   |
|     - Eliminates intermediate PyTorch torch.gather memory allocations.            |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton Fused Prune & Pack Kernel)

```python
import triton
import triton.language as tl

@triton.jit
def _ptp_fused_score_sort_pack_kernel(
    X_ptr, W_score_ptr, Out_ptr, Indices_ptr,
    stride_xb, stride_xl, stride_xd,
    stride_ob, stride_ok, stride_od,
    stride_ib, stride_ik,
    L_ORIG: tl.constexpr,
    K_KEEP: tl.constexpr,
    D: tl.constexpr
):
    batch_id = tl.program_id(axis=0)
    offs_d = tl.arange(0, D)

    # Base pointers for batch
    x_batch = X_ptr + batch_id * stride_xb
    out_batch = Out_ptr + batch_id * stride_ob
    idx_batch = Indices_ptr + batch_id * stride_ib

    # Load scoring weight vector into SRAM
    w_score = tl.load(W_score_ptr + offs_d)

    # Note: In Triton, we compute saliency scores s_i for i in 0..L_ORIG,
    # perform bitonic top-k sort to extract K_KEEP indices,
    # and sort indices monotonically. Below illustrates the indirect gather step:

    for k_idx in range(K_KEEP):
        # Fetch monotonically sorted token index
        source_idx = tl.load(idx_batch + k_idx * stride_ik)

        # Indirect load token representation [D]
        x_ptrs = x_batch + source_idx * stride_xl + offs_d * stride_xd
        token_vec = tl.load(x_ptrs)

        # Store into contiguous packed output buffer
        out_ptrs = out_base + k_idx * stride_ok + offs_d * stride_od
        tl.store(out_ptrs, token_vec)
```

### 6.3 Hardware Acceleration Summary
1. **Fused In-Register Sorting**: Bitonic sorting in warp registers eliminates global DRAM reads and writes for intermediate score matrices.
2. **SIMD Alignment Preservation**: Packing gathered vectors into contiguous FP16 memory blocks `[B, k, D]` maximizes memory bus utilization for all downstream Transformer layers.