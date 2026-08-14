# EAT: Efficient Attention Transformer Block with Progressive Token Pruning

## 1. Abstract & Theoretical Motivation
Multi-Head Self-Attention (MHSA) and Feed-Forward Networks (FFN) in standard Transformer blocks maintain a fixed sequence length $L$ across all model layers. In deep networks, many tokens correspond to redundant syntax, background noise, or uninformative punctuation. Maintaining these non-essential tokens through deep layers wastes memory and quadratic FLOPs $\mathcal{O}(L^2 d)$ in subsequent attention operations.

**EAT (Efficient Attention Transformer Block)** introduces dynamic length reduction by integrating a **Progressive Token Pruning** sub-network directly into the Transformer layer pipeline. EAT executes standard Multi-Head Self-Attention and Feed-Forward feature synthesis, then evaluates token-level salience scores to extract a top-$k$ informative token subset $\mathcal{I}_{\text{keep}}$. Non-essential tokens are dynamically dropped, reducing the active sequence length from $L$ to $L' = \lfloor \gamma \cdot L \rfloor$ (where $\gamma < 1$).

By progressively shrinking sequence length across Transformer depth ($L_0 > L_1 > \dots > L_M$), EAT compounds computational and memory savings down the network hierarchy, accelerating inference while preserving task accuracy.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Multi-Head Self-Attention Sub-Layer
Given input activations $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) projections yield:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \quad \in \mathbb{R}^{B \times H \times L \times d_h}$$
where $H$ is the head count and $d_h = D / H$.

If key padding mask $\mathbf{M}_{\text{pad}} \in \{0, 1\}^{B \times L}$ is provided, invalid padding key positions are masked:
$$\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} + (1 - \mathbf{M}_{\text{pad}}) \cdot (-10^9)$$
$$\mathbf{A} = \operatorname{Softmax}(\mathbf{S}) \in \mathbb{R}^{B \times H \times L \times L}$$
$$\mathbf{X}_{\text{attn}} = \mathbf{A} \mathbf{V} \mathbf{W}_O \quad \in \mathbb{R}^{B \times L \times D}$$

The first residual addition and Layer Normalization yield intermediate representation $\mathbf{X}^{(1)}$:
$$\mathbf{X}^{(1)} = \operatorname{LayerNorm}\left( \mathbf{X} + \mathbf{X}_{\text{attn}} \right) \in \mathbb{R}^{B \times L \times D}$$

### 2.2 Position-wise Feed-Forward Sub-Layer
Intermediate activations $\mathbf{X}^{(1)}$ pass through a 2-layer FFN with expansion factor 4 and GELU non-linearity:
$$\mathbf{F} = \operatorname{GELU}\left( \mathbf{X}^{(1)} \mathbf{W}_1 + \mathbf{b}_1 \right) \mathbf{W}_2 + \mathbf{b}_2 \quad \in \mathbb{R}^{B \times L \times D}$$
where $\mathbf{W}_1 \in \mathbb{R}^{D \times 4D}$ and $\mathbf{W}_2 \in \mathbb{R}^{4D \times D}$.

The second residual addition and Layer Normalization produce updated features $\mathbf{X}^{(2)}$:
$$\mathbf{X}^{(2)} = \operatorname{LayerNorm}\left( \mathbf{X}^{(1)} + \mathbf{F} \right) \in \mathbb{R}^{B \times L \times D}$$

### 2.3 Progressive Token Pruning Stage
If a `ProgressiveTokenPruning` module is attached, token salience scores $\mathbf{s}_{\text{salience}} \in \mathbb{R}^{B \times L}$ are evaluated over $\mathbf{X}^{(2)}$:
$$\mathbf{s}_{\text{salience}} = \operatorname{ScoreNetwork}\left( \mathbf{X}^{(2)} \right)$$

Given a target retention ratio $\gamma \in (0, 1]$, the number of tokens retained is $L' = \lfloor \gamma \cdot L \rfloor$. The top-$L'$ token indices $\mathcal{I}_{\text{keep}}$ are extracted via Top-$k$:
$$\mathcal{I}_{\text{keep}} = \operatorname{TopK}\left( \mathbf{s}_{\text{salience}}, L' \right) \quad \in \{1, \dots, L\}^{B \times L'}$$

The hidden activation sequence is gathered along the sequence dimension:
$$\mathbf{X}^{\text{pruned}} = \operatorname{Gather}\left( \mathbf{X}^{(2)}, \text{dim}=1, \text{index}=\mathcal{I}_{\text{keep}} \right) \quad \in \mathbb{R}^{B \times L' \times D}$$

If original position indices $\mathbf{T}_{\text{idx}} \in \mathbb{N}^{B \times L}$ are tracked across layers, they are gathered identically:
$$\mathbf{T}_{\text{idx}}^{\text{new}} = \operatorname{Gather}\left( \mathbf{T}_{\text{idx}}, \text{dim}=1, \text{index}=\mathcal{I}_{\text{keep}} \right) \quad \in \mathbb{N}^{B \times L'}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Input Activations | - | `[B, L, D]` | Input sequence representation of length $L$ |
| 2 | Q, K, V Projections | `[B, L, D]` | `[B, H, L, d_h]` | Linear map via $W_Q, W_K, W_V$ & split heads |
| 3 | Scaled Dot Attn | `Q`, `K`, `V`, `Mask`| `[B, L, D]` | Multi-head attention with optional key padding mask |
| 4 | Residual Add & Norm 1 | `[B, L, D]` | `[B, L, D]` | Add input residual & apply LayerNorm |
| 5 | Feed-Forward MLP | `[B, L, D]` | `[B, L, D]` | 2-layer MLP with GELU ($D \to 4D \to D$) |
| 6 | Residual Add & Norm 2 | `[B, L, D]` | `[B, L, D]` | Add intermediate residual & apply LayerNorm |
| 7 | Salience Scoring | `[B, L, D]` | `[B, L]` | Evaluate token importance scores |
| 8 | Top-$L'$ Index Extraction| `[B, L]` | `[B, L']` | Extract top-$L'$ indices $\mathcal{I}_{\text{keep}}$ ($L' \le L$) |
| 9 | Token Gathering | `[B, L, D]`, `Indices`| `[B, L', D]` | Slice tensor along sequence axis to pruned length $L'$ |
| 10| Index Alignment | `[B, L]`, `Indices` | `[B, L']` | Gather original position tracking indices |

```
                       [B, L, D] Input Tensor
                           │
       ┌───────────────────┴───────────────────┐
       │ Residual                              ▼
       │                      Multi-Head Self-Attention
       │                         Key Padding Mask [B, L]
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
               Residual Add & LayerNorm 1
                           │
       ┌───────────────────┴───────────────────┐
       │ Residual                              ▼
       │                      Position-wise FFN Network
       │                         GELU(X W_1 + b_1) W_2
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
               Residual Add & LayerNorm 2 (X^(2))
                           │
       ┌───────────────────┴───────────────────┐
       │ Prune Layer == None                   │ Prune Layer Active
       ▼                                       ▼
Return [B, L, D] Output             Token Salience Scoring S(X^(2))
                                               │
                                               ▼
                                    Top-L' Selection (L' <= L)
                                    Indices I_keep [B, L']
                                               │
                                               ▼
                                    Gather Pruned Activations
                                    X_pruned [B, L', D]
                                               │
                                               ▼
                                    Gather Position Tracker T_idx
                                               │
                                               ▼
                                    Return [B, L', D] Output
```

---

## 4. Analytical Complexity Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ input sequence length, $D$ hidden dimension, $d_{\text{ffn}} = 4D$, and $L'$ pruned sequence length ($L' = \lfloor \gamma L \rfloor$).

1. **Multi-Head Self-Attention (MHA)**:
   - Projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}$): $8 B L D^2$
   - Score Matmul ($\mathbf{Q} \mathbf{K}^\top$) & Context Matmul ($\mathbf{A} \mathbf{V}$): $4 B H L^2 d_h = 4 B L^2 D$
   $$\operatorname{FLOPs}_{\text{mha}} = 8 B L D^2 + 4 B L^2 D$$
2. **Position-wise Feed-Forward Network (FFN)**:
   - Inner projection ($D \to 4D$): $2 B L D (4D) = 8 B L D^2$
   - Outer projection ($4D \to D$): $2 B L (4D) D = 8 B L D^2$
   $$\operatorname{FLOPs}_{\text{ffn}} = 16 B L D^2$$
3. **Layer Normalizations & Residual Additions**:
   $$\operatorname{FLOPs}_{\text{norm}} = 6 B L D$$
4. **Progressive Token Pruning Stage**:
   - Salience scoring projection ($D \to 1$): $2 B L D$
   - Top-$k$ sorting: $B \cdot L \log L'$
   $$\operatorname{FLOPs}_{\text{prune}} = 2 B L D + B L \log L'$$

$$\text{Total Block FLOPs} = 24 B L D^2 + 4 B L^2 D + 8 B L D + B L \log L'$$

Downstream layers receive length $L' = \gamma L$, dropping downstream attention FLOPs from $\mathcal{O}(L^2 D)$ to $\mathcal{O}(\gamma^2 L^2 D)$.

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} \approx 12 B L D^2 + 2 B L^2 D + B L D$$
- **Peak Activation Memory**:
  - Attention Score Tensor: $B \cdot H \cdot L \cdot L \times 2 \text{ bytes}$
  - Intermediate FFN Activation Buffer: $B \cdot L \cdot 4D \times 2 \text{ bytes}$
  - Pruned Output Buffer: $B \cdot L' \cdot D \times 2 \text{ bytes}$.
  - Subsequent layer activation memory drops by factor $L' / L = \gamma$.

### 4.3 Asymptotic Scalability Comparison
- **Standard Transformer Layer**: $\mathcal{O}(L^2 D + L D^2)$ compute, $\mathcal{O}(L^2 H + L D)$ memory.
- **EAT Block with Progressive Pruning**: Compounds length reduction across $M$ layers ($L_0 \to L_1 \to \dots \to L_M$). Total attention compute across $M$ layers drops from $M \cdot L_0^2 D$ to $\sum_{m=0}^{M-1} L_m^2 D \ll M L_0^2 D$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Non-Gatherable Positional Encoding Disconnect
When tokens are physically pruned from $L \to L'$:
```python
x, new_indices = self.prune_layer(x)
```
- **Spatial Position Distortion**: If static sinusoidal or absolute positional embeddings were added at layer 0, gathering activations $x[:, \mathcal{I}_{\text{keep}}, :]$ without updating rotary or relative positional bias causes remaining tokens to appear adjacent, corrupting relative distance metrics in deeper layers.
- **Resolution Loss**: Relative Attention Bias matrices of size $L \times L$ fail to broadcast against pruned shapes $L' \times L'$ without explicit index gathering.

### 5.2 Dynamic Sequence Length Batching Stalls
In batched multi-GPU execution ($B > 1$):
- If `ProgressiveTokenPruning` evaluates top-$k$ independently per sample in a batch, sample 1 might retain $L'_1$ tokens while sample 2 retains $L'_2$ tokens ($L'_1 \neq L'_2$).
- **Batch Asynchrony**: Un-even sequence lengths break standard dense tensor representations `[B, L', D]`, forcing CPU synchronization, dynamic padding, or nested ragged tensors that stall GPU SIMD hardware throughput.

### 5.3 Irreversible Early Pruning Errors (False-Positive Eviction)
Token pruning is a hard, non-backpropagable step ($\operatorname{Gather}(x, \mathcal{I}_{\text{keep}})$).
- **Early-Layer Information Loss**: In early layers (e.g., Layer 1 or 2), token representations are un-refined. If a critical keyword is incorrectly assigned a low salience score, it is permanently dropped.
- **Zero Gradient Recovery**: Evicted tokens receive zero gradients during backpropagation ($\nabla_{\mathbf{X}_{\text{evicted}}} \mathcal{L} = 0$), preventing early layers from correcting misclassified token salience.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

To prevent dynamic sequence lengths from breaking GPU batch execution, EAT requires a fused Triton Token Pruning & Re-Ordering kernel operating on contiguous memory buffers.

### 6.1 Triton Hardware Acceleration Layout

```
+-----------------------------------------------------------------------------------+
|                           TRITON FUSED EAT PIPELINE                               |
+-----------------------------------------------------------------------------------+
|  1. FUSED ATTENTION & LN KERNEL                                                   |
|     - Computes FlashAttention and LN1 + LN2 in SRAM registers.                    |
|     - Writes intermediate output X_2 [B, L, D].                                   |
+-----------------------------------------------------------------------------------+
|  2. TOKEN SALIENCE SCORING & BITONIC SORT KERNEL                                  |
|     - Evaluates score vector S = X_2 @ W_score [B, L].                            |
|     - Executes warp-level Bitonic Top-k sort in register file.                    |
|     - Extracts top-L' indices I_keep [B, L'].                                     |
+-----------------------------------------------------------------------------------+
|  3. CONTIGUOUS MEMORY RE-PACKING KERNEL                                           |
|     - Indirect load X_2[b, I_keep[b, i]] into SRAM.                               |
|     - Writes packed, contiguous output X_pruned [B, L', D] to device memory.        |
|     - Updates ragged offset array for downstream attention layers.                 |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton Token Packing Kernel)

```python
import triton
import triton.language as tl

@triton.jit
def _eat_fused_token_prune_pack_kernel(
    X_ptr,          # Input hidden states [B, L, D]
    W_score_ptr,    # Scoring projection weights [D]
    Out_packed_ptr, # Pruned output buffer [B, L_PRUNED, D]
    Idx_out_ptr,    # Kept indices tracking buffer [B, L_PRUNED]
    stride_xb, stride_xl, stride_xd,
    stride_ob, stride_ol, stride_od,
    stride_ib, stride_il,
    L_ORIG: tl.constexpr,
    L_PRUNED: tl.constexpr,
    D: tl.constexpr
):
    batch_id = tl.program_id(axis=0)
    offs_d = tl.arange(0, D)

    # Base pointers for batch
    x_base = X_ptr + batch_id * stride_xb
    out_base = Out_packed_ptr + batch_id * stride_ob
    idx_base = Idx_out_ptr + batch_id * stride_ib

    # Evaluate Salience Score per token
    # (Simplified: in-register dot product with W_score)
    w_score = tl.load(W_score_ptr + offs_d)

    # Store scores in SRAM buffer for sorting
    # In Triton, we load token representations, dot product with w_score -> s_i
    # Perform Bitonic Top-k selection to extract L_PRUNED indices
    # Here we illustrate the indirect pack phase after top-k indices are selected

    for l_out in range(L_PRUNED):
        # Fetch target source token index from TopK selection
        source_token_idx = tl.load(idx_base + l_out * stride_il)

        # Indirect Load target token [D] from un-pruned sequence
        x_ptrs = x_base + source_token_idx * stride_xl + offs_d * stride_xd
        token_vec = tl.load(x_ptrs)

        # Store contiguous packed token vector into pruned output buffer
        out_ptrs = out_base + l_out * stride_ol + offs_d * stride_od
        tl.store(out_ptrs, token_vec)
```

### 6.3 Hardware Optimization Impact
1. **Dynamic Memory Compaction**: Re-packing pruned tokens into contiguous global memory arrays `[B, L', D]` restores full GPU SIMD vectorization for subsequent Transformer layers.
2. **Compound Downstream Speedup**: Dropping sequence length from $L_0 \to L'$ yields an immediate $\left(\frac{L'}{L_0}\right)^2$ reduction in downstream Multi-Head Attention latency.