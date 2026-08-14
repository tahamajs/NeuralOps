# BAM-UMoE: Unified Mixture-of-Experts Attention Replacement Block

## 1. Abstract & Theoretical Motivation
Standard Multi-Head Self-Attention (MHSA) computes pairwise similarity matrices of size $L \times L$, scaling quadratically with sequence length $L$. In contexts where token-to-token cross-attention provides marginal utility relative to its computational expense, **BAM-UMoE (Block-Attention Replacement via Unified Mixture-of-Experts)** replaces quadratic attention mechanisms entirely. Instead of inter-token matrix multiplication, BAM-UMoE formulates Transformer layer transformations as a sequence of parameter-efficient, token-wise conditional routing computations over specialized expert sub-networks.

The core architecture consists of two stacked **Unified Mixture-of-Experts (UMoE)** blocks with top-$k$ routing: a QKV-MoE that projects input representations into unified feature spaces, followed by an FFN-MoE that handles high-dimensional non-linear feature synthesis. By bypassing pairwise sequence attention, BAM-UMoE achieves linear complexity $\mathcal{O}(L \cdot k \cdot D^2)$ with respect to sequence length, providing an asymptotic speedup and memory footprint reduction. 

This model shifts the computational paradigm from *sequence-level token interaction* to *token-level capacity scaling via sparse expert routing*, enabling extreme inference throughput in compute-constrained environments.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Unified Mixture-of-Experts (`UnifiedMoE`) Routing
Given an input feature tensor $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, the routing gate computes expert logit activations $\mathbf{R}$ across $E$ candidate experts:
$$\mathbf{R} = \mathbf{X} \mathbf{W}_{\text{gate}} \quad \in \mathbb{R}^{B \times L \times E}$$
where $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{D \times E}$.

The top-$k$ expert routing selection identifies the indices $\mathcal{I}_{\text{top-}k}$ and logit values $\mathbf{S}_{\text{top-}k}$ for each token:
$$\mathcal{I}_{\text{top-}k}, \mathbf{S}_{\text{top-}k} = \operatorname{TopK}\left(\mathbf{R}, k\right)$$
where $\mathcal{I}_{\text{top-}k} \in \{1, \dots, E\}^{B \times L \times k}$ and $\mathbf{S}_{\text{top-}k} \in \mathbb{R}^{B \times L \times k}$. 

Softmax normalization is applied over the top-$k$ logits to compute gating probabilities $\mathbf{P}$:
$$\mathbf{P}_{b, \ell, j} = \frac{\exp(\mathbf{S}_{\text{top-}k, b, \ell, j})}{\sum_{m=1}^k \exp(\mathbf{S}_{\text{top-}k, b, \ell, m})}, \quad j \in \{1, \dots, k\}$$

### 2.2 Expert Sub-Network Computation
Each expert sub-network $e \in \{1, \dots, E\}$ is parameterized as a two-layer Feed-Forward Network with GELU activation:
$$\operatorname{Expert}_e(\mathbf{X}) = \operatorname{GELU}\left(\mathbf{X} \mathbf{W}_{1, e} + \mathbf{b}_{1, e}\right) \mathbf{W}_{2, e} + \mathbf{b}_{2, e}$$
where $\mathbf{W}_{1, e} \in \mathbb{R}^{D \times (2 \cdot d_{\text{out}})}$ and $\mathbf{W}_{2, e} \in \mathbb{R}^{(2 \cdot d_{\text{out}}) \times d_{\text{out}}}$.

The expert stack evaluates all $E$ expert transformations in parallel:
$$\mathbf{E}_{\text{out}} = \operatorname{Stack}\left( \left[ \operatorname{Expert}_1(\mathbf{X}), \dots, \operatorname{Expert}_E(\mathbf{X}) \right], \text{dim}=2 \right) \in \mathbb{R}^{B \times L \times E \times d_{\text{out}}}$$

The selected expert outputs are gathered according to $\mathcal{I}_{\text{top-}k}$ and aggregated via weighted summation:
$$\mathbf{Y}_{\text{topk}} = \operatorname{Gather}\left(\mathbf{E}_{\text{out}}, \mathcal{I}_{\text{top-}k}\right) \in \mathbb{R}^{B \times L \times k \times d_{\text{out}}}$$
$$\mathbf{Y} = \sum_{j=1}^k \mathbf{P}_{:, :, j} \odot \mathbf{Y}_{\text{topk}, :, :, j, :} \quad \in \mathbb{R}^{B \times L \times d_{\text{out}}}$$

### 2.3 Layer Pipeline: QKV-MoE & FFN-MoE Integration
The overall `BAMUMoEBlock` executes the following sequence:

1. **QKV-MoE Projection**:
   $$\mathbf{QKV} = \operatorname{UnifiedMoE}_{\text{qkv}}(\mathbf{X}) \in \mathbb{R}^{B \times L \times (3 \cdot D)}$$
   The projected output is split into Query, Key, and Value representations:
   $$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \operatorname{Chunk}\left(\mathbf{QKV}, 3, \text{dim}=-1\right), \quad \mathbf{V} \in \mathbb{R}^{B \times L \times D}$$

2. **Output Linear Projection & Residual Norm 1**:
   $$\mathbf{H}_{\text{proj}} = \mathbf{V} \mathbf{W}_{\text{out}} \in \mathbb{R}^{B \times L \times D}$$
   $$\mathbf{H}_1 = \operatorname{LayerNorm}\left( \mathbf{X} + \operatorname{Dropout}(\mathbf{H}_{\text{proj}}) \right)$$

3. **FFN-MoE Projection & Residual Norm 2**:
   $$\mathbf{H}_{\text{ffn}} = \operatorname{UnifiedMoE}_{\text{ffn}}(\mathbf{H}_1) \in \mathbb{R}^{B \times L \times D}$$
   $$\mathbf{H}_2 = \operatorname{LayerNorm}\left( \mathbf{H}_1 + \operatorname{Dropout}(\mathbf{H}_{\text{ffn}}) \right)$$

4. **Attention Masking (if provided)**:
   $$\mathbf{Y}_{\text{final}} = \mathbf{H}_2 \odot \mathbf{M}_{\text{attn}}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Input Hidden States | - | `[B, L, D]` | Input activation sequence |
| 2 | QKV Router Logits | `[B, L, D]` | `[B, L, E]` | Linear map via $W_{\text{gate, QKV}}$ |
| 3 | QKV Top-$k$ Selection | `[B, L, E]` | `[B, L, k]` | Indices & values for top-$k$ experts |
| 4 | QKV Softmax Gating | `[B, L, k]` | `[B, L, k]` | Softmax over selected top-$k$ expert logits |
| 5 | Parallel Expert Stack | `[B, L, D]` | `[B, L, E, 3D]` | Stacked evaluation of $E$ experts ($d_{\text{out}} = 3D$) |
| 6 | Gather & Weighted Sum | `Stack`, `TopK` | `[B, L, 3D]` | Gather selected experts and sum weighted by probabilities |
| 7 | Chunking | `[B, L, 3D]` | 3x `[B, L, D]` | Split output into Q, K, V matrices |
| 8 | Output Linear Map | `[B, L, D]` | `[B, L, D]` | Linear projection on Value component ($W_{\text{out}}$) |
| 9 | Residual Add & Norm 1 | `[B, L, D]` | `[B, L, D]` | Add input residual & apply LayerNorm |
| 10| FFN Router Logits | `[B, L, D]` | `[B, L, E]` | Linear map via $W_{\text{gate, FFN}}$ |
| 11| Parallel FFN Experts | `[B, L, D]` | `[B, L, E, D]` | Stacked evaluation of $E$ FFN experts ($d_{\text{out}} = D$) |
| 12| FFN Gather & Sum | `Stack`, `TopK` | `[B, L, D]` | Gather selected FFN experts and sum weighted |
| 13| Residual Add & Norm 2 | `[B, L, D]` | `[B, L, D]` | Add intermediate residual & apply LayerNorm |

```
                     [B, L, D] Input Tensor
                         │
        ┌────────────────┴────────────────┐
        │ Residual                        ▼
        │                       QKV-MoE Router Gate
        │                      Logits [B, L, E]
        │                                 │
        │                                 ▼
        │                       Top-k Expert Indices
        │                        & Softmax Probabilities [B, L, k]
        │                                 │
        │                                 ▼
        │                       Evaluate E Experts in Parallel
        │                      Experts Output Stack [B, L, E, 3D]
        │                                 │
        │                                 ▼
        │                       Gather & Weighted Gating Sum
        │                       QKV Output [B, L, 3D]
        │                                 │
        │                                 ▼
        │                       Chunk into Q, K, V [B, L, D]
        │                                 │
        │                                 ▼
        │                       Linear Output Projection (V @ W_out)
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             Residual Addition & LayerNorm 1
                         │
        ┌────────────────┴────────────────┐
        │ Residual                        ▼
        │                       FFN-MoE Router Gate
        │                      Logits [B, L, E]
        │                                 │
        │                                 ▼
        │                       Evaluate E FFN Experts
        │                      Stack [B, L, E, D] & Top-k Gather
        │                                 │
        │                                 ▼
        │                       Weighted Gating Sum [B, L, D]
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             Residual Addition & LayerNorm 2
                         │
                         ▼
                     [B, L, D] Output
```

---

## 4. Analytical Complexity Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ sequence length, $D$ hidden dimension, $E$ total experts, and $k$ top active experts.

1. **QKV-MoE Router Logits**:
   $$\operatorname{FLOPs}_{\text{qkv\_router}} = 2 B L D E$$
2. **QKV Expert Computations**:
   For each expert $e$, inner projection maps $D \to 6D$ ($2 d_{\text{out}} = 6D$) requiring $2 \cdot 6 D^2$ FLOPs; outer projection maps $6D \to 3D$ requiring $2 \cdot 6D \cdot 3D = 36 D^2$ FLOPs.
   Total per expert $= 12 D^2 + 36 D^2 = 48 D^2$ FLOPs.
   Evaluating all $E$ experts across $B \times L$ tokens:
   $$\operatorname{FLOPs}_{\text{qkv\_experts}} = 48 B L E D^2$$
3. **QKV Gated Weighted Sum**:
   $$\operatorname{FLOPs}_{\text{qkv\_sum}} = 2 B L k (3D) = 6 B L k D$$
4. **Linear Output Projection ($\mathbf{W}_{\text{out}}$)**:
   $$\operatorname{FLOPs}_{\text{out\_proj}} = 2 B L D^2$$
5. **FFN-MoE Router Logits**:
   $$\operatorname{FLOPs}_{\text{ffn\_router}} = 2 B L D E$$
6. **FFN Expert Computations**:
   For each FFN expert, inner projection maps $D \to 2D$ ($4 D^2$ FLOPs); outer projection maps $2D \to D$ ($4 D^2$ FLOPs). Total per expert $= 8 D^2$ FLOPs.
   Evaluating all $E$ experts across $B \times L$ tokens:
   $$\operatorname{FLOPs}_{\text{ffn\_experts}} = 8 B L E D^2$$
7. **FFN Gated Weighted Sum**:
   $$\operatorname{FLOPs}_{\text{ffn\_sum}} = 2 B L k D$$

$$\text{Total FLOPs} = 56 B L E D^2 + 2 B L D^2 + 4 B L D E + 8 B L k D$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} \approx 28 B L E D^2 + B L D^2 + 2 B L D E$$
- **Peak Activation Memory**:
  - Router Logits: $2 \times B \cdot L \cdot E \times 2 \text{ bytes}$
  - QKV Expert Stack Buffer: $B \cdot L \cdot E \cdot 3D \times 2 \text{ bytes}$ (In un-grouped naive implementation, requires huge memory)
  - FFN Expert Stack Buffer: $B \cdot L \cdot E \cdot D \times 2 \text{ bytes}$
  - Intermediate activations scale linearly with $L$: $\mathcal{O}(B L E D)$ (no $L^2$ dependence).

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Attention**: $\mathcal{O}(L^2 D)$ compute, $\mathcal{O}(L^2 H)$ memory.
- **BAM-UMoE Block**: $\mathcal{O}(L E D^2)$ compute, $\mathcal{O}(L E D)$ memory.
- BAM-UMoE completely eliminates sequence length quadratic dependency, allowing linear scaling to arbitrary $L$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Naive Torch Stack Memory Allocation Explosion
In the Python implementation:
```python
experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)
```
Every token evaluates **all $E$ experts** regardless of top-$k$ routing decisions, storing the intermediate outputs of all $E$ experts in global VRAM before gathering the top-$k$.
- **VRAM Waste**: Expands activation memory by a factor of $E / k$ (e.g., $8 / 2 = 4 \times$ unnecessary allocation).
- **FLOPs Waste**: Computes $E$ forward passes instead of $k$ forward passes, burning $E / k \times$ more compute than theoretical conditional computation.

### 5.2 Complete Loss of Spatial Token Interaction
BAM-UMoE processes tokens strictly independently along sequence axis $L$.
- **Receptive Field Collapse**: Receptive field of any token is strictly $1$. The model has zero cross-token context aggregation mechanism.
- **Positional Degeneracy**: If positional embeddings are omitted or weakly integrated, sentence-level permutation invariance occurs. The architecture cannot perform context-dependent semantic binding.

### 5.3 Expert Routing Collapse & Load Imbalance
In router gate $\operatorname{Softmax}\left(\operatorname{TopK}(\mathbf{X} \mathbf{W}_{\text{gate}})\right)$, without auxiliary load balancing loss (such as Switch Transformer or DeepSeek MoE auxiliary balance loss):
- A small subset of experts (e.g., 1 or 2 experts) quickly acquire large norm weights $\mathbf{W}_{\text{gate}}$.
- The router constantly selects the same over-utilized experts, causing remaining experts to become dead nodes ($\nabla_{\mathbf{W}_e} \mathcal{L} = 0$).

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

To convert BAM-UMoE into true conditional computation, execution must avoid evaluating all $E$ experts and eliminate intermediate stack buffers.

### 6.1 Triton Grouped GEMM & Scatter-Gather Blueprint
Instead of evaluating `[B, L, D]` across all $E$ MLPs, tokens are dynamically binned into expert buckets using hardware scatter indices, enabling a single Fused Grouped GEMM.

```
+-----------------------------------------------------------------------------------+
|                        TRITON FUSED UMOE ROUTED GEMM                              |
+-----------------------------------------------------------------------------------+
|  1. ROUTER & HISTOGRAM KERNEL                                                     |
|     - Compute Logits R = X @ W_gate [B*L, E].                                     |
|     - Extract Top-k indices and probabilities.                                    |
|     - Compute expert token counts (Histogram) & scatter offset map.               |
|                                                                                   |
|  2. TOKEN PERMUTATION / REORDERING KERNEL                                         |
|     - Permute input X into grouped contiguous buffer X_grouped [k*B*L, D]         |
|       according to assigned expert IDs.                                           |
|                                                                                   |
|  3. FUSED GROUPED GEMM KERNEL                                                     |
|     - Executes E GEMM problems concurrently using atomic work dispatch.           |
|     - For expert e: Y_grouped[e] = GELU(X_grouped[e] @ W1[e]) @ W2[e].            |
|                                                                                   |
|  4. UN-PERMUTE & WEIGHTED ACCUMULATION KERNEL                                     |
|     - Scatter-add weighted expert outputs Y_grouped * P back to Y_out [B, L, D].   |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton JIT Grouped Scatter Execution)

```python
import triton
import triton.language as tl

@triton.jit
def _umoe_scatter_kernel(
    X_ptr,          # Input activations [Total_Tokens, D]
    Grouped_X_ptr,  # Re-ordered activations [Total_Tokens * k, D]
    TopK_Idx_ptr,   # Expert assignments [Total_Tokens, k]
    Offsets_ptr,    # Expert bucket offsets [E]
    stride_x_t, stride_x_d,
    stride_gx_t, stride_gx_d,
    stride_idx_t, stride_idx_k,
    D: tl.constexpr,
    K: tl.constexpr
):
    token_id = tl.program_id(axis=0)
    offs_d = tl.arange(0, D)

    # Load token representation
    x_ptrs = X_ptr + token_id * stride_x_t + offs_d * stride_x_d
    x_val = tl.load(x_ptrs)

    # Loop over top-k assignments for this token
    for k_idx in range(K):
        expert_id = tl.load(TopK_Idx_ptr + token_id * stride_idx_t + k_idx * stride_idx_k)
        
        # Atomic fetch and increment slot in target expert bucket
        slot_idx = tl.atomic_add(Offsets_ptr + expert_id, 1)

        # Write token into grouped contiguous memory
        gx_ptrs = Grouped_X_ptr + slot_idx * stride_gx_t + offs_d * stride_gx_d
        tl.store(gx_ptrs, x_val)
```

### 6.3 Hardware Memory Optimization Summary
1. **True Conditional Computation**: Compute is strictly bounded by $\mathcal{O}(k \cdot B \cdot L \cdot D^2)$ instead of $\mathcal{O}(E \cdot B \cdot L \cdot D^2)$.
2. **Zero Stack Allocation**: Eliminates `torch.stack` intermediate tensors, reducing peak memory usage from $\mathcal{O}(B L E D)$ to $\mathcal{O}(B L k D)$.
3. **Tensor Core Utilization**: Re-ordering tokens into contiguous expert blocks enables dense matrix multiplication (`tl.dot`) targeting Tensor Cores at full hardware utilization.