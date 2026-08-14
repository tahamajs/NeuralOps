# Weighted Grouped-Query Attention (WGQA): Learnable Per-Channel Key-Value Importance Scaling

## 1. Abstract & Theoretical Motivation

Grouped-Query Attention (GQA) reduces memory bandwidth pressure during Transformer decoding by sharing a smaller set of Key-Value (KV) heads across multiple Query heads. However, standard GQA imposes a uniform structural prior across all groups, assuming that every KV head contributes equally to its assigned query heads. Weighted Grouped-Query Attention (WGQA) addresses this limitation by introducing learnable per-channel importance weight vectors $w_K, w_V \in \mathbb{R}^{H_{KV}}$ applied directly to key and value representations prior to group expansion.

By equipping each KV head with trainable scalar importance multipliers, WGQA enables end-to-end differentiable head selection, salient subspace amplification, and redundant head suppression without sacrificing the memory savings of GQA. This parametric enhancement introduces minimal memory footprint overhead ($2 \times H_{KV}$ parameters) while restoring representation capacity lost during KV head reduction.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ denote the input hidden representation, where $B$ is batch size, $L$ is sequence length, and $D$ is model hidden dimension. Let $H_Q$ be the number of query heads and $H_{KV}$ be the number of key-value heads ($H_{KV} \le H_Q$), with head dimension $d_k = D / H_Q$. The repetition factor is defined as $\gamma = H_Q / H_{KV}$.

### Linear Subspace Projections:
\[
Q = X W_Q \in \mathbb{R}^{B \times L \times (H_Q \cdot d_k)}, \quad W_Q \in \mathbb{R}^{D \times (H_Q \cdot d_k)}
\]
\[
K = X W_K \in \mathbb{R}^{B \times L \times (H_{KV} \cdot d_k)}, \quad W_K \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\]
\[
V = X W_V \in \mathbb{R}^{B \times L \times (H_{KV} \cdot d_k)}, \quad W_V \in \mathbb{R}^{D \times (H_{KV} \cdot d_k)}
\]

Reshaping into head-separated tensor layouts yields:

\[
Q \in \mathbb{R}^{B \times L \times H_Q \times d_k}, \quad K \in \mathbb{R}^{B \times L \times H_{KV} \times d_k}, \quad V \in \mathbb{R}^{B \times L \times H_{KV} \times d_k}
\]

### Learnable Head Importance Scaling:
Let $w_K \in \mathbb{R}^{H_{KV} \times 1}$ and $w_V \in \mathbb{R}^{H_{KV} \times 1}$ be learnable parameter vectors initialized to $\mathbf{1}$. The per-head modulation is defined as:

\[
\widetilde{K}_{b, l, h, d} = K_{b, l, h, d} \cdot (w_K)_{h, 1}
\]
\[
\widetilde{V}_{b, l, h, d} = V_{b, l, h, d} \cdot (w_V)_{h, 1}
\]

for $h \in \{0, 1, \dots, H_{KV}-1\}$.

### Repeat-Interleave Subspace Expansion:
The modulated key-value representations are expanded along the head dimension by repeating each KV head $\gamma$ times:

\[
K_{\text{exp}} = \text{RepeatInterleave}(\widetilde{K}, \text{repeats}=\gamma, \text{dim}=2) \in \mathbb{R}^{B \times L \times H_Q \times d_k}
\]
\[
V_{\text{exp}} = \text{RepeatInterleave}(\widetilde{V}, \text{repeats}=\gamma, \text{dim}=2) \in \mathbb{R}^{B \times L \times H_Q \times d_k}
\]

Transposing to head-first layouts gives $Q, K_{\text{exp}}, V_{\text{exp}} \in \mathbb{R}^{B \times H_Q \times L \times d_k}$.

### Scaled Dot-Product Attention & Output Aggregation:
The unnormalized attention scores $S \in \mathbb{R}^{B \times H_Q \times L \times L}$ are calculated as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot (K_{\text{exp}})_{b, h, j, m}
\]

With additive attention mask $M \in \mathbb{R}^{B \times 1 \times L \times L}$:

\[
\widetilde{S}_{b, h, i, j} = S_{b, h, i, j} + M_{b, 1, i, j}
\]

Applying row-wise Softmax and dropout:

\[
A_{b, h, i, j} = \text{Softmax}_j (\widetilde{S}_{b, h, i, j}) = \frac{\exp(\widetilde{S}_{b, h, i, j})}{\sum_{l=1}^L \exp(\widetilde{S}_{b, h, i, l})}
\]
\[
\widehat{A} = \text{Dropout}(A, p_{\text{drop}})
\]

Context values are aggregated and projected:

\[
O_{b, h, i, m} = \sum_{j=1}^L \widehat{A}_{b, h, i, j} \cdot (V_{\text{exp}})_{b, h, j, m}
\]
\[
Y = \text{Concat}(O_1, O_2, \dots, O_{H_Q}) \cdot W_O \in \mathbb{R}^{B \times L \times D}
\]

where $W_O \in \mathbb{R}^{(H_Q \cdot d_k) \times D}$.

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, H_Q * d_k) [W_q] ──> Q: [B, L, H_Q, d_k]
  ├──> Linear(D, H_KV * d_k) [W_k] ──> K: [B, L, H_KV, d_k]
  └──> Linear(D, H_KV * d_k) [W_v] ──> V: [B, L, H_KV, d_k]
                                │
                                ├──> Multiply w_K [H_KV, 1] ──> K_scaled: [B, L, H_KV, d_k]
                                ├──> Multiply w_V [H_KV, 1] ──> V_scaled: [B, L, H_KV, d_k]
                                │
                                ├──> repeat_interleave(γ, dim=2) ──> K_exp: [B, L, H_Q, d_k]
                                ├──> repeat_interleave(γ, dim=2) ──> V_exp: [B, L, H_Q, d_k]
                                │
                                ├──> transpose(1, 2) ──> Q, K_exp, V_exp: [B, H_Q, L, d_k]
                                │
                                ├──> MatMul: Q @ K_exp.transpose(-2, -1) / sqrt(d_k) ──> [B, H_Q, L, L]
                                ├──> Add attention_mask ──> [B, H_Q, L, L]
                                ├──> Softmax(dim=-1) & Dropout ──> attn: [B, H_Q, L, L]
                                │
                                └──> MatMul: attn @ V_exp ──> context: [B, H_Q, L, d_k]
                                        │
                                        └──> transpose(1, 2).view([B, L, H_Q * d_k])
                                        └──> Linear(H_Q * d_k, D) [W_o] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Computational Overhead (FLOPs & MACs)

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H_Q$ query heads, $H_{KV}$ key-value heads, $\gamma = H_Q / H_{KV}$, and $d_k = D / H_Q$.

1. **Linear Projections**:
   - $W_Q$: $2 B L D^2$ FLOPs ($B L D^2$ MACs).
   - $W_K, W_V$: $2 \times 2 B L D \cdot (H_{KV} d_k) = \frac{4 B L D^2}{\gamma}$ FLOPs ($\frac{2 B L D^2}{\gamma}$ MACs).

2. **Per-Head Scaling ($w_K, w_V$)**:
   - Broadcast scalar multiplication across batch and sequence dimensions:
   - $2 \times B \cdot L \cdot H_{KV} \cdot d_k = \frac{2 B L D}{\gamma}$ FLOPs. (Negligible relative to GEMM).

3. **Core Attention Computation ($Q K^\top$ and $A V$)**:
   - $Q K^\top$: $2 B L^2 D$ FLOPs ($B L^2 D$ MACs).
   - Softmax: $\approx 5 B H_Q L^2$ FLOPs.
   - $A V$: $2 B L^2 D$ FLOPs ($B L^2 D$ MACs).

4. **Output Projection ($W_O$)**:
   - $2 B L D^2$ FLOPs ($B L D^2$ MACs).

### Total Complexity:
- **Total MACs**: $2 B L D^2 \left(1 + \frac{1}{\gamma}\right) + 2 B L^2 D$
- **Total FLOPs**: $4 B L D^2 \left(1 + \frac{1}{\gamma}\right) + \frac{2 B L D}{\gamma} + 4 B L D^2 + 4 B L^2 D$
- **Additional Parametric Footprint**: $2 \times H_{KV}$ floats (e.g., 8 scalars for $H_{KV}=4$, adding 32 bytes).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Unconstrained Scalar Magnitude Instability
Because $w_K$ and $w_V$ are unconstrained parameters in $\mathbb{R}^{H_{KV}}$, large gradients during training can push certain $w_K$ entries to large magnitudes or negative values. Large values skew score magnitudes into Softmax saturation regions, while negative scalar weights invert key vectors, introducing pathological attention score distributions.

### Gradient Saturation under Head Pruning
If a learnable weight $(w_K)_h$ approaches $0$, the unnormalized attention score for group $h$ collapses to $0$. While this acts as soft head pruning, the gradient $\frac{\partial \mathcal{L}}{\partial (w_K)_h} \propto Q \cdot K$ vanishes as $w_K \to 0$, trapping the head in a permanently deactivated state regardless of input context.

### Materialization Latency of Repeated Tensors
Performing `repeat_interleave` after scaling creates full $[B, L, H_Q, d_k]$ key and value tensors in memory, inheriting the memory bandwidth limitations of standard MHA during execution.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Scaling & Indirect Indexing Kernel
WGQA hardware execution should eliminate explicit tensor scaling and expansion in DRAM by fusing $w_K$ and $w_V$ scaling directly into the FlashAttention tile loader.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Q [B, H_Q, L, d_k]                                      │
      │ K [B, H_KV, L, d_k]  <-- Unexpanded Raw KV Cache       │
      │ V [B, H_KV, L, d_k]  <-- Unexpanded Raw KV Cache       │
      │ w_K [H_KV], w_V [H_KV]                                  │
      └───────────────────────────┬────────────────────────────┘
                                  │ Direct Tile Load + In-Register Scaling
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM / Register Block                   │
 │ 1. Load K Tile: K_tile = Load(K_ptr + (h // γ))                  │
 │ 2. In-Register Scale: K_scaled = K_tile * w_K[h // γ]            │
 │ 3. Load V Tile: V_tile = Load(V_ptr + (h // γ))                  │
 │ 4. In-Register Scale: V_scaled = V_tile * w_V[h // γ]            │
 │ 5. Compute Matrix Multiply: S_tile = Q_tile @ K_scaled.T         │
 │ 6. Execute Online Softmax & Accumulate Output O_tile             │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ O [B, H_Q, L, d_k]                                      │
      └────────────────────────────────────────────────────────┘
```

### Triton Implementation Guidelines:
1. **In-Register Weight Fusion**:
   Load scalar values $w_K[g]$ and $w_V[g]$ (where $g = h // \gamma$) into warp registers during block initialization. Perform elementwise multiplication on key/value tiles directly in SRAM registers prior to matrix multiplication.
2. **Constrained Weight Parameterization**:
   To prevent scalar magnitude instability, re-parameterize $w_K$ and $w_V$ using a smooth activation function:
   \[
   \widetilde{w}_K = 2 \cdot \text{Sigmoid}(\alpha_K)
   \]
   where $\alpha_K$ is the trainable parameter vector, constraining multipliers to the range $(0, 2)$ with a default initialization of $1.0$.
3. **KV Cache Preservation**:
   During autoregressive inference, store raw unweighted key-value representations in the KV cache $[B, H_{KV}, L_{\text{past}}, d_k]$. Apply scale factors $w_K, w_V$ dynamically during the attention read phase, avoiding re-quantization or buffer modification overhead.