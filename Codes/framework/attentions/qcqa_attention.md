# Quality and Capacity-Aware Grouped Query Attention (QCQA): Dynamic Soft-Routing over Compressed Key-Value Subspaces

## 1. Abstract & Theoretical Motivation

Standard Grouped-Query Attention (GQA) assigns a fixed, static mapping between query heads and shared key-value (KV) heads. This rigid structural prior assumes that the spatial and semantic relevance of KV heads remains constant across all tokens and sequence positions. Quality and Capacity-Aware Grouped Query Attention (QCQA) overcomes this limitation by implementing dynamic continuous soft-routing over a compressed set of $H_{KV}$ Key-Value heads.

In QCQA, a query-conditioned linear router network evaluates the feature state of each query vector $q_{b, l, h} \in \mathbb{R}^{d_k}$ at every token position, synthesizing a continuous Softmax routing distribution $\pi_{b, l, h} \in \Delta^{H_{KV}-1}$. Key and Value representations for each query head are dynamically constructed as convex combinations of the compressed KV heads. This approach preserves the KV cache compression benefits of GQA during decoding while dynamically allocating KV subspace capacity based on input token content.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ be the input sequence, where $B$ is batch size, $L$ is sequence length, and $D$ is model dimension. Let $H_Q$ be the number of query heads and $H_{KV}$ be the number of compressed KV heads ($H_{KV} \le H_Q$), with head dimension $d_k = D / H_Q$.

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

Reshaping into head dimensions: $Q \in \mathbb{R}^{B \times L \times H_Q \times d_k}$, $K, V \in \mathbb{R}^{B \times L \times H_{KV} \times d_k}$.

### Query-Conditioned Soft-Routing Distribution:
The linear router $W_R \in \mathbb{R}^{d_k \times H_{KV}}$ projects query vectors into routing logits. For query head $h \in \{1, \dots, H_Q\}$ at sequence position $l$:

\[
R_{b, l, h, j} = \frac{1}{\tau} \sum_{m=1}^{d_k} Q_{b, l, h, m} \cdot (W_R)_{m, j}, \quad j \in \{1, \dots, H_{KV}\}
\]

where $\tau > 0$ is the router temperature hyperparameter. The routing weight vector $\pi_{b, l, h} \in \mathbb{R}^{H_{KV}}$ is obtained via row-wise Softmax over KV heads:

\[
\pi_{b, l, h, j} = \text{Softmax}_j (R_{b, l, h, j}) = \frac{\exp(R_{b, l, h, j})}{\sum_{j'=1}^{H_{KV}} \exp(R_{b, l, h, j'})}
\]

### Dynamic Key-Value Subspace Synthesis:
Using the routing distribution $\pi$, synthesized key vectors $\widetilde{K} \in \mathbb{R}^{B \times L \times H_Q \times d_k}$ and value vectors $\widetilde{V} \in \mathbb{R}^{B \times L \times H_Q \times d_k}$ are generated via convex combinations:

\[
\widetilde{K}_{b, l, h, m} = \sum_{j=1}^{H_{KV}} \pi_{b, l, h, j} \cdot K_{b, l, j, m}
\]
\[
\widetilde{V}_{b, l, h, m} = \sum_{j=1}^{H_{KV}} \pi_{b, l, h, j} \cdot V_{b, l, j, m}
\]

In batched tensor operations, this corresponds to batched matrix multiplication over flattened sequence dimensions:

\[
\widetilde{K}_{\text{flat}} = \text{BMM}\left(\pi_{\text{flat}}, K_{\text{flat}}\right) \in \mathbb{R}^{(B \cdot L) \times H_Q \times d_k}
\]
\[
\widetilde{V}_{\text{flat}} = \text{BMM}\left(\pi_{\text{flat}}, V_{\text{flat}}\right) \in \mathbb{R}^{(B \cdot L) \times H_Q \times d_k}
\]

where $\pi_{\text{flat}} \in \mathbb{R}^{(B \cdot L) \times H_Q \times H_{KV}}$ and $K_{\text{flat}}, V_{\text{flat}} \in \mathbb{R}^{(B \cdot L) \times H_{KV} \times d_k}$.

### Scaled Dot-Product Attention:
Transposing queries and synthesized key-values to $\mathbb{R}^{B \times H_Q \times L \times d_k}$:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot \widetilde{K}_{b, h, j, m}
\]
\[
\widetilde{S}_{b, h, i, j} = S_{b, h, i, j} + M_{b, 1, i, j}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (\widetilde{S}_{b, h, i, j})
\]
\[
O_{b, h, i, m} = \sum_{j=1}^L \text{Dropout}(A, p_{\text{drop}})_{b, h, i, j} \cdot \widetilde{V}_{b, h, j, m}
\]
\[
Y = \text{Concat}(O_1, \dots, O_{H_Q}) \cdot W_O \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, H_Q * d_k) [W_q] ──> Q: [B, L, H_Q, d_k]
  ├──> Linear(D, H_KV * d_k) [W_k] ──> K: [B, L, H_KV, d_k]
  └──> Linear(D, H_KV * d_k) [W_v] ──> V: [B, L, H_KV, d_k]
                                │
                                ├──> Linear(d_k, H_KV) [router] ──> route_logits: [B, L, H_Q, H_KV]
                                ├──> Softmax(dim=-1) / tau ──> route_weights (π): [B, L, H_Q, H_KV]
                                │
                                ├──> Reshape: π_flat [B*L, H_Q, H_KV], K_flat [B*L, H_KV, d_k]
                                ├──> BMM(π_flat, K_flat) ──> K_mixed: [B, L, H_Q, d_k]
                                ├──> BMM(π_flat, V_flat) ──> V_mixed: [B, L, H_Q, d_k]
                                │
                                ├──> transpose(1, 2) ──> Q, K_mixed, V_mixed: [B, H_Q, L, d_k]
                                │
                                ├──> MatMul: Q @ K_mixed.transpose(-2, -1) / sqrt(d_k) ──> [B, H_Q, L, L]
                                ├──> Add attention_mask ──> [B, H_Q, L, L]
                                ├──> Softmax(dim=-1) & Dropout ──> attn: [B, H_Q, L, L]
                                │
                                └──> MatMul: attn @ V_mixed ──> context: [B, H_Q, L, d_k]
                                        │
                                        └──> transpose(1, 2).view([B, L, H_Q * d_k])
                                        └──> Linear(H_Q * d_k, D) [W_o] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### FLOPs & MAC Operations Breakdown

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H_Q$ query heads, $H_{KV}$ key-value heads, $\gamma = H_Q / H_{KV}$, and $d_k = D / H_Q$.

1. **Linear Projections ($W_Q, W_K, W_V$)**:
   - $W_Q$: $2 B L D^2$ FLOPs ($B L D^2$ MACs).
   - $W_K, W_V$: $\frac{4 B L D^2}{\gamma}$ FLOPs ($\frac{2 B L D^2}{\gamma}$ MACs).

2. **Query Routing Projection ($W_R$)**:
   - Operation: $[B \cdot L \cdot H_Q, d_k] \times [d_k, H_{KV}]$.
   - MACs: $B \cdot L \cdot H_Q \cdot d_k \cdot H_{KV} = B L D H_{KV}$
   - FLOPs: $2 B L D H_{KV} + \text{Softmax overhead } (\approx 5 B L H_Q H_{KV})$

3. **Dynamic KV Subspace Mixture ($\text{BMM}(\pi, K)$ & $\text{BMM}(\pi, V)$)**:
   - Operation: $2 \times [B \cdot L, H_Q, H_{KV}] \times [B \cdot L, H_{KV}, d_k]$.
   - MACs: $2 \times B \cdot L \cdot H_Q \cdot H_{KV} \cdot d_k = 2 B L D H_{KV}$
   - FLOPs: $4 B L D H_{KV}$

4. **Attention Matrix Computations ($Q K^\top$ & $A V$)**:
   - MACs: $2 B L^2 D$
   - FLOPs: $4 B L^2 D$

5. **Output Projection ($W_O$)**:
   - MACs: $B L D^2$, FLOPs: $2 B L D^2$

### Total Complexity:
- **Total MACs**: $2 B L D^2 \left(1 + \frac{1}{\gamma}\right) + 3 B L D H_{KV} + 2 B L^2 D$
- **Total FLOPs**: $4 B L D^2 \left(1 + \frac{1}{\gamma}\right) + 6 B L D H_{KV} + 5 B L H_Q H_{KV} + 4 B L^2 D$
- **Autoregressive KV Cache Capacity**: $2 B L_{\text{past}} H_{KV} d_k \times \text{bytes} = \frac{2 B L_{\text{past}} D}{\gamma} \times \text{bytes}$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Dynamic KV Synthesis Memory Overhead
During forward passes in standard PyTorch, synthesizing $\widetilde{K}$ and $\widetilde{V}$ requires allocating intermediate tensors of size $[B, L, H_Q, d_k]$. This quadruples token memory writes back to HBM before computing the attention matrix, negating throughput gains for short sequence lengths.

### Router Entropy Collapse
If the router temperature $\tau$ is too small ($\tau \to 0$), the routing distribution degenerates into a hard ArgMax function, destabilizing gradient propagation to non-selected KV heads. Conversely, high temperatures ($\tau \to \infty$) produce uniform distributions $\pi_{h, j} = \frac{1}{H_{KV}}$, collapsing QCQA to Multi-Query Attention (MQA) with averaged representations.

### KV Subspace Manifold Degeneracy
If key heads $K_j$ align into parallel subspaces during training, $\text{rank}(\widetilde{K}_h) < d_k$, causing effective rank collapse in score computation.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Soft-Routing Triton Kernel Architecture
To achieve optimal execution throughput, routing logic and KV mixture synthesis must be fused directly inside the SRAM inner-loop of Triton FlashAttention.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Q [B, H_Q, L, d_k]                                      │
      │ K [B, H_KV, L, d_k]  <-- Stored Unmixed KV Cache       │
      │ V [B, H_KV, L, d_k]  <-- Stored Unmixed KV Cache       │
      │ W_R [d_k, H_KV]                                        │
      └───────────────────────────┬────────────────────────────┘
                                  │ Load Q & Router Weights
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Compute Routing Logits: R_h = Q_tile @ W_R / tau              │
 │ 2. Compute Softmax Routing Weights: π_h = Softmax(R_h)           │
 │ 3. Synthesize Mixed KV Tiles On-The-Fly:                         │
 │    K_mixed = sum_j (π_{h, j} * K_tile[j])                         │
 │    V_mixed = sum_j (π_{h, j} * V_tile[j])                         │
 │ 4. Compute Attention Scores: S_tile = Q_tile @ K_mixed.T / sqrt  │
 │ 5. Online Softmax Accumulation into Output Tile                  │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Stream Output
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Output O [B, H_Q, L, d_k]                               │
      └────────────────────────────────────────────────────────┘
```

### Triton Implementation Guidelines:
1. **On-Chip Softmax Mixture Fusion**:
   Avoid materializing $[B, L, H_Q, d_k]$ tensors in DRAM. Keep routing logits $R_{b, l, h, j}$ in SRAM vector registers and perform instantaneous linear combinations over loaded KV tiles.
2. **Temperature Scheduling**:
   Implement a dynamic temperature annealing schedule:
   \[
   \tau(t) = \tau_{\text{min}} + (\tau_{\text{max}} - \tau_{\text{min}}) \cdot e^{-\lambda t}
   \]
   allowing broad subspace exploration early in training followed by sharp, deterministic routing in later epochs.
3. **Optimized Autoregressive KV Decoding**:
   During inference decoding ($L_{\text{query}}=1$), read raw unmixed tokens from KV cache $[B, H_{KV}, L_{\text{past}}, d_k]$ and perform single-step fused routing matrix multiplication directly in GPU register files.
