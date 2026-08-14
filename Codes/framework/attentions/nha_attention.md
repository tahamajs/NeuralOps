# Native Hybrid Attention (NHA): Token-Selective Full Attention with Linear Recurrent Kernel Memory

## 1. Abstract & Theoretical Motivation
Native Hybrid Attention (`NHAAttention`, registered as `nha`) introduces a dynamic token-partitioning hybrid self-attention mechanism designed to break the quadratic scaling barrier $\mathcal{O}(N^2)$ in long-sequence Transformers. Standard self-attention allocates equal quadratic compute across all sequence tokens regardless of token salience. NHA dynamically partitions sequence tokens into high-salience heavy hitters $\mathcal{S}_{\text{top}}$ (accounting for fraction $\rho = \text{full\_ratio} \ll 1$) and contextual background tokens. Heavy-hitter tokens undergo exact multi-head softmax self-attention over the selected subspace, while contextual background tokens are processed via a linear-time positive feature map kernel $\phi(x) = \text{ELU}(x) + 1.0$. By fusing exact softmax interactions for top-norm query tokens with $\mathcal{O}(N \cdot d_h^2)$ associative memory aggregation for context tokens, NHA reduces total sequence complexity to $\mathcal{O}(\rho N^2 + N d_h^2)$.

## 2. Mathematical Derivations & Formulations
Given token input representations $X \in \mathbb{R}^{B \times N \times D}$, linear transformations derive multi-head query, key, and value tensors $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ where $d_h = D / H$.

### 1. Salient Heavy-Hitter Token Selection
Token importance scores $s_{b, i}$ are evaluated across sequence positions $i \in \{1, \dots, N\}$ using average query feature vector norms across heads:
$$s_{b, i} = \frac{1}{H} \sum_{h=1}^H \|Q_{b, h, i, :}\|_2 \in \mathbb{R}$$

The top-$k$ index set $\mathcal{S}_{\text{top}}^{(b)}$ is selected per batch sample $b$ under full attention ratio $\rho \in (0, 1]$ ($k = \max(1, \lfloor N \cdot \rho \rfloor)$):
$$\mathcal{S}_{\text{top}}^{(b)} = \text{TopK}\left( \{s_{b, i}\}_{i=1}^N, k \right)$$

A boolean selection mask $M_{\text{full}} \in \{0, 1\}^{B \times H \times N \times d_h}$ is formed:
$$M_{\text{full}}[b, h, i, :] = \begin{cases} \mathbf{1}_{d_h} & \text{if } i \in \mathcal{S}_{\text{top}}^{(b)} \\ \mathbf{0}_{d_h} & \text{otherwise} \end{cases}$$

### 2. Heavy-Hitter Full Softmax Attention Path
Queries and keys are gated by $M_{\text{full}}$:
$$Q_{\text{full}} = Q \odot M_{\text{full}}, \qquad K_{\text{full}} = K \odot M_{\text{full}}$$
$$S_{\text{full}} = \frac{Q_{\text{full}} K_{\text{full}}^\top}{\sqrt{d_h}} + M_{\text{attn}}$$
$$P_{\text{full}} = \text{Dropout}(\text{Softmax}(S_{\text{full}}, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$$
$$\text{Out}_{\text{full}} = P_{\text{full}} V \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 3. Context Linear Attention Path
Context tokens evaluate kernel feature maps $\phi(x) = \text{ELU}(x) + 1.0 \ge 0$:
$$\phi_Q = \text{ELU}(Q) + 1.0, \qquad \phi_K = \text{ELU}(K) + 1.0 \in \mathbb{R}^{B \times H \times N \times d_h}$$

Key-value associative memory matrix $K V_{\text{lin}}$ and normalization denominator $Z_{\text{lin}}$ are computed in linear time:
$$K V_{\text{lin}} = \sum_{j=1}^N \phi_K[:, :, j, :]^\top V[:, :, j, :] \in \mathbb{R}^{B \times H \times d_h \times d_h}$$
$$Z_{\text{lin}} = \sum_{j=1}^N \phi_K[:, :, j, :] \in \mathbb{R}^{B \times H \times d_h}$$
$$\text{Out}_{\text{lin}} = \frac{\phi_Q K V_{\text{lin}}}{\phi_Q Z_{\text{lin}}^\top + \epsilon} \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 4. Output Fusion
The final multi-head context tensor $O_{\text{head}}$ blends the two pathways via element-wise gating:
$$O_{\text{head}} = \mathbf{M}_{\text{full}} \odot \text{Out}_{\text{full}} + (1 - \mathbf{M}_{\text{full}}) \odot \text{Out}_{\text{lin}} \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{Reshape}(O_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step execution path and shape transformations across NHA Attention are traced below:

1. **Input State**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dense Linear Projections**:
   - $Q_{\text{flat}} = X W_Q + b_Q \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{reshape}} [B, H, N, d_h]$
   - $K_{\text{flat}} = X W_K + b_K \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{reshape}} [B, H, N, d_h]$
   - $V_{\text{flat}} = X W_V + b_V \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{reshape}} [B, H, N, d_h]$
3. **Query Norm Importance Scoring & Selection**:
   - $S_{\text{token}} = \|Q\|_{F, \text{head}} \in \mathbb{R}^{B \times N}$
   - Extract top $k = \lfloor N \cdot \rho \rfloor$ indices per batch $\implies M_{\text{full}} \in \{0, 1\}^{B \times H \times N \times d_h}$
4. **Full Attention Branch**:
   - $Q_{\text{full}} = Q \cdot M_{\text{full}}$, $K_{\text{full}} = K \cdot M_{\text{full}}$
   - $S_{\text{full}} = Q_{\text{full}} K_{\text{full}}^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - $P_{\text{full}} = \text{Dropout}(\text{Softmax}(S_{\text{full}} + M_{\text{attn}})) \in \mathbb{R}^{B \times H \times N \times N}$
   - $\text{Out}_{\text{full}} = P_{\text{full}} V \in \mathbb{R}^{B \times H \times N \times d_h}$
5. **Linear Kernel Attention Branch**:
   - $\phi_Q = \text{ELU}(Q) + 1.0$, $\phi_K = \text{ELU}(K) + 1.0 \in \mathbb{R}^{B \times H \times N \times d_h}$
   - Associative memory reduction: $K V_{\text{lin}} = \text{einsum}("b h n d, b h n e \to b h d e", \phi_K, V) \in \mathbb{R}^{B \times H \times d_h \times d_h}$
   - Denominator sum: $Z_{\text{lin}} = \phi_K.\text{sum}(\text{dim}=2) \in \mathbb{R}^{B \times H \times d_h}$
   - Linear context aggregation: $\text{Out}_{\text{lin}} = \text{einsum}("b h n d, b h d e \to b h n e", \phi_Q, K V_{\text{lin}}) / (\phi_Q Z_{\text{lin}}^\top + \epsilon)$
6. **Mask Gated Fusion & Output Projection**:
   - $O_{\text{head}} = \text{where}(M_{\text{full}}, \text{Out}_{\text{full}}, \text{Out}_{\text{lin}}) \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $O = \text{Reshape}(O_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Linear Projections ($Q, K, V, O$): $4 \times 2 B N D^2 = 8 B N D^2$ FLOPs
  - Query Norm Computation: $2 B H N d_h = 2 B N D$ FLOPs
  - Heavy-Hitter Score ($Q_{\text{full}} K_{\text{full}}^\top$): $2 B H (\rho N) N d_h = 2 \rho B N^2 D$ FLOPs
  - Heavy-Hitter Context ($P_{\text{full}} V$): $2 \rho B N^2 D$ FLOPs
  - Linear Memory Contraction ($\phi_K^\top V$): $2 B H N d_h^2 = 2 B N D d_h$ FLOPs
  - Linear Context Aggregation ($\phi_Q K V_{\text{lin}}$): $2 B N D d_h$ FLOPs
  - Total Module FLOPs: $8 B N D^2 + 4 \rho B N^2 D + 4 B N D d_h$ (Sub-quadratic for $\rho \ll 1$)
- **MAC Operations**:
  - Dense Projections: $4 B N D^2$ MACs
  - Heavy-Hitter Matmuls: $2 \rho B N^2 D$ MACs
  - Linear Attention Contractions: $2 B N D d_h$ MACs
- **Memory Footprint & KV Cache**:
  - Full Score Matrix Storage: $[B, H, N, N]$ (can be sparsified to $[B, H, \rho N, N]$)
  - Recurrent Linear Memory State: $K V_{\text{lin}} \in \mathbb{R}^{B \times H \times d_h \times d_h}$ (requiring only $B H d_h^2$ memory elements).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Linear Kernel Masking Artifacts**:
  Adding large negative offsets `(mask - 1) * -1e9` to positive feature maps $\phi_K = \text{ELU}(K) + 1.0$ forces feature values negative. This violates the non-negativity constraint $\phi(x) \ge 0$, causing denominator collapse to zero or NaN outputs.
- **Query Norm Selection Bias**:
  Selecting heavy hitters solely via query Euclidean norm $\|Q\|_2$ assumes query magnitude correlates directly with attention weight salience. In models with out-of-distribution outlier tokens (e.g., sink tokens), query norm selection can persistently allocate heavy-hitter budget to uninformative tokens.
- **Disconnected Attention Graphs**:
  In the heavy-hitter branch, non-selected tokens have their queries zeroed out ($Q_{\text{full}} = 0$). Selected tokens can only attend to other selected tokens, destroying global context visibility for heavy hitters.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Chunk-Wise Linear Attention Kernel**:
  Implement a fused Triton kernel for the linear attention branch ($\phi_K^\top V$). Processing sequence tiles $B_N = 64$ in local SRAM avoids instantiating $B \times H \times d_h \times d_h$ memory state tensors in global HBM.
- **Zero-Padded Masking for Kernel Attention**:
  Replace additive negative masking with zero-multiplication: $\tilde{\phi}_K = (\text{ELU}(K) + 1.0) \odot M_{\text{valid}}$. This strictly preserves positive feature map bounds $\phi_K \ge 0$ and avoids numerical NaN instabilities.
- **FlashAttention Integration for Heavy Hitters**:
  Stream top-$k$ query/key token indices into sparse FlashAttention-2 Triton kernels to evaluate heavy-hitter attention maps directly without computing full $N \times N$ score matrices.