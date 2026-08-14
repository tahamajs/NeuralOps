# Rank-Augmented Linear Attention (RALA): Parametric Low-Rank Kernel Matrix Augmentation

## 1. Abstract & Theoretical Motivation
Rank-Augmented Linear Attention (`RALALinearAttention`, registered as `rala`) presents an expressive, linear-time self-attention architecture designed to overcome representational capacity bottlenecks in positive kernel linear attention. Traditional kernelized linear attention reduces sequence length complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot d_h^2)$ using feature maps $\phi(x) = \text{ELU}(x) + 1.0$. However, kernelized feature maps exhibit limited rank expressivity compared to full softmax attention. RALA addresses this by augmenting the linear attention kernel matrix with a parametric, learnable low-rank matrix correction term $R_h = U_h V_h^\top \in \mathbb{R}^{d_h \times d_h}$ (where $U_h, V_h \in \mathbb{R}^{d_h \times r}$ with rank $r \ll d_h$). Controlled by a learnable scaling parameter $\alpha$, RALA injects global parametric cross-feature interactions into associative memory while maintaining strict $\mathcal{O}(N \cdot d_h^2)$ linear sequence scaling.

## 2. Mathematical Derivations & Formulations
Given multi-head query, key, and value representations $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ derived via linear transformations $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$, RALA evaluates positive feature maps $\phi(x) = \text{ELU}(x) + 1.0 \ge 0$:
$$\phi_Q = \text{ELU}(Q) + 1.0, \qquad \phi_K = \text{ELU}(K) + 1.0 \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 1. Standard Linear Kernel Pathway
The key-value associative memory state matrix $K V \in \mathbb{R}^{B \times H \times d_h \times d_h}$ and normalization vector $Z \in \mathbb{R}^{B \times H \times d_h}$ are accumulated over sequence length $N$:
$$K V = \sum_{j=1}^N \phi_K[:, :, j, :]^\top V[:, :, j, :] \in \mathbb{R}^{B \times H \times d_h \times d_h}$$
$$Z = \sum_{j=1}^N \phi_K[:, :, j, :] \in \mathbb{R}^{B \times H \times d_h}$$

The unaugmented linear context $\text{Out}_{\text{lin}}$ is:
$$\text{Out}_{\text{lin}} = \frac{\phi_Q K V}{\phi_Q Z^\top + \epsilon} \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 2. Parametric Low-Rank Augmentation Pathway
For each attention head $h \in \{1, \dots, H\}$, low-rank correction matrices $U_h, V_h \in \mathbb{R}^{d_h \times r}$ (rank $r = 8$) construct head-wise rank augmentation matrices $R_h$:
$$R_h = U_h V_h^\top \in \mathbb{R}^{d_h \times d_h}$$

The low-rank augmented context $\text{Out}_{\text{aug}}$ is computed by right-multiplying query feature maps $\phi_Q$ with $R_h$:
$$\text{Out}_{\text{aug}} = \frac{\phi_Q R_h}{\phi_Q Z^\top + \epsilon} \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 3. Fused Output Representation
Combining unaugmented linear context and low-rank augmented context yields the complete RALA context representation:
$$O_{\text{head}} = \text{Out}_{\text{lin}} + \alpha \cdot \text{Out}_{\text{aug}} \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{Reshape}(O_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$
where $\alpha \in \mathbb{R}$ is a learnable scalar initialized to $\alpha = 0.1$.

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across RALA Attention are traced below:

1. **Input State**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dense Linear Projections**:
   - $Q_{\text{flat}} = X W_Q + b_Q \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{view}} [B, N, H, d_h] \xrightarrow{\text{transpose}} [B, H, N, d_h]$
   - $K_{\text{flat}} = X W_K + b_K \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{transpose}} [B, H, N, d_h]$
   - $V_{\text{flat}} = X W_V + b_V \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{transpose}} [B, H, N, d_h]$
3. **Feature Map Transformation & Masking**:
   - $\phi_Q = \text{ELU}(Q) + 1.0 \in \mathbb{R}^{B \times H \times N \times d_h}$
   - Mask processing: $\phi_K = \text{apply\_linear\_mask}(\text{ELU}(K) + 1.0, \text{mask}) \in \mathbb{R}^{B \times H \times N \times d_h}$
4. **Associative Memory Reduction**:
   - $K V = \text{einsum}("b h n d, b h n e \to b h d e", \phi_K, V) \in \mathbb{R}^{B \times H \times d_h \times d_h}$
   - Normalization vector: $Z = \phi_K.\text{sum}(\text{dim}=2) \in \mathbb{R}^{B \times H \times d_h}$
   - Denominator: $\mathcal{D} = \text{einsum}("b h n d, b h d \to b h n", \phi_Q, Z).\text{unsqueeze}(-1) + 1e-8 \in \mathbb{R}^{B \times H \times N \times 1}$
5. **Linear Context & Low-Rank Augmentation**:
   - $\text{Out}_{\text{lin}} = \text{einsum}("b h n d, b h d e \to b h n e", \phi_Q, K V) / \mathcal{D}$
   - Low-rank matrix construction: $R = \text{einsum}("h d r, h e r \to h d e", U, V) \in \mathbb{R}^{H \times d_h \times d_h}$
   - $\text{Out}_{\text{aug}} = \text{einsum}("b h n d, h d e \to b h n e", \phi_Q, R) / \mathcal{D}$
6. **Augmentation Fusion & Output Projection**:
   - $O_{\text{head}} = \text{Out}_{\text{lin}} + \alpha \cdot \text{Out}_{\text{aug}} \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $O = \text{Reshape}(O_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Linear Projections ($Q, K, V, O$): $4 \times 2 B N D^2 = 8 B N D^2$ FLOPs
  - Associative Memory Reduction ($\phi_K^\top V$): $2 B H N d_h^2 = 2 B N D d_h$ FLOPs
  - Linear Context Aggregation ($\phi_Q K V$): $2 B N D d_h$ FLOPs
  - Low-Rank Matrix Reconstruction ($U V^\top$): $2 H d_h^2 r$ FLOPs
  - Low-Rank Augmentation Aggregation ($\phi_Q R$): $2 B N D d_h$ FLOPs
  - Total Module FLOPs: $8 B N D^2 + 6 B N D d_h + 2 H d_h^2 r$ (Strictly $\mathcal{O}(N)$ in sequence length)
- **MAC Operations**:
  - Dense Projections: $4 B N D^2$ MACs
  - Associative Memory Contractions: $3 B N D d_h$ MACs
- **Memory Footprint & Parameter Count**:
  - Projection Parameters: $4 D^2$
  - Parametric Correction Parameters ($U, V$): $2 \times H \times d_h \times r = 2 r D$. For $D=768, H=12, r=8$, correction parameters $= 2 \times 8 \times 768 = 12,288$ ($< 0.5\%$ extra parameters).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Linear Masking Offset Instabilities**:
  Applying `_apply_linear_mask` by adding $(1 - M) \cdot (-10^9)$ to $\phi_K$ introduces massive negative numbers into positive feature maps. This violates the non-negativity constraint $\phi(x) \ge 0$, causing denominator underflow or NaNs.
- **Static Parametric Bias Limitations**:
  Low-rank correction matrix $R_h = U_h V_h^\top$ is static with respect to input sequences $X$. While it learns head-specific global feature interactions, it cannot adapt dynamically to input-dependent token relationships.
- **Unbounded Augmentation Scaling**:
  Allowing scalar $\alpha$ to drift unconstrained during optimization can cause $\alpha \cdot \text{Out}_{\text{aug}}$ to dominate $\text{Out}_{\text{lin}}$, degrading associative memory retrieval.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Chunk-Wise RALA Kernel**:
  Implement a custom Triton kernel fusing associative memory contraction ($\phi_K^\top V$), low-rank augmentation ($\phi_Q R$), and denominator normalization into a single CUDA pass, processing sequence tiles $B_N = 64$ in local SRAM.
- **Zero-Padded Kernel Masking**:
  Replace additive negative masking with zero-multiplication: $\tilde{\phi}_K = (\text{ELU}(K) + 1.0) \odot M_{\text{valid}}$. This strictly preserves positive feature map bounds $\phi_K \ge 0$ and avoids numerical NaN instabilities.
- **Bounded Sigmoidal Augmentation Scale**:
  Reparameterize scaling parameter $\alpha$ as $\alpha = 2 \cdot \sigma(\gamma)$ where $\gamma$ is a trainable logit, bounding the augmentation scaling factor to $(0, 2.0)$ for numerical stability.
