# SVD-LLM v2 Attention: Truncation-Aware Singular Value Factorized LLM Attention

## 1. Abstract & Theoretical Motivation
SVD-LLM v2 Attention (`SVDLLMV2Attention`) implements a truncation-aware, low-rank factorized Multi-Head Attention (MHA) module tailored for large language model (LLM) compression and zero-shot weight initialization. Linear transformations across Attention sub-layers ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) account for a large parameter footprint. SVD-LLM v2 decomposes each dense projection into explicit factor buffers: left singular vectors $U \in \mathbb{R}^{d_{\text{out}} \times r}$, diagonal singular value scaling parameters $S \in \mathbb{R}^r$, and right singular vectors $V^\top \in \mathbb{R}^{r \times d_{\text{in}}}$. Weight initialization uses truncated Singular Value Decomposition under energy preservation thresholds $\tau \in (0, 1]$, retaining top singular modes while padding residual ranks with scale-controlled noise. Decoupling singular values $S$ into a dedicated trainable vector enables downstream task adapters to recalibrate spectral scaling without destabilizing orthogonal basis matrices $U$ and $V^\top$.

## 2. Mathematical Derivations & Formulations
For a target linear projection matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, its thin SVD is:
$$W_0 = \bar{U} \Sigma \bar{V}^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \bar{u}_i \bar{v}_i^\top$$
where $\bar{U} \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $\bar{V} \in \mathbb{R}^{d_{\text{in}} \times k}$.

Under a target rank constraint $r \le k$, SVD-LLM v2 initializes factor parameters $(U, S, V^\top)$ as follows:
$$U_{:, :r} = \bar{U}_{:, :r} \text{diag}(\sigma_{1:r}), \qquad S_{1:r} = \mathbf{1}_r, \qquad V^\top_{:r, :} = \bar{V}_{:r, :}^\top$$

If allocated rank $r > k$, missing capacity is padded with random noise ($\epsilon = 10^{-2}$):
$$U_{:, k+1:r} \sim \mathcal{N}(0, 10^{-4}), \qquad S_{k+1:r} = 10^{-2} \cdot \mathbf{1}_{r-k}, \qquad V^\top_{k+1:r, :} \sim \mathcal{N}(0, 10^{-4})$$

Reconstruction of the weight operator during forward evaluation proceeds as:
$$W_{\text{eff}} = (U \odot S) V^\top = (U \text{diag}(S)) V^\top \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
where $\odot$ denotes column-wise Hadamard multiplication.

For input activations $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the factorized transformation evaluates as:
$$y = F.linear(X, W_{\text{eff}}) = X V \text{diag}(S) U^\top \in \mathbb{R}^{B \times N \times d_{\text{out}}}$$

Executing this across Query, Key, and Value projections yields multi-head tensors:
$$Q = \text{MultiHeadSplit}(F.linear(X, W_Q^{\text{eff}})) \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$K = \text{MultiHeadSplit}(F.linear(X, W_K^{\text{eff}})) \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$V = \text{MultiHeadSplit}(F.linear(X, W_V^{\text{eff}})) \in \mathbb{R}^{B \times H \times N \times d_h}$$

Scaled dot-product attention and output projections evaluate as:
$$S_{\text{scores}} = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}, \qquad A = \text{Dropout}(\text{Softmax}(S_{\text{scores}}, \text{dim}=-1))$$
$$C = A V \implies O = F.linear(\text{Reshape}(C), W_O^{\text{eff}}) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across SVD-LLM v2 Attention proceed as follows:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Reconstructed Projection Weight Operations ($Q, K, V$)**:
   - $W_Q = (U_Q \odot S_Q) V_Q^\top \in \mathbb{R}^{D \times D}$ (with $U_Q \in [D, r], S_Q \in [r], V_Q^\top \in [r, D]$)
   - $W_K = (U_K \odot S_K) V_K^\top \in \mathbb{R}^{D \times D}$, $W_V = (U_V \odot S_V) V_V^\top \in \mathbb{R}^{D \times D}$
3. **Linear Activation & Reshape**:
   - $Q_{\text{flat}} = F.linear(X, W_Q) \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{view}} [B, N, H, d_h] \xrightarrow{\text{permute}} [B, H, N, d_h]$
   - $K_{\text{flat}} = F.linear(X, W_K) \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{permute}} [B, H, N, d_h]$
   - $V_{\text{flat}} = F.linear(X, W_V) \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{permute}} [B, H, N, d_h]$
4. **Attention Score & Masking**:
   - $S_{\text{attn}} = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Mask addition: $S_{\text{attn}} \leftarrow S_{\text{attn}} + (1.0 - M) \cdot (-10^9)$
5. **Softmax Activation & Context Multiplication**:
   - $P = \text{Dropout}(\text{Softmax}(S_{\text{attn}}, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Head Concatenation & Output Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $W_O = (U_O \odot S_O) V_O^\top \in \mathbb{R}^{D \times D}$
   - $O = F.linear(\tilde{C}, W_O) \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Weight Reconstruction ($U \odot S @ V^\top$): $4 \times 2 D r D = 8 r D^2$ FLOPs
  - Linear Matmuls ($X W^\top$): $4 \times 2 B N D^2 = 8 B N D^2$ FLOPs
  - Dot-product attention ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Value weighting ($P V$): $2 B N^2 D$ FLOPs
  - Total Module FLOPs: $8 r D^2 + 8 B N D^2 + 4 B N^2 D$
- **MAC Operations**:
  - Projection Reconstructions: $4 r D^2$ MACs
  - Activation GEMMs: $4 B N D^2$ MACs
  - Attention Matrices: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Reduction**:
  - Dense Projection Parameters: $4 D^2$
  - SVD-LLM v2 Parameters: $4 \times (D r + r + r D) = 4 r (2D + 1)$
  - Parameter Compression Ratio: $\frac{4 D^2}{4 r (2D + 1)} \approx \frac{D}{2r}$. For $D=768, r=32$, ratio is $12.0\times$.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **On-the-Fly Weight Reconstruction Overhead**:
  Reconstructing $W_{\text{eff}} = (U \odot S) V^\top$ per forward pass allocates $D \times D$ dense matrices in GPU memory, introducing unnecessary HBM memory traffic compared to evaluating factored projections $(X V) \text{diag}(S) U^\top$.
- **Bias Parameter Omission**:
  SVD-LLM v2 omits linear layer bias parameters $b \in \mathbb{R}^{D}$. Discarding pretrained biases introduces systematic output mean shifts during zero-shot evaluation.
- **Unconstrained Singular Value Drift**:
  Allowing vector $S \in \mathbb{R}^r$ to be trained without non-negativity or monotonic sorting constraints can cause singular values to cross zero or explode during fine-tuning.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Factored GEMM Triton Kernel**:
  Bypass explicit weight matrix reconstruction $W = (U \odot S) V^\top$. Implement a Triton kernel computing $y = \left((X V) \odot S\right) U^\top$ directly, staging $B_N \times r$ intermediate activations in SRAM.
- **Monotonic Singular Value Regularization**:
  Apply softplus activation or monotonic penalty to vector $S$ during backpropagation: $\mathcal{L}_{\text{mono}} = \sum_{i=1}^{r-1} \text{ReLU}(S_{i+1} - S_i)$, enforcing $S_1 \ge S_2 \ge \dots \ge S_r \ge 0$.
- **FlashAttention Integration**:
  Stream intermediate factorized projections $Q, K, V$ into FlashAttention-2 / FlashAttention-3 kernels to execute fused online softmax attention without allocating $N \times N$ score matrices in HBM.