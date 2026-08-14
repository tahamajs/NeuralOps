# Minimal SVD Attention: Zero-Shot Low-Rank Singular Value Factorized Self-Attention

## 1. Abstract & Theoretical Motivation
Minimal SVD Attention (`MinimalSVDAttention`) introduces a zero-shot, parameter-efficient self-attention architecture designed to compress dense multi-head projections ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) via energy-thresholded truncated Singular Value Decomposition (SVD). Transformer parameter footprints are dominated by the weight matrices of query, key, value, and output linear transformations. Minimal SVD Attention mitigates this memory and compute burden by factorizing each projection matrix into a pair of rank-collapsed linear mappings: $W \approx A \cdot B$, where $B \in \mathbb{R}^{r \times d_{\text{in}}}$ down-projects the input and $A \in \mathbb{R}^{d_{\text{out}} \times r}$ up-projects to the target dimension. The rank $r$ is dynamically determined per matrix using an energy retention fraction $\eta \in (0, 1]$ based on the Eckart–Young–Mirsky theorem, which guarantees optimal low-rank matrix approximation in Frobenius norm. By initializing factors directly from singular values $\Sigma$ and orthogonal eigenvectors $U, V^\top$, Minimal SVD Attention enables zero-shot inference without fine-tuning while retaining standard quadratic softmax self-attention over sequence dimension $N$.

## 2. Mathematical Derivations & Formulations
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote a pretrained projection weight matrix. Its thin Singular Value Decomposition is defined as:
$$W = U \Sigma V^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i u_i v_i^\top$$
where $U \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $V \in \mathbb{R}^{d_{\text{in}} \times k}$.

The total Frobenius energy is given by:
$$\mathcal{E}_{\text{total}} = \|W\|_F^2 = \sum_{i=1}^k \sigma_i^2$$

For a target energy threshold $\eta \in (0, 1]$, the adaptive truncated rank $r$ is selected as:
$$r = \min \left\{ r' \in \mathbb{N} \;\middle|\; \frac{\sum_{i=1}^{r'} \sigma_i^2}{\mathcal{E}_{\text{total}}} \ge \eta \right\}$$
clamped such that $1 \le r \le \min(d_{\text{out}}, d_{\text{in}}, r_{\text{max}})$.

The factorized weight matrices $A \in \mathbb{R}^{d_{\text{out}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{in}}}$ are constructed as:
$$A = U_{:, :r} \text{diag}(\sigma_1, \dots, \sigma_r), \qquad B = V_{:r, :}^\top$$
such that the low-rank reconstruction satisfies:
$$W \approx A B = U_{:, :r} \text{diag}(\sigma_{1:r}) V_{:r, :}^\top$$

For input token embeddings $X \in \mathbb{R}^{B \times N \times D}$, the factorized query, key, and value transformations proceed as:
$$Q = X B_Q^\top A_Q^\top + b_Q, \quad K = X B_K^\top A_K^\top + b_K, \quad V = X B_V^\top A_V^\top + b_V$$

Reshaping into multi-head tensors $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ where $d_h = D/H$, the scaled dot-product attention scores and output context matrix are:
$$S = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{additive}}$$
$$A = \text{softmax}(S, \text{dim}=-1)$$
$$C = A V \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$\tilde{C} = \text{Reshape}(C) \in \mathbb{R}^{B \times N \times D}$$
$$O = \tilde{C} B_O^\top A_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The execution flow traces input tensors through down/up projections, multi-head splitting, attention matrix evaluation, context aggregation, and final factorized output projection:

1. **Input State**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Query Down-Projection**: $Q_{\text{low}} = X B_Q^\top \in \mathbb{R}^{B \times N \times r_Q}$ (via `down` layer with weight $[r_Q, D]$)
3. **Query Up-Projection**: $Q_{\text{full}} = Q_{\text{low}} A_Q^\top + b_Q \in \mathbb{R}^{B \times N \times D}$ (via `up` layer with weight $[D, r_Q]$)
4. **Key Down-Up Projections**: $K_{\text{full}} = (X B_K^\top) A_K^\top + b_K \in \mathbb{R}^{B \times N \times D}$ (intermediate rank $r_K$)
5. **Value Down-Up Projections**: $V_{\text{full}} = (X B_V^\top) A_V^\top + b_V \in \mathbb{R}^{B \times N \times D}$ (intermediate rank $r_V$)
6. **Multi-Head Reshape & Transpose**: $Q, K, V \to [B, H, N, d_h]$ where $d_h = D / H$
7. **Attention Score Computation**: $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
8. **Mask Application**: $S = S + (1 - M) \cdot (-10^9) \in \mathbb{R}^{B \times H \times N \times N}$
9. **Softmax & Dropout**: $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
10. **Context Matrix Aggregation**: $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
11. **Head Concatenation**: $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
12. **Output Projection Down-Up**: $O_{\text{low}} = \tilde{C} B_O^\top \in \mathbb{R}^{B \times N \times r_O} \implies O = O_{\text{low}} A_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Linear Down-Projections ($Q, K, V$): $3 \times 2 B N D r$ FLOPs
  - Linear Up-Projections ($Q, K, V$): $3 \times 2 B N r D$ FLOPs
  - Total Projection FLOPs: $2 B N D (r_Q + r_K + r_V + r_O)$ FLOPs (versus uncompressed $8 B N D^2$)
  - Attention Score ($Q K^\top$): $2 B H N^2 d_h = 2 B N^2 D$ FLOPs
  - Softmax & Masking: $\mathcal{O}(B H N^2)$ FLOPs
  - Value Weighting ($P V$): $2 B H N^2 d_h = 2 B N^2 D$ FLOPs
  - Total Module FLOPs: $2 B N D (r_Q + r_K + r_V + r_O) + 4 B N^2 D$
- **MAC Operations**:
  - Projections: $B N D (r_Q + r_K + r_V + r_O)$ MACs
  - Attention Matmuls: $2 B N^2 D$ MACs
- **Memory Footprint & KV Cache**:
  - Weight Parameters: $2 \times r \times D$ per projection vs $D^2$. For $D=768, r=40$, compression ratio is $\frac{768^2}{2 \times 40 \times 768} = 9.6\times$.
  - Activation Memory: Storing intermediate low-rank activations requires $\mathcal{O}(B N r)$ memory during backward pass.
  - Standard KV cache size: $2 \times B \times H \times N \times d_h = 2 B N D$.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Rank Collapse & Singular Value Decay**:
  Truncating singular values past rank $r$ discards energy $\epsilon = \sum_{i=r+1}^D \sigma_i^2$. If the singular spectrum decays slowly (heavy-tailed), low-rank truncation introduces significant spectral approximation error $\|W - AB\|_2 = \sigma_{r+1}$, leading to representation collapse in fine-grained attention patterns.
- **Quadratic Attention Bottleneck**:
  Minimal SVD compresses linear projections from $\mathcal{O}(D^2)$ parameter complexity to $\mathcal{O}(r D)$, but retains $\mathcal{O}(N^2 D)$ sequence length bottleneck. For long contexts ($N \gg D$), the attention score matrix $S \in \mathbb{R}^{B \times H \times N \times N}$ dominates both compute and memory.
- **Ill-Conditioning of SVD Factors**:
  Direct multiplication $A = U_{:, :r} \Sigma_{:r}$ concentrates singular value magnitude entirely into $A$, leaving $B = V_{:r, :}^\top$ orthogonal ($B B^\top = I_r$). This scale imbalance between $A$ and $B$ can degrade gradient propagation during fine-tuning.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Low-Rank Projection Kernel**:
  Instead of launching two separate linear kernels (`down` followed by `up`), a fused Triton kernel can compute $Y = (X B^\top) A^\top$ in a single GPU pass. By tiling sequence length $N$ into $B_N = 64$ and rank $r$ into $B_r = 16$ or $32$, intermediate low-rank activations $X B^\top \in \mathbb{R}^{B \times N \times r}$ are staged directly in SRAM without writing back to HBM.
- **FlashAttention Integration**:
  Once $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ are computed via low-rank projections, the standard $\mathcal{O}(N^2)$ attention map can be entirely fused into FlashAttention-2 / FlashAttention-3 Triton kernels, eliminating global memory roundtrips for the $[B, H, N, N]$ score matrix.
- **Symmetric Scaling Balance**:
  To prevent gradient imbalance, split singular values evenly between factors: $\tilde{A} = U_{:,:r} \Sigma_{:r}^{1/2}$ and $\tilde{B} = \Sigma_{:r}^{1/2} V_{:r,:}^\top$, ensuring matched condition numbers for numerical stability.