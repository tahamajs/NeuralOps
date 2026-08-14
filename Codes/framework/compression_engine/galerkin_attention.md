# Galerkin Transformer: Petrov-Galerkin Subspace Projection Attention

## 1. Abstract & Theoretical Motivation
Galerkin Attention (`GalerkinAttention`, registered as `galerkin`) implements a Petrov-Galerkin subspace projection linear attention framework designed to compress sequence dimension scaling from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot r)$. Standard Transformer Multi-Head Attention evaluates full $N \times N$ pairwise query-key Gram matrices. Galerkin Attention projects high-dimensional key and value sequence spaces onto an $r$-dimensional reduced subspace ($r \ll N$) using a random Gaussian sketching operator $\Omega \in \mathbb{R}^{N \times r}$. By substituting full softmax attention maps with projected $N \times r$ attention scores, Galerkin Attention drastically reduces memory and compute footprints while preserving zero-shot weight compatibility with pretrained standard Multi-Head Attention models.

## 2. Mathematical Derivations & Formulations
Let $X \in \mathbb{R}^{B \times N \times D}$ denote hidden activation inputs. Linear transformations derive multi-head query, key, and value representations $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ ($d_h = D / H$).

### 1. Petrov-Galerkin Random Subspace Sketching
A random Gaussian sketching matrix $\Omega \in \mathbb{R}^{N \times r}$ with scaling factor $1 / \sqrt{N}$ is generated:
$$\Omega_{i, j} \sim \mathcal{N}\left(0, \frac{1}{N}\right) \in \mathbb{R}^{N \times r}$$

Key and value tensors are projected onto the $r$-dimensional reduced sequence subspace:
$$K_{\text{sketch}} = K \Omega \in \mathbb{R}^{B \times H \times r \times d_h}, \qquad V_{\text{sketch}} = V \Omega \in \mathbb{R}^{B \times H \times r \times d_h}$$

### 2. Subspace Attention Score & Context Evaluation
Scaled dot-product attention is computed between full-length query sequence $Q$ and sketched key basis $K_{\text{sketch}}$:
$$S_{\text{sketch}} = \frac{Q K_{\text{sketch}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times N \times r}$$

Under valid sequence mask $M \in \{0, 1\}^{B \times N}$, scores are masked prior to softmax normalization:
$$\tilde{S}_{\text{sketch}} = \begin{cases} S_{\text{sketch}}[b, h, i, k] & \text{if } M[b, i] = 1 \\ -10^9 & \text{if } M[b, i] = 0 \end{cases}$$
$$P_{\text{sketch}} = \text{Dropout}\left(\text{Softmax}(\tilde{S}_{\text{sketch}}, \text{dim}=-1)\right) \in \mathbb{R}^{B \times H \times N \times r}$$

Context representation $C_{\text{head}}$ is obtained via right-multiplication with sketched values $V_{\text{sketch}}$:
$$C_{\text{head}} = P_{\text{sketch}} V_{\text{sketch}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

Multi-head outputs are concatenated and transformed via pretrained output weight $W_O$:
$$O = \text{Reshape}(C_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step execution path across Galerkin Attention is traced below:

1. **Input State**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dense Linear Projections**:
   - $Q = \text{Reshape}(X W_Q + b_Q) \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $K = \text{Reshape}(X W_K + b_K) \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $V = \text{Reshape}(X W_V + b_V \in \mathbb{R}^{B \times H \times N \times d_h}$
3. **Random Subspace Sketch Generation**:
   - $\Omega = \frac{1}{\sqrt{N}} \text{randn}(N, r) \in \mathbb{R}^{N \times r}$
4. **Key & Value Subspace Contracting Contractions**:
   - $K_{\text{sketch}} = \text{einsum}("b h n d, n r \to b h r d", K, \Omega) \in \mathbb{R}^{B \times H \times r \times d_h}$
   - $V_{\text{sketch}} = \text{einsum}("b h n d, n r \to b h r d", V, \Omega) \in \mathbb{R}^{B \times H \times r \times d_h}$
5. **Reduced Attention Score & Softmax Aggregation**:
   - $S_{\text{sketch}} = \text{matmul}(Q, K_{\text{sketch}}^\top) / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times r}$
   - Mask padding positions: $\tilde{S} = \text{masked\_fill}(S_{\text{sketch}}, \neg M, -10^9)$
   - Probability maps: $P = \text{Dropout}(\text{Softmax}(\tilde{S}, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times r}$
   - Subspace context aggregation: $C_{\text{head}} = P V_{\text{sketch}} \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Output Projection**:
   - $O = \text{Reshape}(C_{\text{head}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Dense Linear Projections ($Q, K, V, O$): $8 B N D^2$ FLOPs
  - Random Sketching Contractions ($K \Omega, V \Omega$): $4 B H N r d_h = 4 B N r D$ FLOPs
  - Sketched Score Evaluation ($Q K_{\text{sketch}}^\top$): $2 B H N r d_h = 2 B N r D$ FLOPs
  - Sketched Value Aggregation ($P V_{\text{sketch}}$): $2 B N r D$ FLOPs
  - Total Module FLOPs: $8 B N D^2 + 8 B N r D$ (Strictly $\mathcal{O}(N)$ linear sequence scaling)
- **MAC Operations**:
  - Projection Matmuls: $4 B N D^2$ MACs
  - Subspace Contractions & Attention: $4 B N r D$ MACs
- **Memory Footprint & KV Cache**:
  - Full Attention Map Memory: $[B, H, N, N] \to [B, H, N, r]$ (Memory Reduction Ratio $= \frac{N}{r}\times$).
  - KV Cache Storage: $2 \times B \times H \times r \times d_h = 2 B r D$ elements.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Stochastic Output Variance**:
  Generating random Gaussian sketching matrix $\Omega$ dynamically in every forward pass creates non-deterministic attention outputs during evaluation, degrading output reproducibility unless seeds are fixed.
- **Padding Information Leakage**:
  Multiplication $K_{\text{sketch}} = K \Omega$ contracts sequence dimension $N$ before zero-masking padding positions. Unmasked padding tokens leak into valid subspace representations, introducing noise into $K_{\text{sketch}}$.
- **Isotropic Gaussian Sketching Suboptimality**:
  Using isotropic random sketching matrix $\Omega \sim \mathcal{N}(0, 1/N)$ treats all sequence positions uniformly, ignoring token salience or non-uniform sequence density.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Mask-Aware Pre-Filtered Sketch Contraction**:
  Filter out invalid padding tokens prior to sketch contraction: $\tilde{K} = K \odot M_{\text{valid}}$, $\tilde{V} = V \odot M_{\text{valid}}$. This prevents padding token energy from leaking into reduced subspace basis $K_{\text{sketch}}$.
- **Persistent Buffer Sketching with Orthogonal Initialization**:
  Register sketching matrix $\Omega$ as a persistent module buffer initialized via Stiefel-manifold orthogonal decomposition (QR decomposition) to eliminate forward-pass random generation overhead and reduce projection variance.
- **Fused FMA Triton Kernel for Subspace Contractions**:
  Implement a fused Triton kernel computing $K \Omega$ contraction, $Q K_{\text{sketch}}^\top$ matmul, softmax, and $P V_{\text{sketch}}$ aggregation in a single CUDA tile pass without instantiating $[B, H, N, r]$ intermediate score tensors in GPU DRAM.