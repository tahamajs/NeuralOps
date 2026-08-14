# SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Rectified Attention

## 1. Abstract & Theoretical Motivation
SPECTRA-v2 (`SpectraV2Attention`, registered as `spectra_v2`) establishes a hybrid sparse-attention architecture unifying local intra-block full attention, relative-position-biased inter-block top-$k$ routing, per-head adaptive attention spans, and periodic global Random Fourier Feature (RFF) rectification. Quadratic self-attention scaling $\mathcal{O}(N^2)$ creates computational bottlenecks in long-context Transformers. SPECTRA-v2 addresses this by partitioning sequences into local blocks of size $B_s$, evaluating exact intra-block attention, and selecting the top-$k_{\text{inter}}$ candidates per query from adjacent blocks under relative position distance biases. To prevent context fragmentation across deep layers, every $R_{\text{rect}}$-th layer bypasses block sparsity entirely, deploying an $\mathcal{O}(N \cdot d_r)$ Random Fourier Feature kernel ("AKS") for global context refresh. This hybrid design delivers sub-quadratic $\mathcal{O}(N \cdot (B_s + k_{\text{inter}}) \cdot d_h)$ complexity while maintaining long-range context fidelity.

## 2. Mathematical Derivations & Formulations
Given token activation sequence $X \in \mathbb{R}^{B \times N \times D}$, linear projections derive multi-head query, key, and value tensors $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ ($d_h = D / H$).

### 1. Adaptive Per-Head Span Gating
For sequence length $N$, a soft distance-decay mask $M_{\text{span}} \in \mathbb{R}^{H \times N \times N}$ is evaluated using learnable head-wise span parameters $s_h \in \mathbb{R}$:
$$M_{\text{span}}[h, i, j] = \ln \left( \sigma\left( 5.0 \cdot (s_h - |i - j|) \right) + 10^{-9} \right)$$
where $\sigma(\cdot)$ is the sigmoid function.

### 2. Block-Sparse Intra & Inter-Block Top-$k$ Routing
The sequence is partitioned into blocks of size $B_s = 64$. For query block $b \in \{1, \dots, \lceil N / B_s \rceil\}$ covering sequence interval $[s, e)$:
- **Intra-Block Attention**: $Q_b = Q[:, :, s:e, :]$, $K_{\text{intra}} = K[:, :, s:e, :]$, $V_{\text{intra}} = V[:, :, s:e, :]$.
- **Inter-Block Candidate Gathering**: Inter-block keys $K_{\text{inter}}$ and values $V_{\text{inter}}$ are gathered from adjacent blocks $b-1$ and $b+1$.
- **Relative Position Bias**: For query position $p_q \in [s, e)$ and candidate key position $p_k$, relative distance is clipped to $\delta = \text{clamp}(p_q - p_k, -d_{\text{max}}, d_{\text{max}}) + d_{\text{max}}$. Inter-block scores incorporate relative positional bias parameters $B_{\text{rel}} \in \mathbb{R}^{H \times (2 d_{\text{max}} + 1)}$:
  $$\text{Score}_{\text{inter}} = \frac{Q_b K_{\text{inter}}^\top}{\sqrt{d_h}} + B_{\text{rel}}[:, \delta]$$
- **Top-$k$ Candidate Selection**: For each query token, extract top $k_{\text{inter}}$ keys/values:
  $$\mathcal{K}_{\text{sel}} = \text{TopK}(\text{Score}_{\text{inter}}, k_{\text{inter}}), \qquad \mathcal{V}_{\text{sel}} = \text{Gather}(V_{\text{inter}}, \mathcal{K}_{\text{sel}})$$
- **Combined Block Context Evaluation**:
  $$S_{\text{block}} = \frac{Q_b [K_{\text{intra}}, \mathcal{K}_{\text{sel}}]^\top}{\sqrt{d_h}} + M_{\text{span}}[:, s:e, s:e]$$
  $$A_{\text{block}} = \text{Dropout}(\text{Softmax}(S_{\text{block}}, \text{dim}=-1))$$
  $$O_b = A_{\text{block}} [V_{\text{intra}}, \mathcal{V}_{\text{sel}}] \in \mathbb{R}^{B \times H \times B_s \times d_h}$$

### 3. Periodic Global RFF Rectification (Every $R_{\text{rect}}$ Layers)
At layers where $l \pmod{R_{\text{rect}}} == 0$, block-sparse attention is substituted by a Random Fourier Feature (RFF) linear attention pass. Feature maps $\phi(x) \in \mathbb{R}^{B \times H \times N \times 2 d_r}$ project queries and keys using random weight matrix $W_{\text{rff}} \in \mathbb{R}^{d_h \times d_r}$:
$$\phi(x) = \frac{1}{d_r^{1/4}} \begin{bmatrix} \cos(x W_{\text{rff}}) & \sin(x W_{\text{rff}}) \end{bmatrix}$$
$$O_{\text{rect}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + 10^{-8}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

Final multi-head outputs are concatenated and projected via output weight $W_O$:
$$O = \text{Reshape}(O_{\text{context}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor execution path across SPECTRA-v2 is traced below:

1. **Input Representation**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dense Multi-Head Projections**:
   - $Q = \text{Reshape}(X W_Q + b_Q) \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $K = \text{Reshape}(X W_K + b_K) \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $V = \text{Reshape}(X W_V + b_V) \in \mathbb{R}^{B \times H \times N \times d_h}$
3. **Validity Mask Ingestion**:
   - Normalize arbitrary 2D/3D/4D mask to $T_{\text{valid}} \in \{0, 1\}^{B \times N}$
4. **Layer Branch Dispatch**:
   - **If Layer is Rectification Layer ($l \pmod{R_{\text{rect}}} == 0$)**:
     - Evaluate $\phi_Q, \phi_K \in \mathbb{R}^{B \times H \times N \times 2 d_r}$ using RFF kernel
     - $K V_{\text{rff}} = \phi_K^\top V \in \mathbb{R}^{B \times H \times 2 d_r \times d_h}$, $Z_{\text{rff}} = \phi_K^\top \mathbf{1} \in \mathbb{R}^{B \times H \times 2 d_r \times 1}$
     - $C_{\text{rect}} = \frac{\phi_Q K V_{\text{rff}}}{\phi_Q Z_{\text{rff}} + 10^{-8}} \in \mathbb{R}^{B \times H \times N \times d_h}$
   - **Else (Block-Sparse Layer)**:
     - Iterate across blocks $b = 1 \dots \lceil N / B_s \rceil$:
       - Slice $Q_b \in [B, H, B_s, d_h]$, $K_{\text{intra}}, V_{\text{intra}} \in [B, H, B_s, d_h]$
       - Gather adjacent candidates $K_{\text{inter}}, V_{\text{inter}}$
       - Add relative position bias $B_{\text{rel}}[\delta]$
       - Top-$k$ selection $\implies \mathcal{K}_{\text{sel}}, \mathcal{V}_{\text{sel}} \in [B, H, B_s \cdot k_{\text{inter}}, d_h]$
       - Evaluate $S_{\text{block}} = Q_b [K_{\text{intra}}, \mathcal{K}_{\text{sel}}]^\top / \sqrt{d_h} + M_{\text{span}}$
       - Softmax and value aggregate $\implies O_b \in [B, H, B_s, d_h]$
     - Concatenate block outputs $\implies C_{\text{sparse}} \in \mathbb{R}^{B \times H \times N \times d_h}$
5. **Mask Application & Output Projection**:
   - $C_{\text{final}} = C_{\text{context}} \cdot T_{\text{valid}}$
   - $O = \text{Reshape}(C_{\text{final}}) W_O + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Dense Linear Projections ($Q, K, V, O$): $8 B N D^2$ FLOPs
  - Block-Sparse Attention ($N / B_s$ blocks): $2 B H N (B_s + k_{\text{inter}}) d_h = 2 B N (B_s + k_{\text{inter}}) D$ FLOPs
  - Top-$k$ Inter-Block Sorting: $\mathcal{O}(B H N \cdot 2 B_s \log k_{\text{inter}})$ FLOPs
  - Periodic RFF Rectification Layer: $4 B N D d_r$ FLOPs
  - Total Module FLOPs (Block-Sparse Layer): $8 B N D^2 + 2 B N (B_s + k_{\text{inter}}) D + \mathcal{O}(N \log k_{\text{inter}})$
- **MAC Operations**:
  - Linear Projections: $4 B N D^2$ MACs
  - Block-Sparse Attention: $B N (B_s + k_{\text{inter}}) D$ MACs
- **Memory Footprint & KV Cache**:
  - Block Attention Map Storage: $[B, H, B_s, B_s + k_{\text{inter}}]$ per block vs $[B, H, N, N]$ (Memory Reduction Ratio $= \frac{N}{B_s + k_{\text{inter}}}\times$).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Adjacent Block Horizon Isolation**:
  Inter-block candidate gathering only inspects immediately adjacent blocks ($b-1$ and $b+1$). Tokens requiring long-range interactions across non-adjacent blocks rely entirely on periodic RFF layers, leading to context loss in non-rectifying layers.
- **Top-$k$ Gather GPU Memory Fragmentation**:
  Extracting top-$k$ indices per query position using `torch.gather` on 5D expanded tensors creates uncoalesced DRAM memory accesses, degrading GPU memory throughput.
- **Static Span Parameter Underflow**:
  When adaptive span parameters $s_h$ shrink during optimization ($s_h < 1.0$), log-mask $M_{\text{span}}$ severely attenuates intra-block scores, causing local attention collapse.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Block-Sparse Inter-Block Triton Kernel**:
  Construct a dedicated Triton block-sparse kernel that computes intra-block GEMM and inter-block top-$k$ selection directly within GPU L1 cache/SRAM tiles. This eliminates 5D tensor expansions and global DRAM gathers.
- **Orthogonal RFF Projection Initialization**:
  Replace random Gaussian matrix $W_{\text{rff}} \sim \mathcal{N}(0, 0.02^2)$ with orthogonal QR decomposition (FAVOR+ / Orthogonal Random Features). This reduces kernel variance and guarantees exact softmax approximation.
- **FlashAttention Integration for Rectification Layers**:
  Pass RFF feature projections $\phi_Q, \phi_K$ directly into fused linear FlashAttention kernels to evaluate linear memory state updates without allocating $[B, H, N, 2 d_r]$ intermediate feature tensors.