# Performer Attention: Fast Attention via Positive Orthogonal Random Features (FAVOR+)

## 1. Abstract & Theoretical Motivation
Standard softmax self-attention computes explicit query-key dot-product affinity matrices \(\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{N \times N}\), demanding \(\mathcal{O}(N^2)\) computational complexity and GPU memory. Performer (Choromanski et al., 2021) overcomes this limitation by kernelizing the softmax function through positive orthogonal random feature maps (FAVOR+). 

By expressing the unnormalized softmax kernel as an inner product of finite-dimensional random feature vectors \(\mathbf{K}_{\operatorname{softmax}}(\mathbf{q}, \mathbf{k}) = \exp\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d}}\right) \approx \langle \phi(\mathbf{q}), \phi(\mathbf{k}) \rangle\), the sequence computation can be regrouped via matrix multiplication associativity. This decouples sequence length \(N\) from feature dimension \(m\), yielding strictly linear \(\mathcal{O}(N m d)\) time and memory complexity while maintaining strictly positive kernel values for numerical stability.

---

## 2. Mathematical Derivations & Formulations

Given queries \(\mathbf{Q} \in \mathbb{R}^{B \times H \times N \times d}\), keys \(\mathbf{K} \in \mathbb{R}^{B \times H \times N \times d}\), and values \(\mathbf{V} \in \mathbb{R}^{B \times H \times N \times d}\), we apply a scaling factor of \(d^{-1/4}\) to queries and keys:

\[
\widetilde{\mathbf{Q}} = d^{-1/4} \mathbf{Q}, \quad \widetilde{\mathbf{K}} = d^{-1/4} \mathbf{K}
\]

The FAVOR+ positive random feature map \(\phi: \mathbb{R}^d \to \mathbb{R}_+^m\) maps each \(d\)-dimensional input vector \(\mathbf{x}\) into an \(m\)-dimensional positive space using an orthogonal projection matrix \(\boldsymbol{\Omega} \in \mathbb{R}^{m \times d}\):

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \exp\left( \mathbf{x} \boldsymbol{\Omega}^\top - \frac{\|\mathbf{x}\|_2^2}{2} \mathbf{1}_m^\top \right)
\]

To guarantee unbiased kernel estimation with minimal variance, \(\boldsymbol{\Omega}\) is constructed by stacking \(m/d\) orthogonal matrices obtained from QR decompositions of isotropic Gaussian matrices \(\mathbf{G} \sim \mathcal{N}(0, \mathbf{I}_{d \times d})\), scaled by row norms \(\|\mathbf{g}_i\|_2\).

For numerical stability against exponent overflow/underflow, maximum logit subtraction is integrated directly into the feature computation:

\[
\mathbf{z}(\mathbf{x}) = \mathbf{x} \boldsymbol{\Omega}^\top - \frac{\|\mathbf{x}\|_2^2}{2} \mathbf{1}_m^\top, \quad \mathbf{m}_{\text{max}} = \max_m (\mathbf{z}(\mathbf{x}))
\]
\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{m}} \exp\left( \mathbf{z}(\mathbf{x}) - \mathbf{m}_{\text{max}} \right)
\]

Applying feature transform \(\phi(\cdot)\) to query and key tensors yields \(\mathbf{Q}' = \phi(\widetilde{\mathbf{Q}}) \in \mathbb{R}_+^{B \times H \times N \times m}\) and \(\mathbf{K}' = \phi(\widetilde{\mathbf{K}}) \odot \mathbf{M}_{\text{valid}} \in \mathbb{R}_+^{B \times H \times N \times m}\).

Using matrix associativity, attention computation is reformulated without instantiating the \(N \times N\) score matrix:

\[
\mathbf{K}\mathbf{V} = (\mathbf{K}')^\top \mathbf{V} \in \mathbb{R}^{B \times H \times m \times d}
\]
\[
\mathbf{k}_{\text{sum}} = \sum_{j=1}^N \mathbf{K}'_{:, :, j, :} \in \mathbb{R}^{B \times H \times 1 \times m}
\]
\[
\mathbf{D} = \mathbf{Q}' \mathbf{k}_{\text{sum}}^\top + \epsilon \in \mathbb{R}^{B \times H \times N \times 1}
\]
\[
\mathbf{O} = \left( \frac{\mathbf{Q}' (\mathbf{K}\mathbf{V})}{\mathbf{D}} \right) \odot \mathbf{M}_{\text{valid}} \in \mathbb{R}^{B \times H \times N \times d}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Linear Projections (W_Q, W_K, W_V)
  │    └──> Q, K, V [B, H, N, d]
  │
  ├──> Scale Queries & Keys: Q, K *= d^(-1/4)
  │
  ├──> FAVOR+ Feature Maps:
  │    ├──> Q' = phi(Q) [B, H, N, m]
  │    └──> K' = phi(K) * Valid_Mask [B, H, N, m]
  │
  ├──> Associative State Accumulation:
  │    ├──> KV = (K')^T @ V ──────────────> [B, H, m, d]
  │    └──> K_sum = sum(K', dim=-2) ──────> [B, H, 1, m]
  │
  ├──> Linear Readout & Normalization:
  │    ├──> Num = Q' @ KV ────────────────> [B, H, N, d]
  │    ├──> Den = Q' @ (K_sum)^T + eps ───> [B, H, N, 1]
  │    └──> Context = Num / Den ──────────> [B, H, N, d]
  │
  └──> Query Masking & Projection:
       Context = Context * Valid_Mask ───> [B, N, D]
       Y = Context @ W_O ────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Input Projections | `X: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 2 | Scale Factor Multiply | `Q, K: [B, H, N, d]` | `Q, K: [B, H, N, d]` |
| 3 | Query Feature Map | `Q: [B, H, N, d]`, `omega: [m, d]` | `Q': [B, H, N, m]` |
| 4 | Key Feature Map | `K: [B, H, N, d]`, `omega: [m, d]` | `K': [B, H, N, m]` |
| 5 | Token Masking | `K': [B, H, N, m]`, `Mask: [B, 1, N, 1]` | `K': [B, H, N, m]` |
| 6 | State Accumulation KV | `(K')^T: [B, H, m, N]`, `V: [B, H, N, d]` | `KV: [B, H, m, d]` |
| 7 | Key Sum Reduction | `K': [B, H, N, m]` | `K_sum: [B, H, 1, m]` |
| 8 | Unnormalized Output | `Q': [B, H, N, m]`, `KV: [B, H, m, d]` | `Num: [B, H, N, d]` |
| 9 | Denominator Calculation | `Q': [B, H, N, m]`, `(K_sum)^T: [B, H, m, 1]` | `Den: [B, H, N, 1]` |
| 10 | Element-wise Normalization | `Num: [B, H, N, d]`, `Den: [B, H, N, 1]` | `Context: [B, H, N, d]` |
| 11 | Output Linear Projection | `Context: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections (\(W_Q, W_K, W_V, W_O\))**: \(8 B N D^2\) FLOPs.
2. **Feature Mapping (\(\phi(Q), \phi(K)\))**:
   - Matrix product with \(\boldsymbol{\Omega}\): \(2 \times (2 B H N m d) = 4 B N m D\) FLOPs.
   - Exponential and normalization ops: \(\mathcal{O}(B H N m)\).
3. **KV State Accumulation (\((\mathbf{K}')^\top \mathbf{V}\))**: \(2 B H m N d = 2 B N m D\) FLOPs.
4. **Context Readout (\(\mathbf{Q}' (\mathbf{K}\mathbf{V})\))**: \(2 B H N m d = 2 B N m D\) FLOPs.
5. **Denominator Projection (\(\mathbf{Q}' \mathbf{k}_{\text{sum}}^\top\))**: \(2 B H N m\) FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 8 B N m D + 2 B N H m
\]

For typical settings where \(m \ll N\), computing requirements transition from standard \(\mathcal{O}(N^2)\) to \(\mathcal{O}(N)\).

### Memory Footprint & Tensor Store
- **Parameters**: \(4 D^2\) floats. Buffer \(\boldsymbol{\Omega} \in \mathbb{R}^{m \times d}\) stores \(m d\) floats.
- **Activation Memory**:
  - Transformed Feature Maps \(\mathbf{Q}', \mathbf{K}'\): \(2 B H N m\) floats.
  - State Tensor \(\mathbf{K}\mathbf{V}\): \(B H m d\) floats.
  - Denominator Vector \(\mathbf{D}\): \(B H N\) floats.
  - Peak Memory: \(\mathcal{O}(B N D + B H N m + B H m d)\).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Variance Bottleneck in Random Features**: When sequence length \(N\) is short or hidden dimension \(d\) is large, Monte Carlo estimation error in random features leads to noisy attention distributions. If feature count \(m\) is too small (\(m < d\)), estimation variance degrades model accuracy.
2. **Exponential Recurrent Instability**: Because \(\phi(\mathbf{x})\) uses exponential functions without hard bounds, large dot products \(\mathbf{x} \boldsymbol{\Omega}^\top\) can cause floating-point overflow during FP16 dynamic range execution despite softmax scaling.
3. **Causal Masking Overhead**: In autoregressive generation, causal prefix summation \(\sum_{j=1}^i \phi(\mathbf{k}_j) \mathbf{v}_j^\top\) introduces sequential dependency, requiring prefix-sum associative scans that increase register pressure on GPUs.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
In naive PyTorch, materializing intermediate tensors \(\mathbf{Q}'\) and \(\mathbf{K}'\) requires writing \(2 \times B H N m\) elements to HBM, creating significant memory bandwidth overhead.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused Feature Extraction & State Accumulation**: Implement a custom Triton kernel where \(\mathbf{K}_j\) blocks of size `BLOCK_N` are loaded into SRAM, scaled, and feature mapped \(\phi(\mathbf{K}_j)\) on-the-fly inside GPU registers. 
2. **SRAM Block Reduction**: Perform key-value outer product accumulation \(\mathbf{K}\mathbf{V} = \sum_j \phi(\mathbf{k}_j)^\top \mathbf{v}_j\) entirely inside GPU L1/SRAM registers using atomic accumulation instructions:

```python
# Triton Pseudocode Snippet for KV State Accumulation Block
@triton.jit
def _performer_kv_accumulate_kernel(
    K_ptr, V_ptr, Omega_ptr, KV_ptr,
    N, D, M,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    # Load Key block [BLOCK_N, D]
    k = tl.load(K_ptr + offsets_n[:, None] * D + offsets_d[None, :])
    # Load Omega block [BLOCK_M, D]
    omega = tl.load(Omega_ptr + offsets_m[:, None] * D + offsets_d[None, :])
    # On-chip Feature Map: proj = k @ omega.T -> [BLOCK_N, BLOCK_M]
    proj = tl.dot(k, tl.trans(omega))
    k_prime = tl.exp(proj - norm_k)
    # Compute Outer Product KV_block = k_prime.T @ v -> accumulated into SRAM
    tl.atomic_add(KV_ptr + offsets_m[:, None] * D + offsets_d[None, :], tl.dot(tl.trans(k_prime), v))
```

3. **Shared Memory Reuse for Projection Matrix**: Keep \(\boldsymbol{\Omega}\) in GPU Constant Cache or Shared Memory (`smem`), enabling high arithmetic intensity during sequence streaming.