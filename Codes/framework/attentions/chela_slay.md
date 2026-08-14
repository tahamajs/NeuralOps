# CHELA-SLAY Attention: Spherical Linearized Attention with Yat-Kernel

## 1. Abstract & Theoretical Motivation
Traditional linear attention methods often suffer from severe gradient instability or capacity collapse due to unbounded exponential kernel feature maps or norm variations in high-dimensional key-query spaces. CHELA-SLAY Attention addresses these limitations by introducing Spherical Linearized Attention with the trigonometric Yat-Kernel.

CHELA-SLAY constrains query and key projection vectors onto the unit hypersphere $\mathbb{S}^{d-1}$ via Euclidean $\ell_2$ normalization. It then maps the normalized spherical representations through a trigonometric Random Fourier Feature (Yat-Kernel) mapping $\phi(\mathbf{x}) = F^{-1/4} [\cos(\mathbf{x} \boldsymbol{\Omega}) \,\|\, \sin(\mathbf{x} \boldsymbol{\Omega})]$. By replacing standard exponential kernels with bounded trigonometric functions on the hypersphere, CHELA-SLAY ensures strict numerical stability while maintaining linear time and space complexity $\mathcal{O}(N F d)$.

---

## 2. Mathematical Derivations & Formulations

Given input representations $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, multi-head query, key, and value representations are computed via linear projections:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times H \times N \times d}
\]

### 1. Spherical Hypersphere Normalization
Queries and keys are projected onto the unit hypersphere $\mathbb{S}^{d-1}$:

\[
\hat{\mathbf{Q}} = \frac{\mathbf{Q}}{\|\mathbf{Q}\|_2 + \epsilon} \in \mathbb{R}^{B \times H \times N \times d}, \quad \hat{\mathbf{K}} = \frac{\mathbf{K}}{\|\mathbf{K}\|_2 + \epsilon} \in \mathbb{R}^{B \times H \times N \times d}
\]

### 2. Trigonometric Yat-Kernel Feature Mapping
A static Gaussian random projection matrix buffer $\boldsymbol{\Omega} \in \mathbb{R}^{d \times F}$ is registered, where $F$ is the target feature dimension. The Yat-Kernel mapping projects spherical vectors into a $2F$-dimensional feature space:

\[
\mathbf{Z}_{\mathbf{Q}} = \hat{\mathbf{Q}} \boldsymbol{\Omega} \in \mathbb{R}^{B \times H \times N \times F}, \quad \mathbf{Z}_{\mathbf{K}} = \hat{\mathbf{K}} \boldsymbol{\Omega} \in \mathbb{R}^{B \times H \times N \times F}
\]
\[
\phi(\hat{\mathbf{Q}}) = F^{-1/4} \left[ \cos(\mathbf{Z}_{\mathbf{Q}}) \;\|\; \sin(\mathbf{Z}_{\mathbf{Q}}) \right] \in \mathbb{R}^{B \times H \times N \times 2F}
\]
\[
\phi(\hat{\mathbf{K}}) = F^{-1/4} \left[ \cos(\mathbf{Z}_{\mathbf{K}}) \;\|\; \sin(\mathbf{Z}_{\mathbf{K}}) \right] \in \mathbb{R}^{B \times H \times N \times 2F}
\]

To enforce valid sequence masking, padding positions in key features and values are zeroed out:

\[
\widetilde{\phi}(\hat{\mathbf{K}}) = \phi(\hat{\mathbf{K}}) \odot \mathbf{M}_{\text{valid}}, \quad \widetilde{\mathbf{V}} = \mathbf{V} \odot \mathbf{M}_{\text{valid}}
\]

### 3. Linear Associative State Aggregation
Matrix multiplication associativity reformulates the attention computation without materializing $N \times N$ matrices:

\[
\mathbf{K}\mathbf{V} = \widetilde{\phi}(\hat{\mathbf{K}})^\top \widetilde{\mathbf{V}} = \sum_{j=1}^N \widetilde{\phi}(\hat{\mathbf{K}})_j^\top \widetilde{\mathbf{V}}_j \in \mathbb{R}^{B \times H \times 2F \times d}
\]
\[
\mathbf{k}_{\text{sum}} = \sum_{j=1}^N \widetilde{\phi}(\hat{\mathbf{K}})_j \in \mathbb{R}^{B \times H \times 1 \times 2F}
\]
\[
\mathbf{D} = \phi(\hat{\mathbf{Q}}) \mathbf{k}_{\text{sum}}^\top + \epsilon \in \mathbb{R}^{B \times H \times N \times 1}
\]
\[
\mathbf{O} = \frac{\phi(\hat{\mathbf{Q}}) (\mathbf{K}\mathbf{V})}{\mathbf{D}} \in \mathbb{R}^{B \times H \times N \times d}
\]

The multi-head contextual output is transposed and projected back:

\[
\mathbf{Y} = \operatorname{Concat}(\mathbf{O}_1, \dots, \mathbf{O}_H) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Linear Projections (W_Q, W_K, W_V) -> Q, K, V [B, H, N, d]
  │
  ├──> Spherical L2 Normalization:
  │    ├──> Q_norm = Q / ||Q||_2 ───────────────────────> [B, H, N, d]
  │    └──> K_norm = K / ||K||_2 ───────────────────────> [B, H, N, d]
  │
  ├──> Trigonometric Yat-Kernel Random Mapping:
  │    ├──> Z_q = Q_norm @ Omega ──> [B, H, N, F]
  │    ├──> Q_feat = F^(-1/4) * Cat([cos(Z_q), sin(Z_q)]) -> [B, H, N, 2F]
  │    └──> K_feat = F^(-1/4) * Cat([cos(Z_k), sin(Z_k)]) -> [B, H, N, 2F]
  │
  ├──> Apply Masking: K_feat = K_feat * Mask, V = V * Mask
  │
  ├──> Associative Linear State Aggregation:
  │    ├──> KV = (K_feat)^T @ V ────────────────────────> [B, H, 2F, d]
  │    └──> K_sum = sum(K_feat, dim=2) ─────────────────> [B, H, 1, 2F]
  │
  ├──> Context Readout & Normalization:
  │    ├──> Num = Q_feat @ KV ──────────────────────────> [B, H, N, d]
  │    ├──> Denom = Q_feat @ (K_sum)^T + eps ───────────> [B, H, N, 1]
  │    └──> Context = Num / Denom ──────────────────────> [B, H, N, d]
  │
  └──> Reshape & Output Projection:
       Y = Reshape(Context) @ W_O ──────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Projections ($W_Q, W_K, W_V$) | `X: [B, N, D]` | `Q, K, V: [B, N, D]` |
| 2 | Multi-Head Reshape | `Q, K, V: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 3 | Spherical Normalization | `Q, K: [B, H, N, d]` | `Q_norm, K_norm: [B, H, N, d]` |
| 4 | Random Matrix Product | `Q_norm: [B, H, N, d]`, `Omega: [d, F]` | `Z_q, Z_k: [B, H, N, F]` |
| 5 | Trigonometric Concatenation | `Z_q: [B, H, N, F]` | `Q_feat, K_feat: [B, H, N, 2F]` |
| 6 | Token Mask Multiplication | `K_feat: [B, H, N, 2F]`, `Mask` | `K_feat_masked: [B, H, N, 2F]` |
| 7 | Outer Product Accumulation | `K_feat^T: [B, H, 2F, N]`, `V: [B, H, N, d]` | `KV: [B, H, 2F, d]` |
| 8 | Sum Reduction Denominator | `K_feat: [B, H, N, 2F]` | `K_sum: [B, H, 1, 2F]` |
| 9 | Unnormalized Readout | `Q_feat: [B, H, N, 2F]`, `KV: [B, H, 2F, d]` | `Num: [B, H, N, d]` |
| 10 | Context Normalization | `Num: [B, H, N, d]`, `Denom: [B, H, N, 1]` | `Context: [B, H, N, d]` |
| 11 | Output Linear Projection | `Context: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections ($W_Q, W_K, W_V, W_O$)**: $8 B N D^2$ FLOPs.
2. **Spherical Normalization**: $2 \times (3 B H N d) = 6 B N D$ FLOPs.
3. **Random Fourier Projection ($\hat{\mathbf{Q}} \boldsymbol{\Omega}, \hat{\mathbf{K}} \boldsymbol{\Omega}$)**:
   - $2 \times (2 B H N F d) = 4 B N F D$ FLOPs.
4. **Trigonometric Evaluation ($\cos, \sin$)**: $\mathcal{O}(B H N F)$ operations.
5. **State Accumulation ($\mathbf{K}\mathbf{V} = \phi(\hat{\mathbf{K}})^\top \mathbf{V}$)**: $2 B H (2F) N d = 4 B N F D$ FLOPs.
6. **Context Readout ($\phi(\hat{\mathbf{Q}}) \mathbf{K}\mathbf{V}$)**: $2 B H N (2F) d = 4 B N F D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 12 B N F D + 6 B N D
\]

When $F \ll N$, computational runtime scales strictly linear with sequence length $N$.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2$ floats. Buffer $\boldsymbol{\Omega} \in \mathbb{R}^{d \times F}$ contains $d F$ parameters.
- **Activation Memory**:
  - Feature Maps ($\phi(\hat{\mathbf{Q}}), \phi(\hat{\mathbf{K}})$): $2 \times (2 B H N F) = 4 B H N F$ floats.
  - Recurrent State Matrix ($\mathbf{K}\mathbf{V}$): $2 B H F d = 2 B F D$ floats.
  - Denominator Vector ($\mathbf{D}$): $B H N$ floats.
  - Peak Activation Memory: $\mathcal{O}(B N D + B H N F + B F D)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Information Loss via Spherical Normalization**: Dividing queries and keys by their $\ell_2$ norm removes vector magnitude information, restricting the model from weighting tokens based on activation norms.
2. **Trigonometric Variance in Low Dimensions**: If feature dimension $F$ is set too low ($F < d$), trigonometric random Fourier feature approximation suffers from high Monte Carlo sampling variance, degrading attention precision.
3. **Static Random Matrix Buffer**: Matrix $\boldsymbol{\Omega}$ is an unlearned frozen buffer, preventing domain-specific adaptation of the Fourier feature space during fine-tuning.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
Standard PyTorch implementations materialize feature maps $\phi(\hat{\mathbf{Q}})$ and $\phi(\hat{\mathbf{K}})$ into HBM, generating $4 B H N F$ floats of memory traffic.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused Trigonometric Feature Mapping & State Accumulation**: Implement a Triton kernel where sequence blocks of keys are loaded into SRAM, normalized, mapped via trigonometric functions in registers, and accumulated into state $\mathbf{K}\mathbf{V}$ directly inside SRAM.

```python
# Triton Pseudocode for Fused CHELA-SLAY Kernel
@triton.jit
def _chela_slay_fused_kernel(
    K_ptr, V_ptr, Omega_ptr, KV_ptr,
    N, D, F: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Load Key block [BLOCK_N, d]
    k = tl.load(K_ptr + offsets_k)
    # L2 normalize inside registers
    k_norm = k / (tl.sqrt(tl.sum(k * k, axis=-1, keep_dims=True)) + 1e-8)
    
    # On-chip Random Projection & Trigonometric Mapping:
    proj = tl.dot(k_norm, Omega_ptr)
    k_feat = tl.cat([tl.cos(proj), tl.sin(proj)], axis=-1) * scale
    
    # Outer product accumulation KV_block = k_feat.T @ v inside SRAM registers
    tl.atomic_add(KV_ptr + offsets_kv, tl.dot(tl.trans(k_feat), v))
```

2. **Constant Shared Memory Placement for $\boldsymbol{\Omega}$**: Store the random projection matrix $\boldsymbol{\Omega}$ in GPU Shared Memory (`smem`), ensuring high arithmetic intensity during sequence streaming.