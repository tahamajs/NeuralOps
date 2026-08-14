# Unified Rotary Position Embedding (UnifiedRoPE): Universal Cross-Branch Positional Encoding

## 1. Abstract & Theoretical Motivation

Rotary Position Embedding (RoPE), proposed by Su et al. (2024), embeds relative position geometry by applying complex rotational transformations directly to feature representations. In hybrid Transformer architectures (e.g., combining Scaled Dot-Product Attention with State-Space Models or Recurrent Linear Units), maintaining consistent relative position semantics across distinct structural branches is essential. 

UnifiedRoPE (`UnifiedRoPE`) implements a universal, non-parametric rotational position embedding module. By precomputing inverse frequency rotation buffers and applying a unified complex plane rotation across both attention and state-space branches, UnifiedRoPE guarantees cross-branch positional alignment. This eliminates phase mismatches between contextual routing mechanisms while supporting dynamic length extension without fine-tuning.

---

## 2. Mathematical Derivations & Formulations

Let $x \in \mathbb{R}^{B \times L \times D}$ denote the input activation tensor, where $B$ is batch size, $L$ is sequence length, and $D$ is feature dimension (where $D$ is an even integer). Let $\theta_0 = 10000$ be the base frequency scalar.

### Inverse Frequency Precomputation:
The inverse frequency spectrum $\boldsymbol{\theta} \in \mathbb{R}^{D/2}$ is computed over half the feature channels:

\[
\theta_i = \theta_0^{-\frac{2i}{D}} = \frac{1}{\theta_0^{\frac{2i}{D}}}, \quad i \in \left\{0, 1, \dots, \frac{D}{2} - 1\right\}
\]

For sequence positions $m \in \{0, 1, \dots, L_{\max}-1\}$, the outer product produces the phase matrix $\mathbf{\Theta} \in \mathbb{R}^{L_{\max} \times (D/2)}$:

\[
\mathbf{\Theta}_{m, i} = m \cdot \theta_i
\]

The phase matrix is concatenated along the feature dimension to form the full embedding tensor $\mathbf{E} \in \mathbb{R}^{L_{\max} \times D}$:

\[
\mathbf{E}_{m, :} = [\mathbf{\Theta}_{m, :}, \, \mathbf{\Theta}_{m, :}] \in \mathbb{R}^D
\]

Precomputed cosine and sine rotation buffers are registered:

\[
\mathbf{C} = \cos(\mathbf{E}) \in \mathbb{R}^{L_{\max} \times D}, \quad \mathbf{S} = \sin(\mathbf{E}) \in \mathbb{R}^{L_{\max} \times D}
\]

### Complex Orthogonal Rotation Formulation:
To apply 2D planar rotations across vector pairs $[x_k, x_{k + D/2}]$, the vector half-rotation operator $\mathcal{R}(x)$ is defined as:

\[
\mathcal{R}(x) = [-x_{:, :, D/2:}, \, x_{:, :, :D/2}] \in \mathbb{R}^{B \times L \times D}
\]

The rotated activation tensor $x_{\text{rot}} \in \mathbb{R}^{B \times L \times D}$ is computed as an elementwise linear combination:

\[
x_{\text{rot}} = x \odot \mathbf{C}_{:L} + \mathcal{R}(x) \odot \mathbf{S}_{:L}
\]

Explicitly, for feature pair $(x_k, x_{k + D/2})$ at sequence position $m$:

\[
\begin{bmatrix} (x_{\text{rot}})_k \\ (x_{\text{rot}})_{k + D/2} \end{bmatrix} = \begin{bmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{bmatrix} \begin{bmatrix} x_k \\ x_{k + D/2} \end{bmatrix}
\]

### Relative Inner-Product Preservation:
For query vector $q_i$ at position $i$ and key vector $k_j$ at position $j$, the inner product satisfies:

\[
\langle \text{RoPE}(q_i, i), \text{RoPE}(k_j, j) \rangle = \text{Re} \left( \sum_{m=0}^{D/2 - 1} (q_i)_m (k_j)_m^* e^{i (i - j) \theta_m} \right) = f(q_i, k_j, i - j)
\]

This proves that inner-product similarity depends exclusively on the relative distance $\Delta = i - j$, satisfying translational invariance across sequence positions.

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: x [B, L, D]
  │
  ├──> Check if L > max_seq_len ──> IF True: Rebuild Cache _build_cache(L)
  │
  ├──> Slice cos_cached[:L] ──> cos: [L, D] ──> view [1, L, D]
  ├──> Slice sin_cached[:L] ──> sin: [L, D] ──> view [1, L, D]
  │
  ├──> _rotate_half(x):
  │      x1 = x[..., : D//2], x2 = x[..., D//2 :]
  │      concat(-x2, x1, dim=-1) ──> x_half: [B, L, D]
  │
  └──> Elementwise Multiplication & Addition:
         x_rot = (x * cos) + (x_half * sin) ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Operational & Computational Workload (FLOPs & MACs)

Let $B$ be batch size, $L$ sequence length, $D$ feature dimension, and $L_{\max}$ maximum cached sequence length.

1. **Precomputation Cache Construction (One-time or Dynamic Cache Extension)**:
   - Outer Product ($t \otimes \theta_i$): $L_{\max} \cdot \frac{D}{2}$ FLOPs.
   - Trigonometric Evaluations ($\cos, \sin$): $2 \times L_{\max} \cdot D$ FLOPs.
   - Space: $2 \times L_{\max} \times D \times \text{bytes}$ (Persistent GPU buffers).

2. **Forward Rotation Computations**:
   - Half-rotation vector negation and concatenation: Zero-copy slicing / reshape.
   - Broadcast Cosine Multiplication ($x \odot \mathbf{C}$): $B \cdot L \cdot D$ FLOPs.
   - Broadcast Sine Multiplication ($\mathcal{R}(x) \odot \mathbf{S}$): $B \cdot L \cdot D$ FLOPs.
   - Vector Sum Addition: $B \cdot L \cdot D$ FLOPs.

### Complexity Totals:
- **Total FLOPs per Forward Pass**: $3 B L D$ FLOPs.
- **Total MACs**: $0$ (No matrix multiplication operations).
- **Parametric Footprint**: $0$ trainable parameters (Non-parametric buffer module).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Memory Bandwidth Bound Elementwise Execution
Because RoPE consists purely of elementwise additions and multiplications ($3 B L D$ FLOPs on $2 B L D$ read/write bytes), its arithmetic intensity is extremely low ($\approx 1.5 \text{ FLOP/byte}$). Execution is strictly **memory bandwidth-bound**, incurring memory allocation overhead in naive PyTorch implementations.

### High-Frequency Phase Saturation at Long Sequences
When sequence length $L$ substantially exceeds precomputed frequencies ($L \gg \frac{1}{\theta_{D/2}}$), high-frequency bands rotate rapidly over $2\pi$ boundaries. This introduces high-frequency phase noise, degrading long-context attention accuracy unless frequency scaling (e.g., YaRN or Linear Scaling) is integrated.

```
High-Frequency Saturation Phase Degradation:
Phase Angle = m * θ_i  (Mod 2π)  ──>  Rapid Rotation Instability for L >> 1/θ_min
```

### Channel Splitting Memory Copy Overhead
Computing `_rotate_half(x)` using `torch.cat((-x2, x1), dim=-1)` triggers memory reallocation and tensor copying in GPU DRAM, increasing runtime latency compared to strided pointer access.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Elementwise RoPE Triton Kernel Architecture
To maximize hardware memory bus bandwidth, Triton fuses RoPE rotation, channel splitting, cosine/sine slicing, and elementwise addition into a single CUDA kernel block.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Input x [B, L, D]                                      │
      │ Cosine Buffer cos_cached [L_max, D]                    │
      │ Sine Buffer sin_cached [L_max, D]                      │
      └───────────────────────────┬────────────────────────────┘
                                  │ Coalesced 128-bit Vector Load
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Load Vector Pairs: x_left = x[i], x_right = x[i + D/2]        │
 │ 2. Load Frequencies: c_i = cos[l, i], s_i = sin[l, i]            │
 │ 3. Execute Vector In-Register Rotation:                          │
 │    rot_left  = x_left  * c_i - x_right * s_i                     │
 │    rot_right = x_right * c_i + x_left  * s_i                     │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Direct Coalesced Write
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Rotated Output x_rot [B, L, D]                         │
      └────────────────────────────────────────────────────────┘
```

### Key Triton Implementation Guidelines:
1. **In-Register Interleaved Rotation**:
   Instead of concatenating split half-tensors in DRAM, process pairs $(x_i, x_{i + D/2})$ directly in SIMD vector registers, eliminating memory allocations.
2. **Strided Pointer Arithmetic**:
   Use strided pointer offsets to read `cos` and `sin` directly from precomputed HBM buffers without slicing or copying tensor slices.
3. **Fused Attention Pre-Rotation**:
   Fuse `UnifiedRoPE` directly into the input block loader of FlashAttention kernels, reading $x$, applying rotational shifts in registers, and passing rotated queries and keys directly to SIMD matrix multiplier units.