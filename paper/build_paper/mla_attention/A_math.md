<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Multi-Head Latent Attention (MLA) Formulation

Multi-Head Latent Attention introduces low-rank latent compression to alleviate the memory bandwidth bottlenecks of the key-value (KV) cache in autoregressive generation. Let $X \in \mathbb{R}^{B \times N \times D}$ denote the input activation sequence, where $B$ is batch size, $N$ is sequence length, and $D$ is model hidden dimension. Let $H$ be the number of attention heads and $d_h = D / H$ the per-head dimension.

### A.1.1 Latent Down-Projection & KV Cache Compression

Instead of caching full multi-head keys and values of dimension $2 H d_h$ per token, MLA projects inputs into a compressed latent vector $\mathbf{c}_{KV} \in \mathbb{R}^{B \times N \times d_c}$:

$$
\mathbf{c}_{KV} = X W_{DKV}
\tag{A.1}
$$

where $W_{DKV} \in \mathbb{R}^{D \times d_c}$ and the compression dimension $d_c \ll 2 H d_h$. The cached state at inference position $t$ consists exclusively of $\mathbf{c}_{KV}^{(t)}$ alongside a decoupled rotary key $\mathbf{k}_R^{(t)}$.

### A.1.2 Up-Projection and Attention Score Computation

Keys and values are reconstructed via linear up-projections $W_{UK} \in \mathbb{R}^{d_c \times D}$ and $W_{UV} \in \mathbb{R}^{d_c \times D}$:

$$
K = \mathbf{c}_{KV} W_{UK}, \quad V = \mathbf{c}_{KV} W_{UV}
\tag{A.2}
$$

For query heads $Q_h = X W_{Q,h} \in \mathbb{R}^{B \times N \times d_h}$, the raw pre-softmax attention logit between query position $i$ and key position $j$ evaluates to:

$$
S_{h, ij} = \frac{Q_{h, i} K_{h, j}^\top}{\sqrt{d_h}} = \frac{Q_{h, i} ( \mathbf{c}_{KV, j} W_{UK, h} )^\top}{\sqrt{d_h}}
\tag{A.3}
$$

### A.1.3 Mathematical Absorption for Inference Acceleration

During generation decoding, rather than up-projecting $\mathbf{c}_{KV, j}$ into high-dimensional keys, the associative property of matrix multiplication enables absorbing $W_{UK, h}$ directly into the query:

$$
S_{h, ij} = \frac{(Q_{h, i} W_{UK, h}^\top) \, \mathbf{c}_{KV, j}^\top}{\sqrt{d_h}}
\tag{A.4}
$$

Letting $\tilde{Q}_{h, i} = Q_{h, i} W_{UK, h}^\top \in \mathbb{R}^{1 \times d_c}$, the attention score is computed directly with the compressed cache $\mathbf{c}_{KV} \in \mathbb{R}^{j \times d_c}$. Similarly, for the output context aggregation:

$$
O_{h, i} = \sum_{j=1}^i A_{h, ij} V_{h, j} = \left(\sum_{j=1}^i A_{h, ij} \mathbf{c}_{KV, j}\right) W_{UV, h}
\tag{A.5}
$$

Thus, generation latency and memory bandwidth scale strictly with $d_c$ rather than $2 H d_h$.

## A.2 KV Cache Memory Reduction Ratio

At sequence length $L$, conventional Multi-Head Attention requires caching:

$$
\text{Memory}_{\text{MHA}} = 2 \cdot L \cdot H \cdot d_h \cdot b = 2 L D b \quad \text{bytes}
\tag{A.6}
$$

where $b$ is the bytes-per-parameter precision (e.g., $b=2$ for FP16). Under MLA with latent dimension $d_c$ and decoupled rotary dimension $d_R$:

$$
\text{Memory}_{\text{MLA}} = L \cdot (d_c + d_R) \cdot b \quad \text{bytes}
\tag{A.7}
$$

The theoretical compression factor $\rho$ is:

$$
\rho = \frac{2 D}{d_c + d_R}
\tag{A.8}
$$

For $D=768$, $d_c=128$, and $d_R=64$, $\rho = \frac{1536}{192} = 8.0\times$ reduction in per-token KV memory overhead.
