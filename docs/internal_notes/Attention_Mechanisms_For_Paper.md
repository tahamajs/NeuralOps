# Comprehensive Survey and Mathematical Taxonomy of Attention Mechanisms for Transformer Architectures

**Lead AI Research Scientist Reference Documentation & Formal Academic Compendium**

---

## Abstract

As Transformer and State-Space Architectures scale to multi-million token context windows and multi-modal regimes, standard Multi-Head Attention (MHA) with $\mathcal{O}(N^2)$ algorithmic complexity encounters severe memory, throughput, and latency bottlenecks. This technical compendium delivers an exhaustive, paper-ready mathematical formulation, theoretical complexity analysis, and architectural taxonomy for the entire suite of attention mechanisms, linear recurrent kernels, spectral matrix decompositions, state-space models (SSMs), and hybrid attention variants implemented within our research framework.

---

## Table of Contents

1. [Foundational Paradigms & Multi-Head Variations](#1-foundational-paradigms--multi-head-variations)
   - 1.1 Full Scaled Dot-Product Multi-Head Attention (MHA)
   - 1.2 Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
   - 1.3 Multi-Head Latent Attention (MLA)
   - 1.4 Mixture of Block Attention (MoBA)
2. [Spectral and Singular Value Decomposition (SVD) Attention](#2-spectral-and-singular-value-decomposition-svd-attention)
   - 2.1 Truncated SVD Attention
   - 2.2 SVD Linearized Kernel Attention
   - 2.3 SVD Projection Factorization
   - 2.4 Adaptive SVD Attention (ASVD)
   - 2.5 Normalized Head Attention (NHA)
   - 2.6 Rank-Augmented Linear Attention (RALA)
   - 2.7 SVD-LLM v2 / Minimal SVD
3. [Kernelized, Landmark, and Linear Attention](#3-kernelized-landmark-and-linear-attention)
   - 3.1 Linformer (Low-Rank Sequence Projections)
   - 3.2 Nyströmformer (Landmark Pseudoinverse Approximation)
   - 3.3 Performer (FAVOR+ Positive Orthogonal Random Features)
   - 3.4 Kernelized Decay Attention (KDA)
   - 3.5 AdaSplash / Adaptive Span Linear Attention
4. [State-Space Models & Linear Recurrence](#4-state-space-models--linear-recurrence)
   - 4.1 Gated DeltaNet & HGRN2
   - 4.2 Mamba & Selective State-Space Models
   - 4.3 Griffin SSD (Selective State-Space Duality)
   - 4.4 Jamba / Jamba2 Hybrid SSM-MoE
   - 4.5 Hymba TransX (Cross-Layer Recurrent Fusion)
5. [Sparse, Hashing, and Graph-Structured Attention](#5-sparse-hashing-and-graph-structured-attention)
   - 5.1 Locality-Sensitive Hashing Attention (LSH / Reformer)
   - 5.2 XAttention (Block-Sparse Antidiagonal Routing)
   - 5.3 Vision GNN (ViG) 2D Grid Linear Attention
   - 5.4 Cluster Attention & Routing Attention
   - 5.5 Hamming Attention & Binary Distance Routing
6. [Hybrid, Ensemble & Next-Generation Architectures](#6-hybrid-ensemble--next-generation-architectures)
   - 6.1 Aegis Architecture Suite (v1 through v6)
   - 6.2 BAM U-MoE (Unified Mixture-of-Experts Attention)
   - 6.3 Granite Hybrid Attention
   - 6.4 SOA / Fused Attention Mechanisms
   - 6.5 Spectra & Spectron Multi-Spectral Attention
7. [Comprehensive Theoretical Complexity Master Table](#7-comprehensive-theoretical-complexity-master-table)
8. [Mathematical Notation Glossary](#8-mathematical-notation-glossary)

---

## 1. Foundational Paradigms & Multi-Head Variations

### 1.1 Full Scaled Dot-Product Multi-Head Attention (MHA)
Given input activation sequence $X \in \mathbb{R}^{B \times N \times D}$, queries, keys, and values are computed for head $h \in \{1, \dots, H\}$:

$$Q_h = X W_h^Q, \quad K_h = X W_h^K, \quad V_h = X W_h^V \quad \text{where } W_h^Q, W_h^K \in \mathbb{R}^{D \times d_k}, \; W_h^V \in \mathbb{R}^{D \times d_v}$$

The attention operator computes pairwise softmax affinities:

$$\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_k}} + M\right) V_h$$

$$\text{MHA}(X) = \text{Concat}\left(\text{head}_1, \dots, \text{head}_H\right) W^O$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$
- **Spatial Complexity**: $\mathcal{O}(B \cdot H \cdot N^2)$
- **KV Cache Footprint**: $\mathcal{O}(2 \cdot B \cdot H \cdot N \cdot d_k)$

---

### 1.2 Grouped-Query Attention (GQA)
Interpolates between MHA ($H_Q = H_{KV}$) and MQA ($H_{KV} = 1$) by assigning groups of query heads to share a single Key-Value head:

$$K_{\text{shared}} = \text{repeat\_interleave}\left(K, \frac{H_Q}{H_{KV}}, \text{dim}=\text{head}\right), \quad V_{\text{shared}} = \text{repeat\_interleave}\left(V, \frac{H_Q}{H_{KV}}, \text{dim}=\text{head}\right)$$

$$\text{Attention}_{\text{GQA}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{shared}}^\top}{\sqrt{d_k}} + M\right) V_{\text{shared}}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H_Q \cdot N^2 \cdot d_k)$
- **KV Cache Footprint**: $\mathcal{O}(2 \cdot B \cdot H_{KV} \cdot N \cdot d_k)$, achieving an exact $\frac{H_Q}{H_{KV}}$ memory reduction.

---

### 1.3 Multi-Head Latent Attention (MLA)
Compresses the Key and Value representations into a low-rank latent vector $\mathbf{c}_{KV} \in \mathbb{R}^{B \times N \times d_c}$ prior to projecting into individual head subspaces (DeepSeek-V2/V3):

$$\mathbf{c}_{KV} = X W_{DKV} \in \mathbb{R}^{B \times N \times d_c} \quad (d_c \ll 2 H d_k)$$

$$K = \mathbf{c}_{KV} W_{UK} \in \mathbb{R}^{B \times N \times (H \cdot d_k)}, \quad V = \mathbf{c}_{KV} W_{UV} \in \mathbb{R}^{B \times N \times (H \cdot d_v)}$$

$$\text{Attention}_{\text{MLA}}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

- **Time Complexity**: $\mathcal{O}(B \cdot N \cdot D \cdot d_c + B \cdot H \cdot N^2 \cdot d_k)$
- **KV Cache Footprint**: $\mathcal{O}(B \cdot N \cdot d_c)$, reducing decoding memory by up to 93% compared to standard MHA.

---

### 1.4 Mixture of Block Attention (MoBA)
Routes query tokens dynamically to localized memory blocks and landmark cluster tokens through a learned gating router:

$$g_i = \text{TopK}\left(\text{Softmax}(W_g q_i), k_{\text{blocks}}\right)$$

$$\text{Attention}_{\text{MoBA}}(Q_i) = \sum_{b \in g_i} \text{Softmax}\left(\frac{Q_i K_b^\top}{\sqrt{d_k}}\right) V_b$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot k_{\text{blocks}} \cdot B_s \cdot d_k)$
- **Scaling**: $\mathcal{O}(N)$ when $k_{\text{blocks}}$ and block size $B_s$ remain constant.

---

## 2. Spectral and Singular Value Decomposition (SVD) Attention

### 2.1 Truncated SVD Attention
Directly factors the dense attention affinity matrix $A = \text{softmax}(Q K^\top / \sqrt{d_k}) \in \mathbb{R}^{N \times N}$ using rank-$r$ truncated SVD ($r \ll N$):

$$A \approx U_r \Sigma_r V_r^\top \quad \text{where } U_r \in \mathbb{R}^{N \times r}, \; \Sigma_r \in \mathbb{R}^{r \times r}, \; V_r \in \mathbb{R}^{N \times r}$$

$$\text{Output} = U_r \left( \Sigma_r \left( V_r^\top V \right) \right)$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot r \cdot d_v)$
- **Spatial Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot r)$

---

### 2.2 SVD Linearized Kernel Attention
Linearizes the attention mechanism using non-linear kernel transformations $\phi(x) = \text{ELU}(x) + 1$, then factorizes the resulting inner kernel state $S = \phi(K)^\top V \in \mathbb{R}^{d_k \times d_v}$ via spectral decomposition $U_r \Sigma_r V_r^\top$:

$$\text{Attention}_{\text{SVD-Lin}}(Q, K, V) = \frac{\phi(Q) \left( U_r \Sigma_r V_r^\top \right)}{\phi(Q) \left( \phi(K)^\top \mathbf{1}_N \right)}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k \cdot r)$
- **Memory Scaling**: Strictly linear $\mathcal{O}(N)$ in sequence length.

---

### 2.3 SVD Projection Factorization
Decomposes weight matrices $W \in \mathbb{R}^{D \times D}$ into low-rank factorized pairs $W \approx W_{\text{down}} W_{\text{up}}$ with $W_{\text{down}} \in \mathbb{R}^{D \times r}, W_{\text{up}} \in \mathbb{R}^{r \times D}$:

$$\text{Parameters}: 2 D r \ll D^2 \quad (\text{Compression factor: } D / 2r)$$

---

### 2.4 Adaptive SVD Attention (ASVD)
Dynamically determines the spectral truncation threshold $r_h$ on a per-head basis according to cumulative spectral energy:

$$\min r_h \quad \text{s.t.} \quad \frac{\sum_{i=1}^{r_h} \sigma_i}{\sum_{j=1}^{\min(N, d)} \sigma_j} \ge \tau \quad (\tau \in [0.8, 0.99])$$

- **Time Complexity**: Dynamic $\mathcal{O}(B \sum_h N r_h d_v)$
- **Advantage**: Preserves high rank on attention heads with high spectral entropy while aggressively pruning redundant heads.

---

### 2.5 Normalized Head Attention (NHA)
Stabilizes attention entropy and prevents rank collapse by enforcing covariance normalization across query and key head channels:

$$\tilde{Q}_h = \frac{Q_h - \mathbb{E}[Q_h]}{\sqrt{\text{Var}(Q_h) + \epsilon}}, \quad \tilde{K}_h = \frac{K_h - \mathbb{E}[K_h]}{\sqrt{\text{Var}(K_h) + \epsilon}}$$

$$\text{Attention}_{\text{NHA}}(Q, K, V) = \text{softmax}\left(\frac{\gamma_h \langle \tilde{Q}_h, \tilde{K}_h \rangle}{\sqrt{d_k}}\right) V_h$$

---

### 2.6 Rank-Augmented Linear Attention (RALA)
Augments standard kernel linear attention with an explicit learnable low-rank parameter matrix $U_h V_h^\top \in \mathbb{R}^{d_k \times d_k}$ (rank $r \ll d_k$):

$$\text{Attention}_{\text{RALA}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V + \alpha U V^\top)}{\phi(Q) (\phi(K)^\top \mathbf{1}_N)}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot (d_k d_v + d_k r))$
- **Algorithmic Complexity**: $\mathcal{O}(N)$.

---

## 3. Kernelized, Landmark, and Linear Attention

### 3.1 Linformer
Projects the sequence length dimension $N$ of Keys and Values into a fixed lower-dimensional subspace $k \ll N$ using projection matrices $E, F \in \mathbb{R}^{k \times N}$:

$$K_{\text{proj}} = E K \in \mathbb{R}^{k \times d_k}, \quad V_{\text{proj}} = F V \in \mathbb{R}^{k \times d_v}$$

$$\text{Attention}_{\text{Linformer}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{proj}}^\top}{\sqrt{d_k}} + M\right) V_{\text{proj}}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot k \cdot d_k)$
- **Spatial Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot k)$

---

### 3.2 Nyströmformer
Approximates the full attention matrix via $m$ landmark tokens selected through segment-mean pooling:

$$\tilde{S} = F_1 F_2^+ F_3$$

$$\text{where } F_1 = \text{softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times m}, \quad F_2 = \text{softmax}\left(\frac{\tilde{Q} \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \quad F_3 = \text{softmax}\left(\frac{\tilde{Q} K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times N}$$

$F_2^+$ is the Moore-Penrose pseudoinverse computed via iterative Newton-Schulz iterations ($X_{i+1} = X_i (2I - F_2 X_i)$).

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot (N \cdot m \cdot d_k + m^3 \cdot \text{iters}))$
- **Memory Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot m)$

---

### 3.3 Performer (FAVOR+)
Approximates Gaussian softmax kernels using Positive Orthogonal Random Features:

$$\phi(x) = \frac{h(x)}{\sqrt{M}} \exp\left(W_{\text{ortho}} x - \frac{\|x\|^2}{2}\right)$$

$$\text{Attention}_{\text{Performer}}(Q, K, V) = \frac{\phi(Q) \left( \phi(K)^\top V \right)}{\phi(Q) \left( \phi(K)^\top \mathbf{1}_N \right)}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot M \cdot d_k)$
- **Spatial Complexity**: $\mathcal{O}(B \cdot H \cdot (N \cdot M + M \cdot d_k))$

---

### 3.4 Kernelized Decay Attention (KDA)
Formulates causal sequence processing as an exact linear recurrent state update with a learned exponential decay gating parameter $\beta$:

$$S_t = \sigma(\beta) \odot S_{t-1} + \tilde{k}_t v_t^\top \in \mathbb{R}^{d_k \times d_v} \quad (\text{where } \tilde{k}_t = k_t / \|k_t\|_2)$$

$$o_t = S_t q_t \in \mathbb{R}^{d_v}$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k \cdot d_v)$
- **State Space Memory**: Constant $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ during generation.

---

## 4. State-Space Models & Linear Recurrence

### 4.1 Gated DeltaNet & HGRN2
Gated DeltaNet extends linear attention by introducing associative delta memory update rules:

$$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top$$

$$y_t = S_t q_t \odot \sigma(W_g x_t)$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k \cdot d_v)$
- **Advantage**: Resolves associative recall degradation in standard linear attention.

---

### 4.2 Mamba & Selective State-Space Models
Discretizes continuous state-space ODEs with input-dependent parameters $\Delta, B_t, C_t$:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

$$\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = (\Delta_t A)^{-1} (\exp(\Delta_t A) - I) \cdot \Delta_t B_t$$

- **Time Complexity**: $\mathcal{O}(B \cdot N \cdot D \cdot d_{\text{state}})$ via parallel associative scan.
- **Inference Footprint**: $\mathcal{O}(B \cdot D \cdot d_{\text{state}})$ constant state per step.

---

### 4.3 Griffin SSD (Selective State-Space Duality)
Unifies multi-head attention and selective state-space models into a single matrix multiplication formulation with block-diagonal structured decay masks:

$$Y = \left( (Q K^\top) \odot M_{\text{decay}} \right) V$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k \cdot d_v)$ in recurrent mode; $\mathcal{O}(B \cdot H \cdot N \cdot B_s \cdot d_k)$ in chunked parallel mode.

---

## 5. Sparse, Hashing, and Graph-Structured Attention

### 5.1 Locality-Sensitive Hashing Attention (LSH / Reformer)
Replaces full pairwise comparison with angular random projection hashing $h(x) = \text{sign}(x R)$. Tokens are partitioned into discrete hash buckets $\mathcal{B}_k$, restricting attention to intra-bucket members across $n_r$ independent rounds:

$$\text{Attention}_{\text{LSH}}(Q, K, V) = \frac{1}{n_r} \sum_{r=1}^{n_r} \text{Softmax}\left(\frac{Q_{\mathcal{B}_k}^{(r)} (K_{\mathcal{B}_k}^{(r)})^\top}{\sqrt{d_k}}\right) V_{\mathcal{B}_k}^{(r)}$$

- **Time Complexity**: $\mathcal{O}(n_r \cdot B \cdot H \cdot N \log N \cdot d_k)$
- **Memory Complexity**: $\mathcal{O}(n_r \cdot B \cdot H \cdot N)$

---

### 5.2 XAttention (Block-Sparse with Antidiagonal Routing)
Divides sequence into $M = \lceil N / B_s \rceil$ blocks of size $B_s$. Computes coarse-grained block interaction scores $S_{ij}^{\text{block}} = \frac{1}{B_s^2} \sum_{u, v} Q_{i,u} K_{j,v}^\top / \sqrt{d_k}$ and selectively executes full fine-grained attention only on top-$k$ blocks ($k = (1-s) M$ where $s$ is sparsity):

$$\text{Attention}_{\text{XAttn}}(Q, K, V) = \sum_{j \in \text{TopK}(S_{i,:}^{\text{block}})} \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d_k}}\right) V_j$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot (M^2 + (1-s) N B_s) \cdot d_k)$
- **Memory Complexity**: $\mathcal{O}(B \cdot H \cdot ((N / B_s)^2 + (1-s) N B_s))$

---

### 5.3 Vision GNN (ViG) 2D Grid Linear Attention
Arranges 1D sequences into a 2D spatial grid $H_g \times W_g \approx \sqrt{N} \times \sqrt{N}$. Decouples ELU kernel linear attention along rows and columns, fusing representations through learned dynamic input gating:

$$\text{Output} = g_{\text{row}} \odot \text{LinAttn}_{\text{row}}(X) + g_{\text{col}} \odot \text{LinAttn}_{\text{col}}(X)$$

- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k)$
- **Spatial Complexity**: $\mathcal{O}(B \cdot H \cdot N \cdot d_k)$

---

## 6. Hybrid, Ensemble & Next-Generation Architectures

### 6.1 Aegis Architecture Suite (v1 through v6)
Aegis integrates multi-branch feature pathways combining:
1. Low-rank factorized projections ($W_{\text{down}} W_{\text{up}}$).
2. Spectral linear kernel attention with adaptive head whitening.
3. Local sliding-window convolutions for short-range inductive bias.
4. Input-dependent context fusion gating:

$$\text{Output}_{\text{Aegis}} = g_{\text{local}} \odot \text{Conv}_{\text{local}}(X) + g_{\text{spectral}} \odot \text{LinAttn}_{\text{spectral}}(X) + g_{\text{global}} \odot \text{Attn}_{\text{sparse}}(X)$$

---

### 6.2 BAM U-MoE (Unified Mixture-of-Experts Attention)
Interleaves sparse expert routing directly into attention projections, dynamically dispatching attention computation to specialized head experts based on token syntactic category.

---

### 6.3 Spectra & Spectron Multi-Spectral Attention
Applies discrete orthogonal transforms (DCT / Fast Fourier Transform) across head channels prior to non-linear activation, capturing frequency-domain periodicity in natural language and multi-modal signals.

---

## 7. Comprehensive Theoretical Complexity Master Table

| Category | Architecture / Module | Time Complexity | Memory Complexity | KV Cache Scaling | Algorithmic Class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Multi-Head** | **Full Attention (MHA)** | $\mathcal{O}(B H N^2 d)$ | $\mathcal{O}(B H N^2)$ | $\mathcal{O}(B H N d)$ | Quadratic $\mathcal{O}(N^2)$ |
| | **GQA / MQA** | $\mathcal{O}(B H_Q N^2 d)$ | $\mathcal{O}(B H_Q N^2)$ | $\mathcal{O}(B H_{KV} N d)$ | Quadratic $\mathcal{O}(N^2)$ |
| | **MLA (DeepSeek-V2)** | $\mathcal{O}(B N D d_c + B H N^2 d)$ | $\mathcal{O}(B H N^2)$ | $\mathcal{O}(B N d_c)$ | Quadratic $\mathcal{O}(N^2)$ |
| | **Normalized Head (NHA)** | $\mathcal{O}(B H N^2 d)$ | $\mathcal{O}(B H N^2)$ | $\mathcal{O}(B H N d)$ | Quadratic $\mathcal{O}(N^2)$ |
| **Spectral / SVD** | **SVD Attention** | $\mathcal{O}(B H N r d)$ | $\mathcal{O}(B H N r)$ | $\mathcal{O}(B H N r)$ | Low-Rank $\mathcal{O}(N r)$ |
| | **SVD Linear** | $\mathcal{O}(B H N d r)$ | $\mathcal{O}(B H (d^2 + N d))$ | $\mathcal{O}(B H d^2)$ | Linear $\mathcal{O}(N)$ |
| | **Adaptive SVD (ASVD)**| $\mathcal{O}(B \sum_h N r_h d)$ | $\mathcal{O}(B \sum_h N r_h)$ | $\mathcal{O}(B \sum_h N r_h)$ | Dynamic Low-Rank |
| | **RALA** | $\mathcal{O}(B H N (d^2 + d r))$ | $\mathcal{O}(B H (d^2 + N d))$ | $\mathcal{O}(B H d^2)$ | Linear $\mathcal{O}(N)$ |
| | **SVD-LLM v2** | $\mathcal{O}(B H N r d)$ | $\mathcal{O}(B H N r)$ | $\mathcal{O}(B H N r)$ | Low-Rank Factorized |
| **Kernel / Landmark** | **Linformer** | $\mathcal{O}(B H N k d)$ | $\mathcal{O}(B H N k)$ | $\mathcal{O}(B H k d)$ | Linear $\mathcal{O}(N)$ |
| | **Nyströmformer** | $\mathcal{O}(B H (N m d + m^3))$ | $\mathcal{O}(B H N m)$ | $\mathcal{O}(B H m d)$ | Landmark Linear |
| | **Performer (FAVOR+)** | $\mathcal{O}(B H N M d)$ | $\mathcal{O}(B H (N M + M d))$| $\mathcal{O}(B H M d)$ | Random Feature Linear |
| | **KDA (Decay Attention)**| $\mathcal{O}(B H N d^2)$ | $\mathcal{O}(B H d^2)$ | $\mathcal{O}(B H d^2)$ | Recurrent Linear $\mathcal{O}(N)$ |
| | **ViG 2D Grid** | $\mathcal{O}(B H N d)$ | $\mathcal{O}(B H N d)$ | $\mathcal{O}(B H N d)$ | Spatial Grid Linear |
| **State-Space / SSM** | **Mamba / Mamba2** | $\mathcal{O}(B N D d_{\text{state}})$ | $\mathcal{O}(B N D)$ | $\mathcal{O}(B D d_{\text{state}})$ | State-Space Linear |
| | **Gated DeltaNet** | $\mathcal{O}(B H N d^2)$ | $\mathcal{O}(B H d^2)$ | $\mathcal{O}(B H d^2)$ | Recurrent Delta Linear |
| | **Griffin SSD** | $\mathcal{O}(B H N d^2)$ | $\mathcal{O}(B H d^2)$ | $\mathcal{O}(B H d^2)$ | Duality Linear |
| | **HGRN2** | $\mathcal{O}(B H N d^2)$ | $\mathcal{O}(B H d^2)$ | $\mathcal{O}(B H d^2)$ | Gated Recurrent Linear |
| **Sparse / Routing** | **LSH (Reformer)** | $\mathcal{O}(n_r B H N \log N d)$ | $\mathcal{O}(n_r B H N)$ | $\mathcal{O}(n_r B H N d)$ | Sub-Quadratic $\mathcal{O}(N \log N)$ |
| | **XAttention** | $\mathcal{O}(B H (M^2 + (1-s) N B_s) d)$ | $\mathcal{O}(B H ((N/B_s)^2 + (1-s) N B_s))$ | $\mathcal{O}(B H N d)$ | Block Sparse |
| | **Cluster Attention** | $\mathcal{O}(B H N k_c d)$ | $\mathcal{O}(B H N k_c)$ | $\mathcal{O}(B H k_c d)$ | Clustered Routing |
| | **MoBA** | $\mathcal{O}(B H N k_{\text{blk}} B_s d)$ | $\mathcal{O}(B H N k_{\text{blk}} B_s)$ | $\mathcal{O}(B H N d)$ | Block MoE Sparse |
| **Hybrid / Advanced** | **Aegis Family (v1-v6)** | $\mathcal{O}(B H N d + B H N r d)$ | $\mathcal{O}(B H (N d + d^2))$ | $\mathcal{O}(B H (d^2 + r d))$ | Multi-Branch Hybrid |
| | **Spectra / Spectron** | $\mathcal{O}(B H N \log N d)$ | $\mathcal{O}(B H N d)$ | $\mathcal{O}(B H N d)$ | Spectral Orthogonal |
| | **BAM U-MoE** | $\mathcal{O}(B H N d \cdot k_{\text{expert}})$ | $\mathcal{O}(B H N d)$ | $\mathcal{O}(B H N d)$ | Dynamic Expert MoE |

---

## 8. Mathematical Notation Glossary

| Symbol | Definition |
| :--- | :--- |
| $B$ | Batch size |
| $N$ | Sequence length (token count) |
| $D$ / $d_{\text{model}}$ | Total hidden representation dimension |
| $H$ / $H_Q$ | Number of query attention heads |
| $H_{KV}$ | Number of Key-Value attention heads |
| $d_k, d_v, d$ | Per-head subspace dimension ($d = D / H$) |
| $r$ | SVD truncation rank / low-rank bottleneck rank |
| $d_c$ | MLA Key-Value latent compression dimension |
| $k$ | Linformer sequence projection dimension |
| $m$ | Nyström landmark token count |
| $M$ | Performer random feature projection count |
| $B_s$ | Block size in block-sparse routing |
| $s$ | Sparsity ratio ($0 \le s < 1$) |
| $n_r$ | LSH hashing rounds |
| $\phi(\cdot)$ | Kernel feature map transformation |
| $\sigma(\cdot)$ | Sigmoid activation function |
| $\odot$ | Hadamard (element-wise) product |
| $\otimes$ | Kronecker / outer product |
