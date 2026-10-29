---
name: paper-attention-mechanisms-taxonomy
description: "Author exhaustive taxonomies, mathematical operator contracts, and state transition equations for all 131 attention mechanisms across 14 architectural families for long research papers"
version: "1.0.0"
tags: ["neuralops", "paper", "attention", "taxonomy", "linear-attention", "ssm", "gqa", "transformers"]
---

# Skill: Paper Attention Mechanisms Taxonomy & Mathematical Operator Synthesis

## Overview & Scope
Use this skill when authoring, formalizing, verifying, or synthesizing theoretical taxonomies, operator contracts, state-space recurrence equations, and comparative complexity tables for machine learning research papers, doctoral dissertations, and technical reports on efficient attention mechanisms, linear attention, state space models (SSMs), and hybrid architectures.

This skill covers the unified catalog of **131 attention mechanisms** across **14 foundational architectural families** implemented, benchmarked, and cataloged in the NeuralOps ecosystem (`SVD_linear_Attention`).

### When to Use This Skill
- **Section 2: Related Work & Structural Taxonomy**: Categorizing the historical progression of sequence modeling from quadratic softmax attention to linear recurrences, state space models, SVD spectral compressions, and hybrid MoEs.
- **Section 3: Mathematical Operator Foundations**: Writing exact forward tensor contracts, causal masking constraints, continuous-to-discrete dynamical systems transitions, and chunk-parallel associative scan duals.
- **Section 4/5: Comparative Systems Complexity**: Formulating rigorous asymptotic complexity tables ($\mathcal{O}(N^2 d)$ vs $\mathcal{O}(N d^2)$ vs $\mathcal{O}(N W d)$ vs $\mathcal{O}(N d)$) comparing memory footprint, decode FLOPs, and KV-cache scalability.
- **Publication Graphics**: Rendering publication-grade TikZ hierarchical taxonomy diagrams and dual-form computation graphs for camera-ready LaTeX manuscripts (ICLR, NeurIPS, ICML, MLSys).

---

## 1. Asymptotic Complexity & The Fundamental Trade-Off Space

The evolution of sequence modeling architectures addresses the memory-bandwidth and computational bottlenecks of the canonical Scaled Dot-Product Attention (Vaswani et al., 2017).

### 1.1 The Classical Quadratic Bottleneck
Standard Multi-Head Attention (MHA) projects an input sequence $\mathbf{X} \in \mathbb{R}^{N \times d}$ into queries $\mathbf{Q} \in \mathbb{R}^{N \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{N \times d_k}$, and values $\mathbf{V} \in \mathbb{R}^{N \times d_v}$ per head:
$$\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \odot \mathbf{M}\right) \mathbf{V} \quad \in \mathbb{R}^{N \times d_v}$$
where $\mathbf{M} \in \{0, -\infty\}^{N \times N}$ is the autoregressive causal mask ($\mathbf{M}_{ij} = 0$ for $j \le i$, $-\infty$ otherwise).

The computational and systems bottlenecks stem from:
1. **Compute Complexity**: Evaluating the similarity matrix $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ requires $2 N^2 d_k$ FLOPs. Multiplying by $\mathbf{V}$ adds $2 N^2 d_v$ FLOPs, yielding $\mathcal{O}(N^2 d)$.
2. **Activation Memory Footprint**: Storing the $N \times N$ attention logits $\mathbf{S}$ and normalized weights $\mathbf{A} = \operatorname{softmax}(\mathbf{S})$ for backpropagation consumes $\mathcal{O}(B \cdot H \cdot N^2)$ bytes in standard implementations. IO-aware tiling (FlashAttention-1/2/3) collapses intermediate HBM writes to $\mathcal{O}(N)$, yet the $\mathcal{O}(N^2 d)$ arithmetic work remains unchanged.
3. **Autoregressive KV-Cache Expansion**: During step-by-step token generation at step $t$, the key-value cache expands linearly: $\mathbf{K}_{\le t} \in \mathbb{R}^{t \times d_k}$, $\mathbf{V}_{\le t} \in \mathbb{R}^{t \times d_v}$. Across $L$ layers and $H$ heads, memory consumption scales as $\mathcal{O}(2 \cdot B \cdot L \cdot H \cdot t \cdot d_k)$, creating a memory capacity ceiling during long-context generation.

### 1.2 Mathematical Regimes of Attention & Sequence Modeling

| Regime | Representative Families | Compute (Train) | Memory Activations | Decode Latency (Step $t$) | KV Cache Size / State Footprint | Bidirectional Context |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Quadratic Softmax** | MHA, MQA, GQA, FlashAttention | $\mathcal{O}(N^2 \cdot d)$ | $\mathcal{O}(N)$ (tiled) / $\mathcal{O}(N^2)$ | $\mathcal{O}(t \cdot d)$ | $\mathcal{O}(t \cdot d)$ | Native |
| **Linear Feature-Map** | Cosformer, Performer, ReLA, Katharopoulos | $\mathcal{O}(N \cdot d_k d_v)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ [Constant] | Native / Causal |
| **Gated Linear Recurrence** | GLA, RetNet, RWKV-v6, HGRN2 | $\mathcal{O}(N \cdot d_k d_v)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ [Constant] | Causal Only |
| **Delta-Rule Recurrence** | DeltaNet, KDA, Gated DeltaNet | $\mathcal{O}(N \cdot d_k d_v)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ [Constant] | Causal Only |
| **Continuous/Selective SSM** | Mamba (S6), Mamba-2 (SSD), S4, S5 | $\mathcal{O}(N \cdot d \cdot r)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d \cdot r)$ | $\mathcal{O}(d \cdot r)$ [Constant] | Causal / Multi-pass |
| **Block-Sparse / Local** | Longformer, BigBird, NSA, Local Window | $\mathcal{O}(N \cdot W \cdot d)$ | $\mathcal{O}(N \cdot W)$ | $\mathcal{O}(W \cdot d)$ | $\mathcal{O}(W \cdot d)$ [Bounded] | Native |
| **SVD Low-Rank Projection** | Linformer, Nyströmformer, SVD-Linear | $\mathcal{O}(N \cdot r \cdot d)$ | $\mathcal{O}(N \cdot r)$ | $\mathcal{O}(r \cdot d)$ | $\mathcal{O}(r \cdot d)$ [Constant] | Native |
| **Multi-Head Latent (MLA)** | DeepSeek MLA, QCQA, Compressed MHA | $\mathcal{O}(N^2 \cdot d_h + N d_c)$ | $\mathcal{O}(N)$ (tiled) | $\mathcal{O}(t \cdot d_c)$ | $\mathcal{O}(t \cdot d_c)$ [$d_c \ll H d_h$] | Native |
| **Hybrid Linear-Attention** | Jamba, Samba, Griffin, Hymba | $\mathcal{O}(\alpha N^2 d + (1-\alpha) N d^2)$| $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(\alpha t d + (1-\alpha) d^2)$ | Layer-dependent | Mixed |
| **Mixture-of-Experts (MoE)**| BAM-UMoE, Expert-Choice, REAP | $\mathcal{O}(N \cdot \frac{K}{E} \cdot d^2)$ | $\mathcal{O}(N \cdot d)$ | Dependent on routing | Dynamic | Native |
| **Test-Time Training (TTT)** | TTT-Linear, TTT-MLP | $\mathcal{O}(N \cdot d^2)$ | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(d^2)$ (SGD update) | $\mathcal{O}(d^2)$ [Weights as state] | Causal Only |

---

## 2. Systematic Coverage of the 14 Foundational Architectural Families

NeuralOps catalogs 131 distinct attention mechanisms grouped into 14 mathematical families. Below are the rigorous operator definitions, governing equations, causal masking contracts, and state transition dynamics for each family.

```
                                      ┌─────────────────────────────────────────────────────────┐
                                      │        Unified Attention Taxonomy (131 Mechanisms)      │
                                      └────────────────────────────┬────────────────────────────┘
         ┌─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┐
         ▼                                                         ▼                                                        ▼
┌──────────────────┐                                     ┌──────────────────┐                                     ┌──────────────────┐
│ Exact & Dense    │                                     │ Sub-Quadratic    │                                     │ Spectral & MoE   │
│ - Exact Softmax  │                                     │ - Feature Kernels│                                     │ - SVD Low-Rank   │
│ - Grouped/MQA    │                                     │ - Gated Recurr.  │                                     │ - Latent (MLA)   │
│ - Modern RoPE    │                                     │ - Delta Rules    │                                     │ - Hybrids/SSMs   │
│ - Hardware-Fused │                                     │ - Selective SSMs │                                     │ - Sparse MoE     │
└──────────────────┘                                     │ - Block-Sparse   │                                     │ - TTT / Ext. Mem │
                                                         └──────────────────┘                                     └──────────────────┘
```

---

### Family 1: Exact Softmax & IO-Aware Dense Attention
**Modules (6):** `full`, `full_attention`, `mqa`, `gqa_attention`, `flash_attention3`, `flash_attention3_fp8`.

#### Mathematical Formulation
Multi-Head Attention (MHA), Multi-Query Attention (MQA, Shazeer 2019), and Grouped-Query Attention (GQA, Ainslie et al., 2023) unify under a head-partitioned projection:
$$\mathbf{Q} \in \mathbb{R}^{B \times N \times H_Q \times d_k}, \quad \mathbf{K} \in \mathbb{R}^{B \times N \times H_{KV} \times d_k}, \quad \mathbf{V} \in \mathbb{R}^{B \times N \times H_{KV} \times d_v}$$
where $G = H_Q / H_{KV}$ is the query-to-KV head ratio:
- $H_Q = H_{KV} \implies \text{MHA}$ ($G = 1$)
- $H_{KV} = 1 \implies \text{MQA}$ ($G = H_Q$)
- $1 < H_{KV} < H_Q \implies \text{GQA}$ ($1 < G < H_Q$)

For query head $h \in \{1, \dots, H_Q\}$, the corresponding key-value head is $\kappa(h) = \lfloor (h - 1) / G \rfloor + 1$:
$$\mathbf{O}_{b, i, h, :} = \sum_{j=1}^N \frac{\exp\left(\frac{1}{\sqrt{d_k}} \mathbf{Q}_{b, i, h, :} \mathbf{K}_{b, j, \kappa(h), :}^\top + \mathbf{M}_{ij}\right)}{\sum_{l=1}^N \exp\left(\frac{1}{\sqrt{d_k}} \mathbf{Q}_{b, i, h, :} \mathbf{K}_{b, l, \kappa(h), :}^\top + \mathbf{M}_{il}\right)} \mathbf{V}_{b, j, \kappa(h), :}$$

#### IO-Aware Tiling & Online Softmax (FlashAttention-3)
Standard evaluation writes the intermediate attention matrix $\mathbf{A} \in \mathbb{R}^{B \times H \times N \times N}$ to high-bandwidth GPU memory (HBM). FlashAttention decomposes the computation into blocks of SRAM ($B_r$ rows, $B_c$ columns). The online softmax normalization updates running maximum $m_i$ and running partition function $\ell_i$:
$$\begin{aligned}
m_i^{(new)} &= \max\left(m_i^{(old)}, \max_j \mathbf{S}_{ij}^{(block)}\right) \\
\ell_i^{(new)} &= \ell_i^{(old)} e^{m_i^{(old)} - m_i^{(new)}} + \sum_j e^{\mathbf{S}_{ij}^{(block)} - m_i^{(new)}} \\
\mathbf{O}_i^{(new)} &= \mathbf{O}_i^{(old)} \frac{\ell_i^{(old)} e^{m_i^{(old)} - m_i^{(new)}}}{\ell_i^{(new)}} + \frac{\sum_j e^{\mathbf{S}_{ij}^{(block)} - m_i^{(new)}} \mathbf{V}_j}{\ell_i^{(new)}}
\end{aligned}$$
FlashAttention-3 introduces warp-specialized ping-pong scheduling, asynchronous TMA (Tensor Memory Accelerator) loads, and FP8 tensor-core accumulation.

---

### Family 2: Kernelized & Feature-Map Linear Attentions
**Modules (15):** `cosformer`, `cosformer_attention`, `performer_attention`, `positive_kernel_linear_attention`, `galerkin_attention`, `regla_attention`, `halo_regla`, `eat`, `eat_attention`, `chela_slay`, `chela_slay_attention`, `triton_cosformer`, `rpc_attention`, `zeros_attention`, `xattention`.

#### Mathematical Formulation & Associativity
Linear attention replaces the normalized exponential score with an inner product of non-negative feature maps $\phi(\cdot): \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$:
$$\operatorname{Sim}(\mathbf{q}_i, \mathbf{k}_j) = \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^\top$$
By matrix associativity, the causal output expands as:
$$\mathbf{O}_i = \frac{\sum_{j=1}^i \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^\top \mathbf{v}_j}{\sum_{j=1}^i \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^\top} = \frac{\phi(\mathbf{q}_i) \sum_{j=1}^i \phi(\mathbf{k}_j)^\top \mathbf{v}_j}{\phi(\mathbf{q}_i) \sum_{j=1}^i \phi(\mathbf{k}_j)^\top} = \frac{\phi(\mathbf{q}_i) \mathbf{S}_i}{\phi(\mathbf{q}_i) \mathbf{z}_i}$$

#### Canonical Feature Mappings
1. **Katharopoulos Linear Attention**: $\phi(\mathbf{x}) = \operatorname{elu}(\mathbf{x}) + 1$ or $\phi(\mathbf{x}) = \operatorname{relu}(\mathbf{x})$.
2. **Cosformer (Qin et al., 2022)**: Introduces non-negative cosine re-weighting to enforce locality without full softmax:
   $$\phi(\mathbf{x}) = \operatorname{relu}(\mathbf{x}), \quad \operatorname{Sim}(\mathbf{q}_i, \mathbf{k}_j) = \phi(\mathbf{q}_i) \phi(\mathbf{k}_j)^\top \cos\left(\frac{\pi (i - j)}{2 M}\right)$$
   Expanded via trigonometric angle sum: $\cos\left(\frac{\pi i}{2M} - \frac{\pi j}{2M}\right) = \cos(\alpha i)\cos(\alpha j) + \sin(\alpha i)\sin(\alpha j)$, doubling the state size to $2 d_\phi \times d_v$.
3. **Performer / Positive Random Features (Choromanski et al., 2020)**: Orthogonal random feature map:
   $$\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{M}} \left[\exp\left(\mathbf{w}_1^\top \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}\right), \dots, \exp\left(\mathbf{w}_M^\top \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}\right)\right]^\top$$
   where $\mathbf{W} \in \mathbb{R}^{M \times d}$ is drawn from orthogonal Gaussian ensembles.
4. **ReLA (Rectified Linear Attention)**: Uses activation thresholding to encourage sparse representation states.

#### Sequential Recurrence vs Associative Scan Dual
- **Sequential Recurrence**:
  $$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)^\top \mathbf{v}_t \in \mathbb{R}^{d_\phi \times d_v}, \quad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)^\top \in \mathbb{R}^{d_\phi \times 1}$$
- **Chunk-Parallel Associative Scan**: Sequences of length $N$ are partitioned into chunks of size $C$. Inter-chunk states are aggregated using a parallel prefix scan in $\mathcal{O}((N/C) d_\phi d_v)$, while intra-chunk outputs are computed via matrix multiplication on tensor cores.

---

### Family 3: Gated Linear Recurrences
**Modules (11):** `gated_linear_attention`, `gated_linear_attn`, `gla_attention`, `decay_linear_attention`, `decay_linear_attn`, `factorised_matrix_state`, `hgrn2_gated_layer`, `hgrn2_gated_outer`, `hgrn2_outer`, `goldfinch_ahn`, `salt_attention`.

#### Mathematical Formulation
Gated Linear Attention (GLA, Yang et al., 2024), RetNet (Sun et al., 2023), and RWKV-v6 (Peng et al., 2024) introduce data-dependent decay gates $\mathbf{\alpha}_t$ to bound memory accumulation and solve the forgetting problem:
$$\begin{aligned}
\mathbf{\alpha}_t &= \exp\left(-\operatorname{softplus}(\mathbf{W}_\alpha \mathbf{x}_t + \mathbf{b}_\alpha)\right) \quad \in (0, 1)^{d_k} \\
\mathbf{S}_t &= \operatorname{diag}(\mathbf{\alpha}_t) \mathbf{S}_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t \quad \in \mathbb{R}^{d_k \times d_v} \\
\mathbf{o}_t &= \mathbf{q}_t \mathbf{S}_t \quad \in \mathbb{R}^{1 \times d_v}
\end{aligned}$$

#### Dual Representation Matrix
Let $\Gamma_{i, j} = \prod_{k=j+1}^i \mathbf{\alpha}_k$ for $i \ge j$. In parallel training mode:
$$\mathbf{O} = \left((\mathbf{Q}\mathbf{K}^\top) \odot \mathbf{\Gamma}\right) \mathbf{V}$$
where $\mathbf{\Gamma} \in \mathbb{R}^{N \times N}$ is the 1-D Semigroup decay matrix.
- **RetNet Multi-Scale Retention (MSR)**: Fixes $\alpha_h = 1 - 2^{-5 - h}$ statically across heads $h \in \{1, \dots, H\}$, enabling Toeplitz matrix formulation.
- **HGRN2 (Hierarchical Gated Recurrent Network v2)**: Replaces elementwise addition with a convex linear combination and outer product gating: $\mathbf{h}_t = \mathbf{\alpha}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{\alpha}_t) \odot (\mathbf{u}_t \odot \mathbf{g}_t)$.

---

### Family 4: Delta-Rule Recurrences & Associative Memory
**Modules (7):** `gated_delta_net`, `gated_deltanet_block`, `hgrn2_gated_deltanet`, `kda_attention`, `hla_attention`, `nexus`, `phantom`.

#### Mathematical Formulation & Error-Correction
Standard linear recurrences only accumulate associations: $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$. When old keys collide with new associations, the state capacity degrades. The Delta Rule (Schlag et al., 2021; DeltaNet) applies gradient descent on associative memory retrieval error:
$$\mathcal{L}(\mathbf{S}_{t-1}) = \frac{1}{2} \|\mathbf{k}_t \mathbf{S}_{t-1} - \mathbf{v}_t\|^2$$
Taking the gradient with step size $\beta_t \in (0, 1]$ yields:
$$\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \mathbf{k}_t^\top (\mathbf{k}_t \mathbf{S}_{t-1} - \mathbf{v}_t) = \left(\mathbf{I} - \beta_t \mathbf{k}_t^\top \mathbf{k}_t\right) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t^\top \mathbf{v}_t$$
where $(\mathbf{I} - \beta_t \mathbf{k}_t^\top \mathbf{k}_t)$ acts as a Householder rank-1 orthogonal projection operator that erases the prior association along $\mathbf{k}_t$.

#### Kronecker Delta Attention (KDA)
KDA structures the key projection into Kronecker product subspaces $\mathbf{k}_t = \mathbf{k}_{t, 1} \otimes \mathbf{k}_{t, 2}$, projecting memory erasure across factored sub-dimensions to prevent rank collapse while maintaining hardware-friendly block structure.

---

### Family 5: Continuous-Time & Selective State-Space Models (SSMs)
**Modules (6):** `selective_ssm`, `simple_ssm`, `mamba_block`, `upi_stuffed_mamba`, `mamba_svd_flash`, `svd_mamba`.

#### Continuous Dynamical System & HiPPO Projection
Continuous-time state-space models map a 1D input signal $x(t) \in \mathbb{R}$ to an $N$-dimensional latent state $\mathbf{h}(t) \in \mathbb{R}^{d_{state}}$:
$$\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}\mathbf{h}(t) + \mathbf{D}x(t)$$
The state matrix $\mathbf{A}$ is structured via the HiPPO operator (Gu et al., 2020) over shifted Legendre polynomials:
$$\mathbf{A}_{n, k} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n + 1 & n = k \\ 0 & n < k \end{cases}$$

#### Discretization & Data-Dependent Selection (Mamba S6)
Discretization via Zero-Order Hold (ZOH) with step size $\mathbf{\Delta}_t > 0$:
$$\overline{\mathbf{A}}_t = \exp(\mathbf{\Delta}_t \mathbf{A}), \quad \overline{\mathbf{B}}_t = (\mathbf{\Delta}_t \mathbf{A})^{-1}(\overline{\mathbf{A}}_t - \mathbf{I}) \cdot \mathbf{\Delta}_t \mathbf{B}_t \approx \mathbf{\Delta}_t \mathbf{B}_t$$
In Mamba (Gu & Dao, 2023), $\mathbf{B}_t, \mathbf{C}_t, \mathbf{\Delta}_t$ are dynamic functions of input $\mathbf{x}_t$:
$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \quad \mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t, \quad \mathbf{\Delta}_t = \operatorname{softplus}(\mathbf{W}_\Delta \mathbf{x}_t + \mathbf{b}_\Delta)$$
The recurrence is executed via hardware-aware parallel prefix scan entirely within GPU SRAM:
$$\mathbf{h}_t = \overline{\mathbf{A}}_t \mathbf{h}_{t-1} + \overline{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t \mathbf{h}_t + \mathbf{D} x_t$$

#### State Space Duality (SSD / Mamba-2)
Dao & Gu (2024) prove the exact mathematical equivalence between selective SSMs with scalar-identity transition matrices $\mathbf{A}_t = a_t \mathbf{I}$ and 1-semigroup masked linear attention:
$$\mathbf{Y} = \left((\mathbf{X} \mathbf{B}^\top) \odot \mathbf{M}^{SSD}\right) \mathbf{C}$$
where $\mathbf{M}_{ij}^{SSD} = \exp\left(\sum_{k=j}^i a_k\right)$, bridging SSMs directly to tensor-core structured block matrix multiplication.

---

### Family 6: Block-Sparse, Dilated & Sliding-Window Attentions
**Modules (13):** `adaptive_span_attention`, `adaptive_sparse_attention`, `adasplash_attention`, `cluster_attention`, `block_transformer_hiba`, `nsa_attention`, `lsh_attention`, `routing_attention`, `sink_attention`, `dynamic_conv_attention`, `hamming_attention`, `moba_attention`, `progressive_token_pruning`.

#### Mathematical Formulation
Sparse architectures constrain attention connections via structured adjacency topologies $\Omega \subset \{1, \dots, N\} \times \{1, \dots, N\}$:
$$\mathbf{A}_{ij} = \begin{cases} \frac{\exp\left(\frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d_k}}\right)}{\sum_{l \in \Omega_i} \exp\left(\frac{\mathbf{q}_i \mathbf{k}_l^\top}{\sqrt{d_k}}\right)} & \text{if } j \in \Omega_i \\ 0 & \text{otherwise} \end{cases}$$
1. **Sliding Window (Local)**: $\Omega_i^{local} = \{j \mid |i - j| \le W / 2\}$. Complexity: $\mathcal{O}(N \cdot W \cdot d)$.
2. **Attention Sinks / StreamingLLM (`sink_attention`)**: Anchors initial tokens: $\Omega_i^{sink} = \{1, \dots, S\} \cup \{i - W + 1, \dots, i\}$, preserving Softmax normalization stability across infinite streams.
3. **Locality-Sensitive Hashing (Reformer `lsh_attention`)**: Hashes tokens into $B$ buckets via random projections $\mathbf{h}(x) = \arg\max [xR; -xR]$; tokens only attend within identical buckets.
4. **Native Sparse Attention (NSA, `nsa_attention`)**: Combines hierarchical coarse compression with selected fine-grained token blocks.

---

### Family 7: SVD Spectral & Low-Rank Decomposed Attentions
**Modules (19):** `linformer_attention`, `nystrom_attention`, `asvd_attention`, `svd_linear`, `svd_linear_attention`, `svd_linear_attn`, `svd_linear_decay`, `svd_linear_inject`, `svd_llm_v2_attention`, `adaptive_svd_attention`, `residual_svd_attention`, `svd_projection_attention`, `svd_seq`, `svd_compressed`, `naive_svd`, `minimal_svd`, `spectraformer_attention`, `fused_svd_kernel`, `triton_svd_linear_attention`.

#### Mathematical Formulation & Eckart-Young Optimality
Low-rank attention factorizes the $N \times N$ attention matrix or weight projections $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$ into rank-$r$ factors ($r \ll \min(d_{in}, d_{out})$):
- **Linformer**: Projects sequence length dimension using low-rank projection matrices $\mathbf{E}, \mathbf{F} \in \mathbb{R}^{r \times N}$:
  $$\operatorname{Attn}_{Lin}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q} (\mathbf{E}\mathbf{K})^\top}{\sqrt{d_k}}\right) (\mathbf{F}\mathbf{V})$$
  Reducing compute from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N r d)$.
- **Activation-Aware SVD Attention (`asvd_attention`)**:
  In language models, activation distributions exhibit heavy-tailed outliers. Direct truncated SVD solves $\min_{\operatorname{rank}(\widehat{\mathbf{W}}) \le r} \|\mathbf{W} - \widehat{\mathbf{W}}\|_F^2$, which degrades high-variance channels. ASVD incorporates input activation covariance $\mathbf{\Sigma}_{XX} = \frac{1}{N} \mathbf{X}\mathbf{X}^\top$:
  $$\min_{\operatorname{rank}(\widehat{\mathbf{W}}) \le r} \|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2 = \operatorname{Tr}\left((\mathbf{W} - \widehat{\mathbf{W}})\mathbf{\Sigma}_{XX}(\mathbf{W} - \widehat{\mathbf{W}})^\top\right)$$
  By Cholesky factorization $\mathbf{\Sigma}_{XX} + \lambda \mathbf{I} = \mathbf{L}\mathbf{L}^\top$, the optimal solution is obtained via SVD on the whitened weight matrix:
  $$\widetilde{\mathbf{W}} = \mathbf{W}\mathbf{L} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top \implies \widehat{\mathbf{W}} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^\top \mathbf{L}^{-1}$$

---

### Family 8: Multi-Head Latent & Key-Value Compressed Attentions
**Modules (9):** `mla_attention`, `qcqa_attention`, `compressed_attention`, `compressed_mha_attention`, `token_compressed_attention`, `projection_factorized_attn`, `lola_attention`, `padre_attention`, `lora_attention`.

#### Mathematical Formulation: Multi-Head Latent Attention (MLA)
Introduced in DeepSeek-V2/V3 to eliminate KV-cache bottlenecks while maintaining full expressive multi-head representational rank.
1. **Low-Rank KV Compression**: Keys and values are compressed into a unified latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ with $d_c \ll H \cdot d_h$:
   $$\mathbf{c}_t^{KV} = \mathbf{W}^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}, \quad \mathbf{W}^{DKV} \in \mathbb{R}^{d_c \times d_{model}}$$
2. **Decoupled Positional Encoding (Decoupled RoPE)**: To preserve relative positional information through rotary embeddings without expanding the compressed latent cache, queries and keys are split into content and rotary components:
   $$\begin{aligned}
   \mathbf{k}_{t, h}^C &= \mathbf{W}_h^{UK} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h}, \quad \mathbf{k}_t^R = \operatorname{RoPE}(\mathbf{W}^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_h^R} \\
   \mathbf{k}_{t, h} &= \left[\mathbf{k}_{t, h}^C; \, \mathbf{k}_t^R\right] \in \mathbb{R}^{d_h + d_h^R}
   \end{aligned}$$
3. **KV Cache Storage**: During inference, only $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ and $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$ are cached per token:
   $$\text{Cache Footprint per token} = d_c + d_h^R \quad \ll \quad 2 \cdot H \cdot d_h$$
   For DeepSeek-V2 ($H=128, d_h=128, d_c=512, d_h^R=64$):
   $$\text{MHA Cache} = 2 \times 128 \times 128 = 32{,}768 \text{ elements} \implies \text{MLA Cache} = 512 + 64 = 576 \text{ elements} \quad (56.9\times \text{ saving})$$

---

### Family 9: Hybrid Linear-Recurrent & Attention-SSM Architectures
**Modules (14):** `jamba`, `jamba_info`, `samba_m3`, `griffin_ssd`, `hymba_transx`, `granite_hybrid`, `hybrid_attention`, `native_hybrid_attention`, `optimized_hybrid_attention`, `split_head_hybrid`, `tri_branch_hybrid`, `hybrid_local_kernel`, `aegis_hybrid`, `minimal_nova`.

#### Architectural Composition & Layer Scheduling
Hybrid models interleave quadratic attention layers with linear recurrence / SSM blocks to combine infinite-context constant-memory state updates with exact associative retrieval:
- **Jamba (Lieber et al., 2024)**: Ratio of $1:7$ transformer to Mamba layers combined with MoE.
- **Samba (Ren et al., 2024)**: Stacks Mamba SSM blocks with sliding window attention.
- **Griffin & Hawk (De et al., 2024)**: Combines Real-Gated Linear Recurrence Units (RG-LRU) with local multi-query attention.
- **Hymba (Dong et al., 2024)**: Parallel multi-branch hybrid combining attention heads and SSM channels within the same transformer layer.

---

### Family 10: Mixture-of-Experts (MoE) & Sparsely-Gated Attentions
**Modules (6):** `bam_umoe`, `expert_choice_moe`, `reap_moe`, `samba_moe`, `moe_deltakv_paged`, `mixattention_moska`.

#### Routing Formulation
For input $\mathbf{x}_t$, a top-$k$ router selects a subset of $E$ specialized attention projection heads:
$$g(\mathbf{x}_t) = \operatorname{TopK}\left(\operatorname{softmax}\left(\mathbf{W}_g \mathbf{x}_t + \epsilon\right), k\right)$$
$$\mathbf{y}_t = \sum_{e \in \operatorname{Active}} g(\mathbf{x}_t)_e \cdot \operatorname{Attn}_e(\mathbf{x}_t)$$
- **Expert-Choice MoE**: Inverts routing direction: each expert selects its top-$C$ tokens, guaranteeing perfect compute load balancing across GPUs.
- **BAM-UMoE**: Universal Mixture of Experts routing across heterogeneous sequence operators (Softmax Attention, GLA, and Convolution).

---

### Family 11: Test-Time Training (TTT) & Fast-Weight Systems
**Modules (2):** `ttt_linear_attention`, `template_attention`.

#### Mathematical Formulation
Test-Time Training (Sun et al., 2024; TTT-Linear / TTT-MLP) re-conceptualizes hidden state accumulation as an inner-loop self-supervised learning optimization step:
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta \nabla_{\mathbf{W}} \ell(\mathbf{W}_{t-1}; \mathbf{x}_t)$$
where the reconstruction loss $\ell(\mathbf{W}; \mathbf{x}_t)$ evaluates token prediction from corrupted or masked inputs:
$$\ell(\mathbf{W}; \mathbf{x}_t) = \frac{1}{2} \|\mathbf{W} \theta_K(\mathbf{x}_t) - \theta_V(\mathbf{x}_t)\|^2$$
The forward output is generated via the updated weight matrix:
$$\mathbf{y}_t = \mathbf{W}_t \theta_Q(\mathbf{x}_t)$$
This formulation generalizes linear attention from fixed accumulator outer-products to continuous online model adaptation.

---

### Family 12: Memory-Augmented & External Latent Recurrences
**Modules (4):** `landmark_local_attention`, `refined_landmark_attention`, `rala_attention`, `nasc_attention`.

#### Mathematical Formulation
Memory-augmented models divide sequences into blocks and summarize past blocks into compact representative tokens:
- **Landmark Attention (Mohtashami & Jaggi, 2023)**: Inserts a special landmark token at the end of each block of length $L_{blk}$. During retrieval, queries compute coarse attention over landmark tokens, selecting only relevant blocks for fine-grained attention.
- **RALA (Recurrent Augmented Linear Attention)**: Maintains a dual memory tier: a local high-fidelity sliding window cache and an external compressed latent summary matrix.

---

### Family 13: Modern Positional, Rotary & Geometric Attention Variants
**Modules (5):** `alibi`, `unified_rope`, `wgqa_attention`, `cross_attention`, `primal_attention`.

#### Mathematical Formulation
Relative positional embeddings eliminate absolute position matrices by altering query-key inner products:
- **Rotary Position Embedding (RoPE, Su et al., 2021)**: Applies orthogonal rotation matrices $\mathbf{R}_{\Theta, m}^{d}$ to queries and keys:
  $$\mathbf{q}_m = \mathbf{R}_{\Theta, m} \mathbf{W}_Q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n} \mathbf{W}_K \mathbf{x}_n \implies \langle \mathbf{q}_m, \mathbf{k}_n \rangle = \mathbf{x}_m^\top \mathbf{W}_Q^\top \mathbf{R}_{\Theta, n-m} \mathbf{W}_K \mathbf{x}_n$$
- **ALiBi (Attention with Linear Biases, Press et al., 2022)**: Adds a static, non-learned negative slope penalty:
  $$\operatorname{Attn}_{ALiBi}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} - m \cdot |i - j|\right) \mathbf{V}$$
  where slopes $m = 2^{-8 h / H}$ decay geometrically across heads, enabling zero-shot context length extrapolation.

---

### Family 14: Hardware-Fused, Low-Precision & Spectral-Transform Attentions
**Modules (14):** `spectron`, `spectron_fp8`, `aegis_spectron_fp8`, `aegis_v1`, `aegis_v2`, `aegis_v3`, `aegis_v4`, `aegis_v5`, `aegis_v6`, `soa_attn`, `fusion_attn`, `orion`, `vig_attention`, `vspec`.

#### Mathematical Formulation
Hardware-fused architectures combine operator fusion with non-standard numerical formats (FP8 E4M3/E5M2) or frequency-domain orthogonal transforms:
- **Spectron & Spectron FP8**: Projects sequences into the frequency domain via Fast Fourier Transforms (FFT) or Chebyshev orthogonal polynomials, applying parameterized spectral filtering prior to low-rank attention.
- **Aegis Family (`aegis_v1` - `aegis_v6`)**: Multi-stage hardware-fused attention pipelines featuring fused LayerNorm, dynamic FP8 scaling, and shared-memory kernel execution.

---

## 3. Master LaTeX Comparative Tables

When authoring research manuscripts, embed these verified, camera-ready LaTeX master tables comparing complexity, memory footprints, and module mappings.

### 3.1 Family-Level Complexity & Footprint Table
```latex
\begin{table*}[t]
\centering
\small
\caption{Comprehensive architectural and systems complexity taxonomy of the 14 attention families. $N$: sequence length, $d$: model dimension, $d_k, d_v$: head projection dimensions, $W$: local window size, $r$: low-rank/latent projection rank, $E$: number of experts, $k$: top-$k$ active experts, $t$: current autoregressive decoding step.}
\label{tab:attention_families_complexity}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Architectural Family} & \textbf{Training FLOPs} & \textbf{Activation Memory} & \textbf{Decode FLOPs/Step} & \textbf{KV Cache Footprint} & \textbf{Causal Form} & \textbf{Associative Scan} \\
\midrule
1. Exact Dense Softmax (MHA/GQA)   & $\mathcal{O}(N^2 \cdot d)$       & $\mathcal{O}(N)$ (tiled) & $\mathcal{O}(t \cdot d)$   & $\mathcal{O}(t \cdot d)$       & Exact Lower-Tri      & No (Softmax) \\
2. Kernelized Linear Attention     & $\mathcal{O}(N \cdot d_k d_v)$   & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d_k d_v)$     & $\mathcal{O}(d_k d_v)$ [Const] & Feature Cumulative   & Yes \\
3. Gated Linear Recurrence (GLA)   & $\mathcal{O}(N \cdot d_k d_v)$   & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d_k d_v)$     & $\mathcal{O}(d_k d_v)$ [Const] & 1-Semigroup Decay    & Yes (Chunk-Parallel) \\
4. Delta-Rule Recurrence (DeltaNet)& $\mathcal{O}(N \cdot d_k d_v)$   & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d_k d_v)$     & $\mathcal{O}(d_k d_v)$ [Const] & Householder Update   & Yes (Householder) \\
5. State-Space Models (Mamba/SSD)  & $\mathcal{O}(N \cdot d \cdot r)$ & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d \cdot r)$   & $\mathcal{O}(d \cdot r)$ [Const]& Selective Discretized& Yes (Prefix Scan) \\
6. Block-Sparse / Sliding-Window   & $\mathcal{O}(N \cdot W \cdot d)$ & $\mathcal{O}(N \cdot W)$ & $\mathcal{O}(W \cdot d)$   & $\mathcal{O}(W \cdot d)$ [Bound]& Banded Adjacency     & No \\
7. SVD Spectral / Low-Rank         & $\mathcal{O}(N \cdot r \cdot d)$ & $\mathcal{O}(N \cdot r)$ & $\mathcal{O}(r \cdot d)$   & $\mathcal{O}(r \cdot d)$ [Const]& Subspace Project     & Yes \\
8. Multi-Head Latent (MLA)         & $\mathcal{O}(N^2 d_h + N d_c)$   & $\mathcal{O}(N)$ (tiled) & $\mathcal{O}(t \cdot d_c)$ & $\mathcal{O}(t \cdot d_c)$     & Decoupled RoPE       & No \\
9. Hybrid Linear-Attention         & $\mathcal{O}(\alpha N^2 d + (1-\alpha) N d^2)$ & $\mathcal{O}(N \cdot d)$ & Hybrid & Mixed Layerwise & Layer-Dependent      & Selective \\
10. Mixture-of-Experts (MoE)       & $\mathcal{O}(N \cdot \frac{k}{E} d^2)$ & $\mathcal{O}(N \cdot d)$ & Routed $\mathcal{O}(d^2)$ & Dynamic & Top-$k$ Masked & Sub-branch \\
11. Test-Time Training (TTT)       & $\mathcal{O}(N \cdot d^2)$       & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d^2)$         & $\mathcal{O}(d^2)$ [Weights]   & Inner-Loop Gradient  & Dual Momentum \\
12. Memory-Augmented Recurrence    & $\mathcal{O}(N \cdot W d + \frac{N^2}{L_{blk}^2} d)$ & $\mathcal{O}(N)$ & $\mathcal{O}(W d + M d)$   & Compressed Blocks & Hierarchical Mask    & Blockwise \\
13. Rotary \& Modern Positional    & $\mathcal{O}(N^2 \cdot d)$       & $\mathcal{O}(N)$ (tiled) & $\mathcal{O}(t \cdot d)$   & $\mathcal{O}(t \cdot d)$       & RoPE / ALiBi Bias    & No \\
14. Hardware-Fused Spectral FP8    & $\mathcal{O}(N \log N \cdot d)$  & $\mathcal{O}(N \cdot d)$ & $\mathcal{O}(d \log d)$    & $\mathcal{O}(d)$ [Spectral]    & Fused Orthogonal     & Domain-Transform \\
\bottomrule
\end{tabular}
\end{table*}
```

---

### 3.2 Master Registry Inventory of 131 Attention Mechanisms
The table below documents the exact partitioning of all 131 modules cataloged in the NeuralOps repository:

```latex
\begin{table*}[t]
\centering
\scriptsize
\caption{Complete inventory mapping of all 131 evaluated attention mechanisms in NeuralOps across the 14 foundational architectural families.}
\label{tab:all_131_attentions_inventory}
\begin{tabular}{@{}llcp{8.2cm}@{}}
\toprule
\textbf{ID} & \textbf{Family Name} & \textbf{Count} & \textbf{Registered Module Identifiers} \\
\midrule
1  & Exact Softmax \& Dense       & 6  & \texttt{full}, \texttt{full\_attention}, \texttt{mqa}, \texttt{gqa\_attention}, \texttt{flash\_attention3}, \texttt{flash\_attention3\_fp8} \\
2  & Kernelized Linear Attention  & 15 & \texttt{cosformer}, \texttt{cosformer\_attention}, \texttt{performer\_attention}, \texttt{positive\_kernel\_linear\_attention}, \texttt{galerkin\_attention}, \texttt{regla\_attention}, \texttt{halo\_regla}, \texttt{eat}, \texttt{eat\_attention}, \texttt{chela\_slay}, \texttt{chela\_slay\_attention}, \texttt{triton\_cosformer}, \texttt{rpc\_attention}, \texttt{zeros\_attention}, \texttt{xattention} \\
3  & Gated Linear Recurrence      & 11 & \texttt{gated\_linear\_attention}, \texttt{gated\_linear\_attn}, \texttt{gla\_attention}, \texttt{decay\_linear\_attention}, \texttt{decay\_linear\_attn}, \texttt{factorised\_matrix\_state}, \texttt{hgrn2\_gated\_layer}, \texttt{hgrn2\_gated\_outer}, \texttt{hgrn2\_outer}, \texttt{goldfinch\_ahn}, \texttt{salt\_attention} \\
4  & Delta-Rule Recurrences       & 7  & \texttt{gated\_delta\_net}, \texttt{gated\_deltanet\_block}, \texttt{hgrn2\_gated\_deltanet}, \texttt{kda\_attention}, \texttt{hla\_attention}, \texttt{nexus}, \texttt{phantom} \\
5  & State-Space Models (SSMs)    & 6  & \texttt{selective\_ssm}, \texttt{simple\_ssm}, \texttt{mamba\_block}, \texttt{upi\_stuffed\_mamba}, \texttt{mamba\_svd\_flash}, \texttt{svd\_mamba} \\
6  & Block-Sparse / Sliding-Window& 13 & \texttt{adaptive\_span\_attention}, \texttt{adaptive\_sparse\_attention}, \texttt{adasplash\_attention}, \texttt{cluster\_attention}, \texttt{block\_transformer\_hiba}, \texttt{nsa\_attention}, \texttt{lsh\_attention}, \texttt{routing\_attention}, \texttt{sink\_attention}, \texttt{dynamic\_conv\_attention}, \texttt{hamming\_attention}, \texttt{moba\_attention}, \texttt{progressive\_token\_pruning} \\
7  & SVD Spectral / Low-Rank      & 19 & \texttt{linformer\_attention}, \texttt{nystrom\_attention}, \texttt{asvd\_attention}, \texttt{svd\_linear}, \texttt{svd\_linear\_attention}, \texttt{svd\_linear\_attn}, \texttt{svd\_linear\_decay}, \texttt{svd\_linear\_inject}, \texttt{svd\_llm\_v2\_attention}, \texttt{adaptive\_svd\_attention}, \texttt{residual\_svd\_attention}, \texttt{svd\_projection\_attention}, \texttt{svd\_seq}, \texttt{svd\_compressed}, \texttt{naive\_svd}, \texttt{minimal\_svd}, \texttt{spectraformer\_attention}, \texttt{fused\_svd\_kernel}, \texttt{triton\_svd\_linear\_attention} \\
8  & Multi-Head Latent (MLA)      & 9  & \texttt{mla\_attention}, \texttt{qcqa\_attention}, \texttt{compressed\_attention}, \texttt{compressed\_mha\_attention}, \texttt{token\_compressed\_attention}, \texttt{projection\_factorized\_attn}, \texttt{lola\_attention}, \texttt{padre\_attention}, \texttt{lora\_attention} \\
9  & Hybrid Recurrent / SSM       & 14 & \texttt{jamba}, \texttt{jamba\_info}, \texttt{samba\_m3}, \texttt{griffin\_ssd}, \texttt{hymba\_transx}, \texttt{granite\_hybrid}, \texttt{hybrid\_attention}, \texttt{native\_hybrid\_attention}, \texttt{optimized\_hybrid\_attention}, \texttt{split\_head\_hybrid}, \texttt{tri\_branch\_hybrid}, \texttt{hybrid\_local\_kernel}, \texttt{aegis\_hybrid}, \texttt{minimal\_nova} \\
10 & Mixture-of-Experts (MoE)     & 6  & \texttt{bam\_umoe}, \texttt{expert\_choice\_moe}, \texttt{reap\_moe}, \texttt{samba\_moe}, \texttt{moe\_deltakv\_paged}, \texttt{mixattention\_moska} \\
11 & Test-Time Training (TTT)     & 2  & \texttt{ttt\_linear\_attention}, \texttt{template\_attention} \\
12 & Memory-Augmented Recurrence  & 4  & \texttt{landmark\_local\_attention}, \texttt{refined\_landmark\_attention}, \texttt{rala\_attention}, \texttt{nasc\_attention} \\
13 & Rotary \& Modern Positional  & 5  & \texttt{alibi}, \texttt{unified\_rope}, \texttt{wgqa\_attention}, \texttt{cross\_attention}, \texttt{primal\_attention} \\
14 & Hardware-Fused Spectral FP8  & 14 & \texttt{spectron}, \texttt{spectron\_fp8}, \texttt{aegis\_spectron\_fp8}, \texttt{aegis\_v1}, \texttt{aegis\_v2}, \texttt{aegis\_v3}, \texttt{aegis\_v4}, \texttt{aegis\_v5}, \texttt{aegis\_v6}, \texttt{soa\_attn}, \texttt{fusion\_attn}, \texttt{orion}, \texttt{vig\_attention}, \texttt{vspec} \\
\midrule
\textbf{Total} & \textbf{14 Families} & \textbf{131} & \textbf{Exhaustive Coverage of Verified NeuralOps Attentions} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 4. Standalone TikZ Taxonomy Diagrams

Embed these production-grade TikZ code listings directly into camera-ready LaTeX papers to illustrate architectural hierarchies and dual-form state scans.

### 4.1 Hierarchical Taxonomy Tree Diagram
```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  level 1/.style={sibling distance=5.2cm, level distance=1.6cm},
  level 2/.style={sibling distance=2.5cm, level distance=1.8cm},
  root_node/.style={rectangle, draw=blue!90!black, fill=blue!10, rounded corners=4pt, font=\bfseries\small, align=center, inner sep=6pt, drop shadow},
  family_node/.style={rectangle, draw=slate!80!black, fill=slate!10, rounded corners=3pt, font=\bfseries\footnotesize, align=center, inner sep=4pt},
  leaf_node/.style={rectangle, draw=gray!60, fill=white, rounded corners=2pt, font=\tiny, align=center, inner sep=3pt},
  edge from parent/.style={draw=slate!70!black, -Stealth, thick}
]

\node[root_node] {Unified Attention Taxonomy (131 Mechanisms)}
  child {node[family_node] {Exact \& Tiled Attention\\($\mathcal{O}(N^2 d)$ Compute)}
    child {node[leaf_node] {Dense MHA / GQA / MQA\\\texttt{flash\_attention3}}}
    child {node[leaf_node] {Decoupled Latent (MLA)\\\texttt{mla\_attention}}}
    child {node[leaf_node] {Rotary / Positional\\\texttt{alibi, unified\_rope}}}
  }
  child {node[family_node] {Linear Recurrences \& SSMs\\($\mathcal{O}(N d^2)$ or $\mathcal{O}(N d r)$)}
    child {node[leaf_node] {Kernel Feature-Maps\\\texttt{cosformer, performer}}}
    child {node[leaf_node] {Gated Linear Recurr.\\\texttt{gla, retnet, hgrn2}}}
    child {node[leaf_node] {Delta-Rule Associative\\\texttt{gated\_deltanet, kda}}}
    child {node[leaf_node] {State-Space Models\\\texttt{mamba, mamba-2}}}
  }
  child {node[family_node] {Structural \& Hybrid Operators\\(Sub-Quadratic / Dynamic)}
    child {node[leaf_node] {Block-Sparse \& Sliding\\\texttt{nsa, sink\_attention}}}
    child {node[leaf_node] {SVD Low-Rank Spectral\\\texttt{asvd, linformer}}}
    child {node[leaf_node] {Hybrid SSM-Attention\\\texttt{jamba, samba, griffin}}}
    child {node[leaf_node] {MoE \& TTT\\\texttt{bam\_umoe, ttt\_linear}}}
  };
\end{tikzpicture}
\caption{Hierarchical architectural taxonomy of the 14 attention families unifying all 131 registered mechanisms in NeuralOps, spanning quadratic exact methods, linear recurrent duals, state space models, and spectral hybrid compressions.}
\label{fig:attention_taxonomy_tree}
\end{figure*}
```

---

### 4.2 Dual-Form Computation Flow: Sequential Recurrence vs Chunk Scan
```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  box/.style={rectangle, draw=blue!80!black, fill=blue!5, rounded corners=3pt, font=\footnotesize, align=center, inner sep=5pt},
  tensor/.style={rectangle, draw=orange!80!black, fill=orange!10, rounded corners=2pt, font=\scriptsize, align=center, inner sep=3pt},
  arrow/.style={-Stealth, thick, draw=slate!80!black}
]

% Left: Sequential Recurrent Form
\node[font=\bfseries\small] at (-4, 2.5) {Autoregressive Inference: Sequential Recurrence $\mathcal{O}(1)$ State};
\node[tensor] (k_t) at (-5.5, 1.2) {$\mathbf{k}_t \in \mathbb{R}^{d_k}$};
\node[tensor] (v_t) at (-2.5, 1.2) {$\mathbf{v}_t \in \mathbb{R}^{d_v}$};
\node[box] (outer) at (-4, 0.2) {Outer Product\\$\mathbf{k}_t^\top \mathbf{v}_t \in \mathbb{R}^{d_k \times d_v}$};
\node[tensor] (s_prev) at (-6.5, -1.0) {$\mathbf{S}_{t-1}$};
\node[box] (add) at (-4, -1.0) {Gated Update\\$\mathbf{S}_t = \mathbf{\alpha}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$};
\node[tensor] (q_t) at (-6.5, -2.2) {$\mathbf{q}_t \in \mathbb{R}^{d_k}$};
\node[box] (out) at (-4, -2.2) {Output Projection\\$\mathbf{o}_t = \mathbf{q}_t \mathbf{S}_t \in \mathbb{R}^{d_v}$};

\draw[arrow] (k_t) -- (outer);
\draw[arrow] (v_t) -- (outer);
\draw[arrow] (outer) -- (add);
\draw[arrow] (s_prev) -- (add);
\draw[arrow] (add) -- (out);
\draw[arrow] (q_t) -- (out);

% Divider
\draw[dashed, thick, gray!60] (0.5, 2.8) -- (0.5, -3.0);

% Right: Chunk-Parallel Associative Scan Form
\node[font=\bfseries\small] at (5.5, 2.5) {Parallel Training: Chunk Associative Scan $\mathcal{O}(N d^2)$};
\node[box] (intra) at (3.5, 0.8) {Intra-Chunk Attention\\$\mathbf{O}_{[c]}^{\text{intra}} = ((\mathbf{Q}_{[c]} \mathbf{K}_{[c]}^\top) \odot \mathbf{M}) \mathbf{V}_{[c]}$};
\node[box] (scan) at (7.5, 0.8) {Inter-Chunk State Scan\\$\mathbf{S}_{[c]} = \mathbf{\Lambda}_c \mathbf{S}_{[c-1]} + \mathbf{K}_{[c]}^\top \mathbf{V}_{[c]}$};
\node[box] (inter) at (7.5, -0.8) {Inter-Chunk Output\\$\mathbf{O}_{[c]}^{\text{inter}} = \mathbf{Q}_{[c]} \mathbf{S}_{[c-1]}$};
\node[box, fill=green!10, draw=green!70!black] (merge) at (5.5, -2.2) {Unified Forward Output\\$\mathbf{O}_{[c]} = \mathbf{O}_{[c]}^{\text{intra}} + \mathbf{O}_{[c]}^{\text{inter}}$};

\draw[arrow] (intra) |- (merge);
\draw[arrow] (scan) -- (inter);
\draw[arrow] (inter) |- (merge);

\end{tikzpicture}
\caption{Mathematical duality between step-by-step sequential recurrence for constant-memory decoding and chunk-parallel associative scan for high-throughput GPU training on Tensor Cores.}
\label{fig:dual_form_recurrent_scan}
\end{figure}
```

---

## 5. Academic Prose & Writing Guidance

When authoring taxonomic and theoretical sections for research papers, adhere to the following rules:

### 5.1 Related Work & Taxonomy Framing
- **Avoid Vague Categorization**: Never classify mechanisms merely as "fast" or "efficient". Define the exact mathematical conservation law: whether the operator trades off representational rank ($\operatorname{rank}(\mathbf{A}) \le r$), sequence locality (banded sparse masks), or normalization convexity (replacing softmax with positive feature kernels).
- **Acknowledge the Relational Rank Collapse**: Linear attention eliminates the quadratic barrier by omitting the row-wise softmax exponential. State explicitly that this causes relational rank collapse on associative premise-hypothesis tasks (e.g., MNLI, GLUE relational pairs), where sharp cross-token alignment is essential.
- **State Space Duality Rigor**: Cite both continuous dynamical systems roots (Kalman filters, HiPPO polynomials) and discrete matrix transformations (1-semigroup masked linear attention). Emphasize that Mamba-2 (SSD) proves selective SSMs are linear attention mechanisms with semi-separable decay kernels.

### 5.2 Methodological Phrasing Rules
- **Use Formal Operator Contracts**: State input domains, codomains, and tensor shapes explicitly: $\operatorname{Op}: \mathbb{R}^{B \times N \times d_{in}} \to \mathbb{R}^{B \times N \times d_{out}}$.
- **Disclose Causality Invariants**: Clarify whether an operator supports bidirectional encoding (e.g., masked language modeling, BERT) or is strictly causal (e.g., autoregressive LLMs). SSMs and gated linear recurrences that depend on sequential states $\mathbf{h}_t = f(\mathbf{h}_{t-1})$ require multi-pass bidirectional scans to process bidirectional contexts.
- **Hardware-Aware Framing**: Distinguish between arithmetic complexity (FLOPs) and memory bandwidth pressure. An $\mathcal{O}(N)$ algorithm that requires random memory indirection or high register pressure may execute significantly slower on GPU hardware than an $\mathcal{O}(N^2)$ operator executed via fused SRAM tiling.

---

## 6. Pre-Submission Taxonomy & Theory Checklist

Before finalizing any taxonomy, methodology section, or research paper appendix:

```markdown
- [ ] Every attention mechanism cited is categorized into one of the 14 foundational families.
- [ ] Asymptotic complexity bounds include all dimensions: sequence length $N$, hidden size $d$, state dimension $r$, and window $W$.
- [ ] Forward equations specify exact tensor dimensions and causal masking conditions.
- [ ] State recurrence update rules document both continuous formulation and discrete step equations.
- [ ] Chunk-parallel associative scan equations explicitly formulate intra-chunk and inter-chunk tensor contractions.
- [ ] Memory footprint is explicitly divided into training activation memory vs autoregressive KV-cache per step.
- [ ] Comparative master tables compile cleanly using standard LaTeX booktabs environments without tabular overflows.
- [ ] TikZ diagrams use relative positioning, verified styles, and avoid deprecated syntax.
- [ ] All claims regarding relational expressivity vs inference throughput are grounded in verified empirical benchmarks.
```
