---
name: paper-linear-recurrent-associative-scans
description: "Author mathematical formulations, proofs, and algorithmic sections on parallel associative prefix scans (Blelloch), chunk-parallel state recurrences, and dual linear attention forms for research papers"
version: "1.0.0"
tags: ["neuralops", "paper", "theory", "scans", "associative", "recurrent", "prefix-sum", "parallel"]
---

# Skill: Paper Linear Recurrent & Associative Scans

## Overview & Scope
Use this skill when authoring, formalizing, deriving, verifying, or typeset-polishing publication-grade theoretical sections, algorithmic boxes, mathematical proofs, and systems analyses for machine learning conference papers (NeurIPS, ICLR, ICML, MLSys) and doctoral dissertations focusing on:

1. **First-Order Linear Recurrences & Associative Binary Operators**: Algebraic formulation of sequential hidden states into generalized monoid operations permitting parallel evaluation.
2. **Parallel Associative Prefix Scans (Blelloch & Hillis-Steele)**: Algorithmic tree reductions reducing sequential $\mathcal{O}(N)$ dependency graphs to $\mathcal{O}(\log N)$ parallel circuit depth across GPU thread warps and cooperative thread arrays (CTAs).
3. **Chunk-Parallel State Recurrences & Dual Linear Attention**: Hybrid algorithms partitioning sequences of length $N$ into chunks of size $C$ to balance compute intensity (intra-chunk Tensor Core GEMM) with streaming state carry-over (inter-chunk recurrence).
4. **Delta-Rule Associative Memory (DeltaNet, GDN)**: Error-driven associative memory updates derived from online gradient descent on key-value retrieval objectives, overcoming the saturation limit of classical linear attention.
5. **Architectural Cross-Comparison**: Rigorous comparative analysis across state-of-the-art linear recurrent architectures: Gated Linear Attention (GLA), DeltaNet / Gated DeltaNet, Retention Network (RetNet), and RWKV-v6 (Finch).
6. **Publication-Ready Typesetting Standards**: Verified AMS-LaTeX theorem environments, proof blocks with explicit step justifications, and standard `algorithm2e` / `algorithmicx` algorithmic listings.

---

## 1. Mathematical Foundations of Linear Recurrence & Associative Operators

### 1.1 The General First-Order Linear Recurrence
Consider a generic discrete-time linear dynamical system operating on a sequence of inputs $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^{d_{\text{in}}}$:
$$\mathbf{S}_t = \mathbf{A}_t \mathbf{S}_{t-1} + \mathbf{B}_t \mathbf{X}_t, \quad t \in \{1, \dots, N\}$$
where:
- $\mathbf{S}_t \in \mathcal{S}$ represents the hidden memory state at step $t$ (either a vector in $\mathbb{R}^{d}$ or a matrix in $\mathbb{R}^{d_k \times d_v}$).
- $\mathbf{A}_t \in \mathcal{A}$ is the transition / decay operator (scalar decay $\gamma_t \in \mathbb{R}$, diagonal matrix $\operatorname{diag}(\mathbf{a}_t) \in \mathbb{R}^{d \times d}$, or full matrix $\mathbf{A}_t \in \mathbb{R}^{d \times d}$).
- $\mathbf{B}_t \mathbf{X}_t \in \mathcal{S}$ is the input projection driving the state update (e.g., outer product $\mathbf{k}_t \mathbf{v}_t^\top$ or rank-1 update $\mathbf{k}_t \otimes \mathbf{v}_t$).
- $\mathbf{S}_0 \in \mathcal{S}$ denotes the initial state (typically $\mathbf{0}$).

Unrolling this recurrence sequentially yields:
$$\mathbf{S}_t = \sum_{j=1}^t \left( \prod_{\tau=j+1}^t \mathbf{A}_\tau \right) \mathbf{B}_j \mathbf{X}_j$$
where the product of operators follows reverse chronological ordering:
$$\prod_{\tau=j+1}^t \mathbf{A}_\tau \triangleq \mathbf{A}_t \mathbf{A}_{t-1} \cdots \mathbf{A}_{j+1}$$
with the empty product defined as the identity operator $\mathbf{I}$.

### 1.2 Binary Associative Operator $\otimes$
While the sequential unrolling has computational depth $\mathcal{O}(N)$, the linear recurrence can be recast as an all-prefix-sums problem over an algebraic semigroup $(\mathcal{G}, \otimes)$.

Let each state transition be represented by a tuple $u_t = (\mathbf{A}_t, \mathbf{U}_t) \in \mathcal{A} \times \mathcal{S}$, where $\mathbf{U}_t = \mathbf{B}_t \mathbf{X}_t$. The composition of step $i$ followed by step $j$ (where $i < j$) is defined by the binary operator $\otimes$:
$$(\mathbf{A}_i, \mathbf{U}_i) \otimes (\mathbf{A}_j, \mathbf{U}_j) \triangleq (\mathbf{A}_j \mathbf{A}_i, \, \mathbf{A}_j \mathbf{U}_i + \mathbf{U}_j)$$

Under this operator, applying $u_1 \otimes u_2 \otimes \dots \otimes u_t$ to an initial state $\mathbf{S}_0 = \mathbf{0}$ directly produces the cumulative transition and updated state $(\prod_{\tau=1}^t \mathbf{A}_\tau, \mathbf{S}_t)$.

### 1.3 Formal Proof of Associativity
A binary operator $\otimes$ permits parallel prefix scan if and only if it is associative.

$$\begin{aligned}
\textbf{Theorem 1 (Semigroup Associativity)}: \quad &\text{Let } u_1 = (\mathbf{A}_1, \mathbf{U}_1), \, u_2 = (\mathbf{A}_2, \mathbf{U}_2), \, u_3 = (\mathbf{A}_3, \mathbf{U}_3) \in \mathcal{A} \times \mathcal{S}. \\
&\text{Then } (u_1 \otimes u_2) \otimes u_3 = u_1 \otimes (u_2 \otimes u_3).
\end{aligned}$$

$$\begin{aligned}
\textit{Proof.} \quad &\text{Evaluating the left-hand grouping: } \\
(u_1 \otimes u_2) &= (\mathbf{A}_2 \mathbf{A}_1, \, \mathbf{A}_2 \mathbf{U}_1 + \mathbf{U}_2) \\
(u_1 \otimes u_2) \otimes u_3 &= ((\mathbf{A}_2 \mathbf{A}_1), \, (\mathbf{A}_2 \mathbf{U}_1 + \mathbf{U}_2)) \otimes (\mathbf{A}_3, \mathbf{U}_3) \\
&= (\mathbf{A}_3 (\mathbf{A}_2 \mathbf{A}_1), \, \mathbf{A}_3 (\mathbf{A}_2 \mathbf{U}_1 + \mathbf{U}_2) + \mathbf{U}_3) \\
&= (\mathbf{A}_3 \mathbf{A}_2 \mathbf{A}_1, \, \mathbf{A}_3 \mathbf{A}_2 \mathbf{U}_1 + \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3) \\[1em]
&\text{Evaluating the right-hand grouping: } \\
(u_2 \otimes u_3) &= (\mathbf{A}_3 \mathbf{A}_2, \, \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3) \\
u_1 \otimes (u_2 \otimes u_3) &= (\mathbf{A}_1, \mathbf{U}_1) \otimes (\mathbf{A}_3 \mathbf{A}_2, \, \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3) \\
&= ((\mathbf{A}_3 \mathbf{A}_2) \mathbf{A}_1, \, (\mathbf{A}_3 \mathbf{A}_2) \mathbf{U}_1 + (\mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3)) \\
&= (\mathbf{A}_3 \mathbf{A}_2 \mathbf{A}_1, \, \mathbf{A}_3 \mathbf{A}_2 \mathbf{U}_1 + \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3)
\end{aligned}$$

Because matrix multiplication and vector addition are associative and distributive in standard linear algebra over $\mathbb{R}$, both groupings produce identical tuples. Thus, $(u_1 \otimes u_2) \otimes u_3 = u_1 \otimes (u_2 \otimes u_3)$, completing the proof. $\blacksquare$

### 1.4 Algebraic Monoid & Identity Element
The structure $(\mathcal{G}, \otimes)$ forms a **monoid** with the two-sided identity element:
$$e = (\mathbf{I}, \, \mathbf{0})$$
Verify:
$$(e \otimes u) = (\mathbf{I}, \mathbf{0}) \otimes (\mathbf{A}, \mathbf{U}) = (\mathbf{A} \mathbf{I}, \mathbf{A} \mathbf{0} + \mathbf{U}) = (\mathbf{A}, \mathbf{U}) = u$$
$$(u \otimes e) = (\mathbf{A}, \mathbf{U}) \otimes (\mathbf{I}, \mathbf{0}) = (\mathbf{I} \mathbf{A}, \mathbf{I} \mathbf{U} + \mathbf{0}) = (\mathbf{A}, \mathbf{U}) = u$$

---

## 2. Parallel Associative Prefix Scans: Blelloch vs. Hillis-Steele

The associativity of $\otimes$ decouples execution from strict temporal sequentiality. Rather than computing step-by-step in $\mathcal{O}(N)$ serial steps, tree-based parallel scans compute all prefix states $\mathbf{S}_1, \dots, \mathbf{S}_N$ concurrently.

```
═════════════════════════════════════════════════════════════════════════════════════════
                   BLELLOCH WORK-EFFICIENT PARALLEL PREFIX SCAN
═════════════════════════════════════════════════════════════════════════════════════════
   1. UP-SWEEP (REDUCE) PHASE                           2. DOWN-SWEEP PHASE
   Span: O(log2 N) steps, Work: O(N) ops                Span: O(log2 N) steps, Work: O(N) ops

    x1   x2   x3   x4   x5   x6   x7   x8               x1   x2   x3   x4   x5   x6   x7   x8
     \   /     \   /     \   /     \   /                 │    │    │    │    │    │    │    │
      x12       x34       x56       x78                  │    │    │    │    │    │    │   [0] (root=id)
        \       /           \       /                    │    │    │    │    │    │   /    \
          x1-4                x5-8                       │    │    │   [0]   │    │  x1-4   \
             \               /                           │    │   /    \     │    │  /       \
                   x1-8                                  │   [0] x12    \    │   x1-4 x1-6    \
                                                         │  /  \  │      \   │  /  \   │       \
                                                        s0 s1  s2 s3     s4 s5 s6  s7      s8
═════════════════════════════════════════════════════════════════════════════════════════
```

### 2.1 Blelloch Work-Efficient Scan
The Blelloch (1990) scan algorithm operates on an array of length $N$ (assumed a power of 2 without loss of generality) in two balanced binary tree sweeps:

#### Phase 1: Up-Sweep (Parallel Reduction)
Builds a reduction tree from leaves to root in $d = \log_2 N$ steps. At step $k \in \{0, \dots, \log_2 N - 1\}$ with stride $2^{k+1}$:
$$\mathbf{x}[j] \leftarrow \mathbf{x}[j - 2^k] \otimes \mathbf{x}[j], \quad \forall j \in \{2^{k+1} - 1, 2 \cdot 2^{k+1} - 1, \dots, N-1\}$$

- **Number of operations at step $k$**: $N / 2^{k+1}$ applications of $\otimes$.
- **Total Up-Sweep Operations**:
  $$\sum_{k=0}^{\log_2 N - 1} \frac{N}{2^{k+1}} = N \sum_{m=1}^{\log_2 N} 2^{-m} = N \left(1 - \frac{1}{N}\right) = N - 1$$

#### Phase 2: Down-Sweep (Prefix Distribution)
The root element is set to the algebraic identity: $\mathbf{x}[N-1] \leftarrow (\mathbf{I}, \mathbf{0})$. The tree is traversed downward from root to leaves in $\log_2 N$ steps. At step $k \in \{\log_2 N - 1, \dots, 0\}$ with stride $2^{k+1}$:
For each node index $j \in \{2^{k+1} - 1, 2 \cdot 2^{k+1} - 1, \dots, N-1\}$:
$$\begin{aligned}
\mathbf{t} &\leftarrow \mathbf{x}[j - 2^k] \\
\mathbf{x}[j - 2^k] &\leftarrow \mathbf{x}[j] \\
\mathbf{x}[j] &\leftarrow \mathbf{t} \otimes \mathbf{x}[j]
\end{aligned}$$

- **Total Down-Sweep Operations**: Exactly $N - 1$ applications of $\otimes$.

$$\begin{aligned}
\textbf{Theorem 2 (Blelloch Complexity)}: \quad &\text{The Blelloch parallel prefix scan executes: } \\
&\text{1. Total Work: } W(N) = 2(N - 1) = \mathcal{O}(N) \text{ operations}. \\
&\text{2. Circuit Depth (Span): } D(N) = 2 \log_2 N = \mathcal{O}(\log N) \text{ parallel steps}.
\end{aligned}$$

Because $W(N) = \mathcal{O}(N)$, the Blelloch scan is asymptotically **work-efficient** (it performs no more total operations up to a constant factor than the sequential loop).

### 2.2 Comparison with Hillis-Steele Parallel Scan
The Hillis-Steele (1986) algorithm computes prefix sums in place in $\log_2 N$ synchronous steps:
At step $k \in \{0, \dots, \log_2 N - 1\}$:
$$\mathbf{x}[i] \leftarrow \mathbf{x}[i - 2^k] \otimes \mathbf{x}[i], \quad \forall i \ge 2^k$$

| Metric | Sequential Scan | Hillis-Steele Scan | Blelloch Scan |
| :--- | :---: | :---: | :---: |
| **Total Work** | $\mathcal{O}(N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ (**Work-Efficient**) |
| **Parallel Depth (Span)** | $\mathcal{O}(N)$ | $\log_2 N = \mathcal{O}(\log N)$ | $2 \log_2 N = \mathcal{O}(\log N)$ |
| **GPU Warp Divergence** | None | Low (single-pass) | High (two-pass tree traversal) |
| **Hardware Fit** | CPU / Autoregressive | Intra-Warp (SIMD `__shfl`) | Inter-Block / Large Sequence |

### 2.3 GPU Microarchitecture Mapping
On modern NVIDIA GPUs (e.g., Ada Lovelace SM 8.9, Hopper SM 90):
1. **Intra-Warp Reduction (32 Threads)**: Executed via register shuffle primitives `__shfl_up_sync(0xFFFFFFFF, val, delta)` in $\log_2(32) = 5$ instructions. Hillis-Steele is preferred inside warps because hardware execution is synchronous and SIMD threads do not suffer memory overhead.
2. **Intra-Block Reduction (1024 Threads / 32 Warps)**: Warp leaders store their 32 partial sums into Shared Memory (L1 cache / SRAM), where a single warp performs a second 32-element scan.
3. **Inter-Block Reduction ($N > 1024$)**: Executed either via multi-pass hierarchical dispatch or via single-pass decoupled look-back (Research standard in CUDA C++ / Triton).

---

## 3. Chunk-Parallel Decomposition & Hardware-Optimal Execution

While Blelloch scans possess $\mathcal{O}(\log N)$ theoretical depth, a naive full-sequence associative scan across all tokens suffers from low **Arithmetic Intensity** (FLOPs per byte transferred from HBM), becoming bound by DRAM bandwidth.

Modern high-throughput sequence models (GLA, DeltaNet, RetNet) resolve this using **Chunk-Parallel Decomposition**:
- Sequences of length $N$ are partitioned into $B = \lceil N / C \rceil$ non-overlapping chunks of length $C$ (typically $C \in \{64, 128\}$).
- **Intra-Chunk Computation**: Formulated as causal matrix multiplication $(\mathbf{Q}_{[c]} \mathbf{K}_{[c]}^\top) \mathbf{V}_{[c]}$, executed at maximum peak TFLOPS via Tensor Core HGEMMs.
- **Inter-Chunk Recurrence**: The recurrent state $\mathbf{S}_{[c]} \in \mathbb{R}^{d_k \times d_v}$ is updated across chunk boundaries via associative prefix scan over $B$ chunks.

```
═════════════════════════════════════════════════════════════════════════════════════════
                CHUNK-PARALLEL RECURRENT ATTENTION DECOMPOSITION
═════════════════════════════════════════════════════════════════════════════════════════
                     CHUNK c-1                            CHUNK c
             [Token (c-2)C+1 ... (c-1)C]           [Token (c-1)C+1 ... cC]
            ┌───────────────────────────┐         ┌───────────────────────────┐
            │   INTRA-CHUNK ATTENTION   │         │   INTRA-CHUNK ATTENTION   │
            │                           │         │                           │
            │   Q_[c-1]       K_[c-1]   │         │    Q_[c]         K_[c]    │
            │      │             │      │         │      │             │      │
            │      └──────┬──────┘      │         │      └──────┬──────┘      │
            │             ▼             │         │             ▼             │
            │  (Q K^T * M_causal * D)   │         │  (Q K^T * M_causal * D)   │
            │             │             │         │             │             │
            │             ▼ * V_[c-1]   │         │             ▼ * V_[c]     │
            │        O_[c-1]^intra      │         │         O_[c]^intra       │
            └─────────────┬─────────────┘         └─────────────┬─────────────┘
                          │                                     │
         CHUNK BOUNDARY   ▼                                     ▼
         STATE UPDATE: ┌─────────────┐   S_[c-1]   ┌──────────────────────────┐
                       │ S_[c-1]     ├────────────►│ O_[c]^inter = Q_[c]S_[c-1]│
                       └─────────────┘             └────────────┬─────────────┘
                                                                │
                                                                ▼
                                                   O_[c] = O_[c]^intra + O_[c]^inter
═════════════════════════════════════════════════════════════════════════════════════════
```

### 3.1 Mathematical Decomposition of Output Tokens
For token $t = (c-1)C + i$ located at index $i \in \{1, \dots, C\}$ within chunk $c \in \{1, \dots, B\}$:
$$\mathbf{o}_t = \sum_{j=1}^t \left(\prod_{\tau=j+1}^t \gamma_\tau\right) \left(\mathbf{q}_t \mathbf{k}_j^\top\right) \mathbf{v}_j$$

Splitting the summation into historical tokens ($j \le (c-1)C$) and current-chunk tokens ($(c-1)C < j \le t$):
$$\mathbf{o}_t = \underbrace{\sum_{j=(c-1)C+1}^t \left(\prod_{\tau=j+1}^t \gamma_\tau\right) \left(\mathbf{q}_t \mathbf{k}_j^\top\right) \mathbf{v}_j}_{\mathbf{o}_t^{\text{intra}}} + \underbrace{\sum_{j=1}^{(c-1)C} \left(\prod_{\tau=j+1}^t \gamma_\tau\right) \left(\mathbf{q}_t \mathbf{k}_j^\top\right) \mathbf{v}_j}_{\mathbf{o}_t^{\text{inter}}}$$

### 3.2 Intra-Chunk Computation (Tensor Core HGEMM)
Within chunk $c$, let $\mathbf{Q}_{[c]}, \mathbf{K}_{[c]} \in \mathbb{R}^{C \times d_k}$ and $\mathbf{V}_{[c]} \in \mathbb{R}^{C \times d_v}$.
The intra-chunk output is computed as:
$$\mathbf{O}_{[c]}^{\text{intra}} = \left( \left(\mathbf{Q}_{[c]} \mathbf{K}_{[c]}^\top \odot \mathbf{M}_{\text{causal}}\right) \odot \mathbf{D}_{[c]}^{\text{intra}} \right) \mathbf{V}_{[c]}$$
where:
- $\mathbf{M}_{\text{causal}} \in \{0, 1\}^{C \times C}$ with $M_{i, j} = 1 \iff i \ge j$.
- $\mathbf{D}_{[c]}^{\text{intra}} \in \mathbb{R}^{C \times C}$ denotes the relative intra-chunk decay:
  $$(D_{[c]}^{\text{intra}})_{i, j} = \exp\left( \sum_{\tau=(c-1)C + j + 1}^{(c-1)C + i} \log \gamma_\tau \right), \quad i \ge j$$

This operation requires two standard batched GEMMs:
1. $\mathbf{A}_{[c]} = \mathbf{Q}_{[c]} \mathbf{K}_{[c]}^\top \in \mathbb{R}^{C \times C}$ ($2 C^2 d_k$ FLOPs).
2. $\mathbf{O}_{[c]}^{\text{intra}} = \widetilde{\mathbf{A}}_{[c]} \mathbf{V}_{[c]} \in \mathbb{R}^{C \times d_v}$ ($2 C^2 d_v$ FLOPs).

Because $C \le 128$, the intermediate attention tile $\mathbf{A}_{[c]}$ fits entirely in GPU SRAM (Shared Memory), requiring **zero** DRAM roundtrips!

### 3.3 Inter-Chunk Recurrence and Prefix Scan
The inter-chunk contribution connects the historical recurrent state $\mathbf{S}_{[c-1]} \in \mathbb{R}^{d_k \times d_v}$ to chunk $c$:
$$\mathbf{S}_{[c]} = \mathbf{\Lambda}_{[c]} \mathbf{S}_{[c-1]} + \mathbf{\Delta S}_{[c]}$$
where:
- $\mathbf{\Lambda}_{[c]} = \prod_{\tau=1}^C \gamma_{(c-1)C + \tau}$ is the cumulative decay across chunk $c$.
- $\mathbf{\Delta S}_{[c]} = \sum_{j=1}^C \left(\prod_{\tau=j+1}^C \gamma_{(c-1)C + \tau}\right) \mathbf{k}_{(c-1)C + j}^\top \mathbf{v}_{(c-1)C + j} \in \mathbb{R}^{d_k \times d_v}$ is the new state generated inside chunk $c$.

In matrix form:
$$\mathbf{\Delta S}_{[c]} = \left(\mathbf{K}_{[c]}^\top \operatorname{diag}\left(\mathbf{d}_{[c]}^{\text{to\_end}}\right)\right) \mathbf{V}_{[c]}$$
where $(d_{[c]}^{\text{to\_end}})_j = \prod_{\tau=j+1}^C \gamma_{(c-1)C + \tau}$.

The sequence of chunk updates $(\mathbf{\Lambda}_{[1]}, \mathbf{\Delta S}_{[1]}), \dots, (\mathbf{\Lambda}_{[B]}, \mathbf{\Delta S}_{[B]})$ is scanned using the associative operator $\otimes$ across $B = N/C$ chunks:
- **Scan Length**: Reduced from $N$ tokens to $B = N/C$ chunk boundaries!
- For $N = 8192$ and $C = 64$, the scan operates over only $B = 128$ states.

### 3.4 Inter-Chunk Output Projection
The state entering chunk $c$ ($\mathbf{S}_{[c-1]}$) is projected to all tokens $i \in \{1, \dots, C\}$:
$$\mathbf{O}_{[c]}^{\text{inter}} = \operatorname{diag}\left(\mathbf{d}_{[c]}^{\text{from\_start}}\right) \mathbf{Q}_{[c]} \mathbf{S}_{[c-1]}$$
where $(d_{[c]}^{\text{from\_start}})_i = \prod_{\tau=1}^i \gamma_{(c-1)C + \tau}$.

Total token output:
$$\mathbf{O}_{[c]} = \mathbf{O}_{[c]}^{\text{intra}} + \mathbf{O}_{[c]}^{\text{inter}}$$

$$\begin{aligned}
\textbf{Theorem 3 (Chunk-Parallel Complexity)}: \quad &\text{For sequence length } N, \text{ chunk size } C, \text{ and hidden dimensions } d_k = d_v = d: \\
&\text{1. Compute Complexity: } \mathcal{O}\left(N \cdot C \cdot d + \frac{N}{C} \cdot d^2\right) \text{ FLOPs}. \\
&\text{2. SRAM Footprint: } \mathcal{O}(C^2 + C d) \text{ per streaming block}. \\
&\text{3. Memory Transfers: } \mathcal{O}(N d) \text{ bytes (IO-Optimal, Linear in } N\text{)}.
\end{aligned}$$

Setting $C \approx \sqrt{d}$ balances intra-chunk GEMM compute with inter-chunk state transfer, minimizing wall-clock runtime.

---

## 4. Delta-Rule Associative Memory & Error-Correcting Recurrences

### 4.1 The Saturation Failure of Additive Linear Attention
Standard linear attention accumulates key-value outer products additively:
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top$$
Because $\mathbf{S}_t$ is the unweighted sum of outer products, its spectral norm $\|\mathbf{S}_t\|_2$ grows linearly with sequence length $N$. When querying the memory with an existing key $\mathbf{k}_m$:
$$\widehat{\mathbf{v}}_m = \mathbf{k}_m^\top \mathbf{S}_N = \|\mathbf{k}_m\|_2^2 \mathbf{v}_m + \sum_{j \ne m} (\mathbf{k}_m^\top \mathbf{k}_j) \mathbf{v}_j$$
The secondary crosstalk summation $\sum_{j \ne m} (\mathbf{k}_m^\top \mathbf{k}_j) \mathbf{v}_j$ acts as catastrophic noise. In multi-needle retrieval tasks (Needle-in-a-Haystack), additive linear attention retrieval accuracy degrades rapidly past context lengths of $2048$ tokens.

### 4.2 Mathematical Derivation from Online Gradient Descent
DeltaNet resolves memory saturation by transforming the recurrent state update into an **online learning rule with error correction (the Widrow-Hoff Delta Rule)**.

$$\begin{aligned}
\textbf{Derivation (Delta Rule from Key-Value Retrieval Loss)}: \\
&\text{At step } t, \text{ define the instant associative retrieval loss on the current key-value pair } (\mathbf{k}_t, \mathbf{v}_t): \\
&\mathcal{L}_t(\mathbf{S}) \triangleq \frac{1}{2} \|\mathbf{k}_t^\top \mathbf{S} - \mathbf{v}_t^\top\|_2^2, \quad \text{where } \mathbf{S} \in \mathbb{R}^{d_k \times d_v}, \, \mathbf{k}_t \in \mathbb{R}^{d_k}, \, \mathbf{v}_t \in \mathbb{R}^{d_v}
\end{aligned}$$

Taking the matrix gradient of $\mathcal{L}_t$ with respect to $\mathbf{S}$ evaluated at the previous state $\mathbf{S}_{t-1}$:
$$\nabla_{\mathbf{S}} \mathcal{L}_t(\mathbf{S}_{t-1}) = \mathbf{k}_t \left(\mathbf{k}_t^\top \mathbf{S}_{t-1} - \mathbf{v}_t^\top\right) = \mathbf{k}_t \mathbf{k}_t^\top \mathbf{S}_{t-1} - \mathbf{k}_t \mathbf{v}_t^\top$$

Applying a single step of gradient descent with step size (learning rate) $\beta_t > 0$:
$$\begin{aligned}
\mathbf{S}_t &= \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \mathcal{L}_t(\mathbf{S}_{t-1}) \\
&= \mathbf{S}_{t-1} - \beta_t \left(\mathbf{k}_t \mathbf{k}_t^\top \mathbf{S}_{t-1} - \mathbf{k}_t \mathbf{v}_t^\top\right) \\
&= \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top
\end{aligned}$$

Taking the transpose (for layout $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ where retrieval is $\mathbf{S} \mathbf{k}_t$):
$$\mathbf{S}_t = \mathbf{S}_{t-1} \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$$
which can also be written in prediction-error correction form:
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t \left(\mathbf{v}_t - \mathbf{S}_{t-1} \mathbf{k}_t\right) \mathbf{k}_t^\top$$

```
═════════════════════════════════════════════════════════════════════════════════════════
                       DELTA-RULE ERROR-CORRECTION DYNAMICS
═════════════════════════════════════════════════════════════════════════════════════════
                    v_t (Ground Truth Target)
                     │
                     │  v_corr = v_t - pred_v  (Instantaneous Residual)
                     ▼
  S_{t-1} ──────► [ S_{t-1} * k_t ] ──► pred_v
     │                                     │
     │      (Contraction via I - beta*k*k^T)│
     ▼                                     ▼
 [ * alpha_t ] ◄──────────────────── [ + beta_t * (v_corr (x) k_t) ]
     │
     ▼
    S_t (Updated State)
═════════════════════════════════════════════════════════════════════════════════════════
```

### 4.3 Theorem: Contraction Mapping & Numerical Stability
$$\begin{aligned}
\textbf{Theorem 4 (Eigenvalue Spectrum and Stability)}: \quad &\text{Let } \|\mathbf{k}_t\|_2 = 1 \text{ and } \beta_t \in (0, 2). \\
&\text{The transition matrix } \mathbf{A}_t = \mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top \text{ has spectrum: } \\
&\operatorname{spec}(\mathbf{A}_t) = \{1 - \beta_t, \, 1, \, 1, \dots, 1\}.
\end{aligned}$$

$$\begin{aligned}
\textit{Proof.} \quad &\text{For eigenvector } \mathbf{k}_t: \quad \mathbf{A}_t \mathbf{k}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{k}_t = \mathbf{k}_t - \beta_t \mathbf{k}_t (1) = (1 - \beta_t) \mathbf{k}_t. \\
&\text{For any vector } \mathbf{w} \in \mathbb{R}^d \text{ orthogonal to } \mathbf{k}_t \text{ (where } \mathbf{k}_t^\top \mathbf{w} = 0\text{)}: \\
&\mathbf{A}_t \mathbf{w} = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{w} = \mathbf{w} - \mathbf{0} = 1 \cdot \mathbf{w}.
\end{aligned}$$
Thus, $\mathbf{A}_t$ has one eigenvalue $\lambda_1 = 1 - \beta_t$ and $d-1$ eigenvalues equal to 1.
For $\beta_t \in (0, 1]$, $|1 - \beta_t| \le 1$, ensuring that the state update is a **non-expansive contractive mapping** ($\|\mathbf{A}_t\|_2 \le 1$). When $\beta_t = 1$, $\mathbf{A}_t = \mathbf{I} - \mathbf{k}_t \mathbf{k}_t^\top$ acts as an exact orthogonal projection onto the subspace orthogonal to $\mathbf{k}_t$, completely erasing past memory that would interfere with $\mathbf{k}_t$. $\blacksquare$

### 4.4 Chunk-Parallel DeltaNet & Triangular Inversion
Because $\mathbf{A}_t = \mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ is a rank-1 perturbed matrix that varies per token, unrolling the recurrence inside a chunk involves non-commuting matrices:
$$\prod_{\tau=j+1}^i \left(\mathbf{I} - \beta_\tau \mathbf{k}_\tau \mathbf{k}_\tau^\top\right)$$
which cannot be evaluated via simple scalar decay multiplication.

Chunked DeltaNet resolves this by expressing the intra-chunk recurrent states via a strictly lower-triangular causal Gram matrix:
Let $\mathbf{K}_{[c]} \in \mathbb{R}^{C \times d_k}$ and $\mathbf{V}_{[c]} \in \mathbb{R}^{C \times d_v}$.
The intra-chunk prediction errors satisfy the linear system:
$$\mathbf{E}_{[c]} = \mathbf{V}_{[c]} - \left(\mathbf{K}_{[c]} \mathbf{K}_{[c]}^\top \odot \mathbf{M}_{\text{strictly\_causal}}\right) \operatorname{diag}(\boldsymbol{\beta}_{[c]}) \mathbf{E}_{[c]}$$
Rearranging terms:
$$\left( \mathbf{I}_C + \left(\mathbf{K}_{[c]} \mathbf{K}_{[c]}^\top \odot \mathbf{M}_{\text{strictly\_causal}}\right) \operatorname{diag}(\boldsymbol{\beta}_{[c]}) \right) \mathbf{E}_{[c]} = \mathbf{V}_{[c]} - \mathbf{K}_{[c]} \mathbf{S}_{[c-1]}^\top$$

Let $\mathbf{T} = \mathbf{I}_C + \left(\mathbf{K}_{[c]} \mathbf{K}_{[c]}^\top \odot \mathbf{M}_{\text{strictly\_causal}}\right) \operatorname{diag}(\boldsymbol{\beta}_{[c]})\in \mathbb{R}^{C \times C}$.
Because $\mathbf{M}_{\text{strictly\_causal}}$ is strictly lower triangular ($T_{i, j} = 0$ for $j \ge i$), $\mathbf{T}$ is a unit lower triangular matrix!
- **Exact Solve**: $\mathbf{E}_{[c]} = \mathbf{T}^{-1} (\mathbf{V}_{[c]} - \mathbf{K}_{[c]} \mathbf{S}_{[c-1]}^\top)$, computed in $\mathcal{O}(C^2)$ via triangular substitution (`trsm` or fast Triton SRAM solve).
- **Neumann Series Approximation** (for high-throughput streaming):
  $$\mathbf{T}^{-1} \approx \mathbf{I} - \mathbf{N} + \mathbf{N}^2 - \mathbf{N}^3 + \dots$$
  where $\mathbf{N} = \left(\mathbf{K}_{[c]} \mathbf{K}_{[c]}^\top \odot \mathbf{M}_{\text{strictly\_causal}}\right) \operatorname{diag}(\boldsymbol{\beta}_{[c]})$. For small $C$ ($C=32$ or $C=64$), truncation at degree 2 or 3 guarantees numerical parity with exact autograd checks!

---

## 5. Comparative Taxonomy: GLA, DeltaNet, RetNet, and RWKV-v6

To position research contributions accurately in MLSys/NeurIPS papers, use this definitive mathematical and architectural taxonomy:

```
═════════════════════════════════════════════════════════════════════════════════════════
                     LINEAR RECURRENT ARCHITECTURE FAMILIES
═════════════════════════════════════════════════════════════════════════════════════════
   Architecture    State Transition A_t           Update B_t X_t       Associativity Class
  ─────────────────────────────────────────────────────────────────────────────────────
   GLA             alpha_t * I  (data-dep scalar) k_t v_t^T            Commutative Decay
   DeltaNet / GDN  alpha_t(I - beta_t k_t k_t^T)  beta_t v_t k_t^T     Non-Commutative Projective
   RetNet          gamma * I    (static scalar)   k_t v_t^T            Stationary Monoid
   RWKV-v6         diag(exp(-exp(w_t)))           k_t v_t^T            Vector-Gated Commutative
═════════════════════════════════════════════════════════════════════════════════════════
```

### 5.1 Gated Linear Attention (GLA)
- **Recurrence**:
  $$\mathbf{S}_t = \operatorname{diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top$$
  where $\boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t \mathbf{W}_\alpha + \mathbf{b}_\alpha) \in (0, 1)^{d_k}$ is a data-dependent vector gate.
- **Strengths**: Preserves commutativity of decay $\prod_\tau \boldsymbol{\alpha}_\tau$; enables fully parallel Blelloch scan across arbitrary chunk sizes; highly efficient Triton chunked implementation.
- **Limitations**: Additive outer-product accumulation still suffers from capacity saturation on long sequences.

### 5.2 DeltaNet & Gated DeltaNet (GDN)
- **Recurrence**:
  $$\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$$
  where $\alpha_t \in (0, 1)$ is a forgetting factor and $\beta_t = \sigma(\mathbf{x}_t \mathbf{w}_\beta) \in (0, 1)$ is the data-dependent write rate.
- **Strengths**: Solves associative recall degradation; achieves transformer-grade performance on Needle-in-a-Haystack tasks; prevents norm explosion.
- **Systems Challenge**: Non-commutative transition requires intra-chunk triangular solve ($\mathbf{T}^{-1}$) or specialized chunked checkpointing.

### 5.3 Retention Network (RetNet)
- **Recurrence**:
  $$\mathbf{S}_t = \gamma \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top, \quad \gamma \in (0, 1) \text{ fixed per head}$$
  where decay $\gamma_h = 1 - 2^{-5 - h}$ is a hardwired, non-data-dependent exponential hyperparameter.
- **Dual Representation**: Supports three equivalent representations:
  1. *Parallel*: Pure matrix multiply with static Toeplitz decay matrix.
  2. *Recurrent*: $\mathcal{O}(1)$ step latency during inference.
  3. *Chunkwise*: Intra-chunk GEMM + inter-chunk state propagation.
- **Limitations**: Static decay cannot dynamically preserve salient information across varying sequence lengths.

### 5.4 RWKV-v6 (Finch)
- **Recurrence**:
  $$\mathbf{S}_t = \operatorname{diag}(\mathbf{w}_t) \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top$$
  where $\mathbf{w}_t = \exp(-\exp(\mathbf{w}_0 + \tanh(\mathbf{x}_t \mathbf{W}_1) \mathbf{W}_2))$ is a low-rank data-dependent vector decay.
- **Key Innovation**: Dynamic token shift / data-dependent interpolation:
  $$\mathbf{r}_t = \mathbf{W}_r (\mathbf{x}_t \odot \boldsymbol{\mu}_x + \mathbf{x}_{t-1} \odot (1 - \boldsymbol{\mu}_x))$$
  combines channel-mixing and time-mixing with head-specific matrix states.

### 5.5 Canonical Synthesis Matrix

| Dimension / Metric | RetNet | GLA | RWKV-v6 (Finch) | Gated DeltaNet (GDN) |
| :--- | :---: | :---: | :---: | :---: |
| **State Dimension $\mathcal{S}$** | $d_k \times d_v$ | $d_k \times d_v$ | $d_k \times d_v$ | $d_k \times d_v$ |
| **Decay Nature** | Static scalar $\gamma_h$ | Data-dependent vector $\boldsymbol{\alpha}_t$ | Data-dependent vector $\mathbf{w}_t$ | Data-dep scalar $\alpha_t$ + projection |
| **Update Rule** | Additive Hebbian | Additive Gated | Additive Shifted | **Error-Correcting Delta** |
| **Associativity Class** | Commutative Monoid | Commutative Monoid | Commutative Monoid | Non-Commutative Projective |
| **Training Parallelism** | Parallel / Chunked | Chunk-Parallel GEMM | Chunked Scan / WKV6 | Chunked Triangular Solve |
| **Decoding Latency** | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ |
| **Memory Footprint** | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ | $\mathcal{O}(d_k d_v)$ |
| **Associative Recall Capacity** | Low ($N > 4\text{k}$) | Moderate ($N > 8\text{k}$) | Moderate ($N > 8\text{k}$) | **High (Near-Transformer)** |

---

## 6. Publication-Grade LaTeX Environments

### 6.1 Formal Theorem & Lemma Environments (AMS-LaTeX)

```latex
% Preamble requirements:
% \usepackage{amsmath,amssymb,amsthm,mathtools}
% \newtheorem{theorem}{Theorem}
% \newtheorem{lemma}[theorem]{Lemma}
% \newtheorem{definition}{Definition}

\begin{definition}[Associative Recurrent State Operator]
\label{def:associative_recurrent_operator}
Let $\mathcal{A}$ denote the set of transition operators on a memory manifold $\mathcal{S}$, and let $\mathcal{U}$ denote the input space. We define the state transition tuple $u_t = (\mathbf{A}_t, \mathbf{U}_t) \in \mathcal{A} \times \mathcal{S}$ and the binary composition operator $\otimes: (\mathcal{A} \times \mathcal{S}) \times (\mathcal{A} \times \mathcal{S}) \to \mathcal{A} \times \mathcal{S}$ as:
\begin{equation}
(\mathbf{A}_i, \mathbf{U}_i) \otimes (\mathbf{A}_j, \mathbf{U}_j) \triangleq \left(\mathbf{A}_j \mathbf{A}_i, \, \mathbf{A}_j \mathbf{U}_i + \mathbf{U}_j\right).
\end{equation}
\end{definition}

\begin{theorem}[Prefix Reduction via Associative Monoid]
\label{thm:prefix_reduction_monoid}
The algebraic structure $(\mathcal{A} \times \mathcal{S}, \otimes)$ forms a monoid with two-sided identity $e = (\mathbf{I}, \mathbf{0})$. Consequently, the sequential recurrence $\mathbf{S}_t = \mathbf{A}_t \mathbf{S}_{t-1} + \mathbf{U}_t$ with $\mathbf{S}_0 = \mathbf{0}$ satisfies:
\begin{equation}
\left(\prod_{\tau=1}^t \mathbf{A}_\tau, \, \mathbf{S}_t\right) = \bigotimes_{\tau=1}^t (\mathbf{A}_\tau, \mathbf{U}_\tau),
\end{equation}
permitting parallel prefix evaluation in $\mathcal{O}(\log t)$ algorithmic depth.
\end{theorem}

\begin{proof}
Associativity follows from the associativity of matrix multiplication and distributivity over addition:
\begin{align*}
\left( (\mathbf{A}_1, \mathbf{U}_1) \otimes (\mathbf{A}_2, \mathbf{U}_2) \right) \otimes (\mathbf{A}_3, \mathbf{U}_3) 
&= (\mathbf{A}_2 \mathbf{A}_1, \, \mathbf{A}_2 \mathbf{U}_1 + \mathbf{U}_2) \otimes (\mathbf{A}_3, \mathbf{U}_3) \\
&= \left(\mathbf{A}_3 (\mathbf{A}_2 \mathbf{A}_1), \, \mathbf{A}_3(\mathbf{A}_2 \mathbf{U}_1 + \mathbf{U}_2) + \mathbf{U}_3\right) \\
&= \left(\mathbf{A}_3 \mathbf{A}_2 \mathbf{A}_1, \, \mathbf{A}_3 \mathbf{A}_2 \mathbf{U}_1 + \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3\right) \\
&= (\mathbf{A}_1, \mathbf{U}_1) \otimes \left(\mathbf{A}_3 \mathbf{A}_2, \, \mathbf{A}_3 \mathbf{U}_2 + \mathbf{U}_3\right) \\
&= (\mathbf{A}_1, \mathbf{U}_1) \otimes \left( (\mathbf{A}_2, \mathbf{U}_2) \otimes (\mathbf{A}_3, \mathbf{U}_3) \right).
\end{align*}
The identity property is verified directly: $(\mathbf{I}, \mathbf{0}) \otimes (\mathbf{A}, \mathbf{U}) = (\mathbf{A} \mathbf{I}, \mathbf{A}\mathbf{0} + \mathbf{U}) = (\mathbf{A}, \mathbf{U})$.
By Blelloch's prefix theorem, any associative monoid operation over sequence length $N$ admits a work-efficient tree reduction with total work $\mathcal{O}(N)$ and circuit depth $\mathcal{O}(\log N)$.
\end{proof}
```

---

### 6.2 Publication Algorithm 1: Blelloch Work-Efficient Scan (`algorithm2e`)

```latex
\begin{algorithm}[t]
\caption{Blelloch Work-Efficient Parallel Associative Prefix Scan}
\label{alg:blelloch_associative_scan}
\SetAlgoLined
\KwIn{Array of state transitions $\mathbf{X} = [u_0, u_1, \dots, u_{N-1}]$ where $u_i = (\mathbf{A}_i, \mathbf{U}_i)$, sequence length $N = 2^d$}
\KwOut{Array of prefix states $\mathbf{S} = [s_0, s_1, \dots, s_{N-1}]$ where $s_i = \bigotimes_{j=0}^i u_j$}

\tcp{Phase 1: Up-Sweep (Parallel Tree Reduction)}
\For{$k \leftarrow 0$ \KwTo $\log_2 N - 1$}{
    \For{$j \leftarrow 0$ \KwTo $\frac{N}{2^{k+1}} - 1$ \textbf{in parallel}}{
        $\text{idx} \leftarrow (j + 1) \cdot 2^{k+1} - 1$\;
        $\text{left} \leftarrow \text{idx} - 2^k$\;
        $\mathbf{X}[\text{idx}] \leftarrow \mathbf{X}[\text{left}] \otimes \mathbf{X}[\text{idx}]$\;
    }
}

\tcp{Phase 2: Down-Sweep (Prefix Distribution)}
$\mathbf{X}[N - 1] \leftarrow (\mathbf{I}, \mathbf{0})$ \tcp*{Set root to algebraic identity}
\For{$k \leftarrow \log_2 N - 1$ \KwDownTo $0$}{
    \For{$j \leftarrow 0$ \KwTo $\frac{N}{2^{k+1}} - 1$ \textbf{in parallel}}{
        $\text{idx} \leftarrow (j + 1) \cdot 2^{k+1} - 1$\;
        $\text{left} \leftarrow \text{idx} - 2^k$\;
        $\mathbf{T} \leftarrow \mathbf{X}[\text{left}]$\;
        $\mathbf{X}[\text{left}] \leftarrow \mathbf{X}[\text{idx}]$\;
        $\mathbf{X}[\text{idx}] \leftarrow \mathbf{T} \otimes \mathbf{X}[\text{idx}]$\;
    }
}
\Return{$\mathbf{X}$}\;
\end{algorithm}
```

---

### 6.3 Publication Algorithm 2: Chunk-Parallel Linear Recurrent Attention (`algorithm2e`)

```latex
\begin{algorithm}[t]
\caption{Chunk-Parallel Gated Linear Recurrence Forward Pass}
\label{alg:chunk_parallel_linear_attention}
\SetAlgoLined
\KwIn{Queries $\mathbf{Q}$, Keys $\mathbf{K} \in \mathbb{R}^{N \times d_k}$, Values $\mathbf{V} \in \mathbb{R}^{N \times d_v}$, Decay Gates $\boldsymbol{\gamma} \in (0, 1]^N$, Chunk Size $C$}
\KwOut{Attention Output $\mathbf{O} \in \mathbb{R}^{N \times d_v}$}

$B \leftarrow \lceil N / C \rceil$\;
Initialize chunk state buffer $\mathbf{S} \leftarrow \mathbf{0}_{B \times d_k \times d_v}$\;

\tcp{Step 1: Intra-Chunk Attention and Local State Generation}
\For{$c \leftarrow 1$ \KwTo $B$ \textbf{in parallel}}{
    $\mathbf{Q}_c \leftarrow \mathbf{Q}[(c-1)C+1 : cC], \quad \mathbf{K}_c \leftarrow \mathbf{K}[(c-1)C+1 : cC], \quad \mathbf{V}_c \leftarrow \mathbf{V}[(c-1)C+1 : cC]$\;
    $\mathbf{D}_c^{\text{intra}} \leftarrow \text{ComputeDecayMatrix}(\boldsymbol{\gamma}[(c-1)C+1 : cC])$\;
    $\mathbf{A}_c \leftarrow (\mathbf{Q}_c \mathbf{K}_c^\top \odot \mathbf{M}_{\text{causal}}) \odot \mathbf{D}_c^{\text{intra}}$ \tcp*{Intra-chunk SRAM GEMM}
    $\mathbf{O}_c^{\text{intra}} \leftarrow \mathbf{A}_c \mathbf{V}_c$\;
    $\mathbf{\Delta S}_c \leftarrow (\mathbf{K}_c^\top \operatorname{diag}(\mathbf{d}_c^{\text{to\_end}})) \mathbf{V}_c$\;
    $\Lambda_c \leftarrow \prod_{\tau=1}^C \gamma_{(c-1)C + \tau}$\;
}

\tcp{Step 2: Inter-Chunk Associative State Scan across B chunks}
$\mathbf{S}_0 \leftarrow \mathbf{0}_{d_k \times d_v}$\;
\For{$c \leftarrow 1$ \KwTo $B$}{
    $\mathbf{S}_c \leftarrow \Lambda_c \mathbf{S}_{c-1} + \mathbf{\Delta S}_c$\;
}

\tcp{Step 3: Inter-Chunk Output Projection and Additive Fusion}
\For{$c \leftarrow 1$ \KwTo $B$ \textbf{in parallel}}{
    $\mathbf{O}_c^{\text{inter}} \leftarrow \operatorname{diag}(\mathbf{d}_c^{\text{from\_start}}) \mathbf{Q}_c \mathbf{S}_{c-1}$\;
    $\mathbf{O}[(c-1)C+1 : cC] \leftarrow \mathbf{O}_c^{\text{intra}} + \mathbf{O}_c^{\text{inter}}$\;
}
\Return{$\mathbf{O}$}\;
\end{algorithm}
```

---

### 6.4 Publication Algorithm 3: Chunked Gated DeltaNet Forward Pass (`algorithmicx`)

```latex
\begin{algorithm}[t]
\caption{Chunked Gated DeltaNet with Error-Correction Projection}
\label{alg:chunked_gated_deltanet}
\begin{algorithmic}[1]
\Require Query $\mathbf{Q}$, Key $\mathbf{K} \in \mathbb{R}^{N \times d}$, Value $\mathbf{V} \in \mathbb{R}^{N \times d}$, Decay $\boldsymbol{\alpha} \in (0, 1)^N$, Learning Rate $\boldsymbol{\beta} \in (0, 1)^N$, Chunk Size $C$
\Ensure Output $\mathbf{Y} \in \mathbb{R}^{N \times d}$
\State $B \gets N / C$
\State $\mathbf{S}_0 \gets \mathbf{0}_{d \times d}$
\For{$c = 1 \textbf{ to } B$}
    \State $\mathbf{K}_c, \mathbf{V}_c, \mathbf{Q}_c \gets \mathbf{K}_{[c]}, \mathbf{V}_{[c]}, \mathbf{Q}_{[c]}$
    \State $\mathbf{G}_c \gets (\mathbf{K}_c \mathbf{K}_c^\top \odot \mathbf{M}_{\text{strictly\_causal}}) \operatorname{diag}(\boldsymbol{\beta}_{[c]})$ \Comment{Strictly lower-triangular Gram matrix}
    \State $\mathbf{T}_c \gets \mathbf{I}_C + \mathbf{G}_c$
    \State $\mathbf{V}_c^{\text{proj}} \gets \mathbf{V}_c - \mathbf{K}_c \mathbf{S}_{c-1}^\top$ \Comment{Subtract historical memory prediction}
    \State $\mathbf{E}_c \gets \operatorname{SolveTriangular}(\mathbf{T}_c, \mathbf{V}_c^{\text{proj}})$ \Comment{Exact error correction solve via trsm}
    \State $\mathbf{U}_c \gets \operatorname{diag}(\boldsymbol{\beta}_{[c]}) \mathbf{E}_c$
    \State $\mathbf{Y}_c^{\text{intra}} \gets (\mathbf{Q}_c \mathbf{K}_c^\top \odot \mathbf{M}_{\text{causal}}) \mathbf{U}_c$
    \State $\mathbf{Y}_c^{\text{inter}} \gets \mathbf{Q}_c \mathbf{S}_{c-1}$
    \State $\mathbf{Y}_{[c]} \gets \mathbf{Y}_c^{\text{intra}} + \mathbf{Y}_c^{\text{inter}}$
    \State $\mathbf{S}_c \gets \alpha_c \mathbf{S}_{c-1} + \mathbf{U}_c^\top \mathbf{K}_c$ \Comment{Chunk state carry-over}
\EndFor
\State \Return $\mathbf{Y}$
\end{algorithmic}
\end{algorithm}
```

---

## 7. Production PyTorch and Triton Verification References

### 7.1 Numerically Stable PyTorch Chunk-Parallel Reference
Use this reference when verifying mathematical consistency against unit tests or gradcheck:

```python
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Production PyTorch Reference: Chunk-Parallel Linear Recurrence with Decay.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def chunk_parallel_linear_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Evaluates causal linear attention with exponential decay using chunk parallelism.

    Args:
        q: (B, H, N, D_k) Query tensor
        k: (B, H, N, D_k) Key tensor
        v: (B, H, N, D_v) Value tensor
        gamma: (B, H, N) Tokenwise decay factor in (0, 1]
        chunk_size: Block partition size C (default: 64)

    Returns:
        out: (B, H, N, D_v) Processed sequence output
    """
    B, H, N, D_k = q.shape
    D_v = v.shape[-1]
    assert N % chunk_size == 0, "Sequence length N must be divisible by chunk_size."
    num_chunks = N // chunk_size

    # Reshape into chunked views: (B, H, num_chunks, C, D)
    q_c = q.view(B, H, num_chunks, chunk_size, D_k)
    k_c = k.view(B, H, num_chunks, chunk_size, D_k)
    v_c = v.view(B, H, num_chunks, chunk_size, D_v)
    g_c = gamma.view(B, H, num_chunks, chunk_size)

    # Compute log cumulative decay: log_g = cumsum(log(gamma))
    log_g = torch.log(torch.clamp(g_c, min=1e-7))
    log_cum_decay = torch.cumsum(log_g, dim=-1)  # (B, H, num_chunks, C)

    # 1. Intra-chunk decay mask: D[i, j] = exp(log_cum_decay[i] - log_cum_decay[j]) for i >= j
    decay_diff = log_cum_decay.unsqueeze(-1) - log_cum_decay.unsqueeze(-2)
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))
    intra_decay = torch.exp(decay_diff) * causal_mask

    # 2. Intra-chunk attention GEMM
    # A_intra = (Q @ K.T) * intra_decay -> (B, H, num_chunks, C, C)
    qk = torch.matmul(q_c, k_c.transpose(-1, -2))
    a_intra = qk * intra_decay
    out_intra = torch.matmul(a_intra, v_c)  # (B, H, num_chunks, C, D_v)

    # 3. Inter-chunk state updates
    # State contribution generated inside each chunk:
    # d_to_end[j] = exp(log_cum_decay[-1] - log_cum_decay[j])
    d_to_end = torch.exp(log_cum_decay[..., -1:] - log_cum_decay)  # (B, H, num_chunks, C)
    k_decayed = k_c * d_to_end.unsqueeze(-1)
    delta_s = torch.matmul(k_decayed.transpose(-1, -2), v_c)  # (B, H, num_chunks, D_k, D_v)

    # Total decay across chunk: Lambda = exp(log_cum_decay[-1])
    lambda_chunk = torch.exp(log_cum_decay[..., -1])  # (B, H, num_chunks)

    # Sequential scan across chunk boundaries (num_chunks is small, e.g., 64-128)
    s_historical = torch.zeros(B, H, num_chunks, D_k, D_v, device=q.device, dtype=torch.float32)
    curr_s = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)

    for c in range(num_chunks):
        s_historical[:, :, c] = curr_s
        l_c = lambda_chunk[:, :, c].unsqueeze(-1).unsqueeze(-1)
        curr_s = l_c * curr_s + delta_s[:, :, c].float()

    # 4. Inter-chunk projection: out_inter = d_from_start * (Q @ S_prev)
    d_from_start = torch.exp(log_cum_decay)  # (B, H, num_chunks, C)
    q_s = torch.matmul(q_c.float(), s_historical)  # (B, H, num_chunks, C, D_v)
    out_inter = (q_s * d_from_start.unsqueeze(-1)).to(q.dtype)

    out = out_intra + out_inter
    return out.view(B, H, N, D_v)
```

---

### 7.2 Triton Chunked Associative Scan Kernel Pattern
In accordance with NeuralOps architecture design rules:
- Mainline Triton code strictly avoids block pointers (`tl.make_block_ptr` / `tl.advance`) to maintain universal compatibility across ROCm, CUDA, and Triton releases.
- FP32 accumulation guardrails for all recurrent state sums.

```python
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton JIT Kernel: Intra-Chunk Associative State Accumulation.
Demonstrates pointer arithmetic standard without banned block pointers.
"""

from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def _intra_chunk_state_kernel(
    k_ptr,
    v_ptr,
    delta_s_ptr,
    decay_to_end_ptr,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_sb,
    stride_sh,
    stride_sc,
    stride_sdk,
    stride_sdv,
    CHUNK_SIZE: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """
    Computes Delta_S = (K_c^T * d_to_end) @ V_c for a single chunk.
    Grid: (num_chunks, B * H)
    """
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_k = tl.arange(0, D_K)
    offs_v = tl.arange(0, D_V)

    # Initialize FP32 accumulator in SRAM
    acc_s = tl.zeros((D_K, D_V), dtype=tl.float32)

    # Loop over tokens inside chunk C
    for i in range(CHUNK_SIZE):
        tok_idx = pid_c * CHUNK_SIZE + i
        decay_val = tl.load(decay_to_end_ptr + pid_bh * stride_kh + tok_idx)

        # Explicit pointer arithmetic (no tl.make_block_ptr)
        k_ptrs = k_ptr + pid_bh * stride_kb + tok_idx * stride_kn + offs_k * stride_kd
        v_ptrs = v_ptr + pid_bh * stride_vb + tok_idx * stride_vn + offs_v * stride_vd

        k_val = tl.load(k_ptrs).to(tl.float32) * decay_val
        v_val = tl.load(v_ptrs).to(tl.float32)

        # Rank-1 outer product accumulation
        acc_s += k_val[:, None] * v_val[None, :]

    # Store back to global memory
    out_ptrs = (
        delta_s_ptr
        + pid_bh * stride_sb
        + pid_c * stride_sc
        + offs_k[:, None] * stride_sdk
        + offs_v[None, :] * stride_sdv
    )
    tl.store(out_ptrs, acc_s.to(delta_s_ptr.dtype.element_ty))
```

---

## 8. Quality Gate, Style Conventions & Mathematical Sanity Checklist

Before inserting theoretical formulations or algorithms into papers, verify compliance against this checklist:

- [ ] **Algebraic Associativity Proof**: Did you verify that the composition satisfies $(u_1 \otimes u_2) \otimes u_3 = u_1 \otimes (u_2 \otimes u_3)$?
- [ ] **Work Efficiency Check**: Is the Blelloch total work strictly $\mathcal{O}(N)$ rather than $\mathcal{O}(N \log N)$?
- [ ] **Span / Depth Clarification**: Did you explicitly distinguish **work** ($\text{FLOP count}$) from **span / depth** ($\text{step count on parallel PRAM}$)?
- [ ] **Delta-Rule Stability Condition**: Did you include the non-expansive contraction bound ($\beta_t \in (0, 2 / \|\mathbf{k}_t\|_2^2)$) and explain orthogonal subspace projection?
- [ ] **Clean AMS-LaTeX Theorem Environments**: Are all lemmas, theorems, and definitions wrapped in formal AMS environments with `\label{...}` and Q.E.D. blocks?
- [ ] **Standard Algorithm Packages**: Are algorithms typeset using either `algorithm2e` (`\SetAlgoLined`, `\KwIn`, `\KwOut`) or `algorithmicx` (`\begin{algorithmic}[1]`, `\Require`, `\Ensure`)?
- [ ] **No Prohibited References**: Ensure zero mentions of remote hosts, IP addresses, or environment credentials.
- [ ] **Notation Consistency**:
  - $N$: Sequence length
  - $C$: Chunk / block size ($C \in \{64, 128\}$)
  - $B = \lceil N/C \rceil$: Number of chunks
  - $d_k, d_v, d$: Feature / state dimensions
  - $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$: Recurrent state
  - $\otimes$: Associative binary operator
