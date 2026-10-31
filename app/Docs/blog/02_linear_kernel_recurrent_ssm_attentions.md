# ⚡ Part 2: Linear, Kernelized, Recurrent & State Space (SSM) Attentions

> **The Visual Guide to Escaping Quadratic Complexity: How Matrix Associativity, Kernel Maps, Selective Recurrences, and State Space Duality Power the Next Generation of Sub-Quadratic Language Models.**

---

## 📑 Table of Contents
1. [Katharopoulos Linear Attention (Associative Kernel Property)](#1-katharopoulos-linear-attention)
2. [Performer Attention (FAVOR+ Positive Random Features)](#2-performer-attention)
3. [Linformer Attention (Low-Rank Projected Key-Value Length)](#3-linformer-attention)
4. [Mamba & State Space Models (Selective Structured SSMs)](#4-mamba--state-space-models)
5. [MambaSVD-Flash Hybrid (SSM Recurrence + Low-Rank Projections)](#5-mambasvd-flash-hybrid)
6. [RetNet: Retention Networks (Chunkwise Parallel & Recurrent Inference)](#6-retnet-retention-networks)
7. [RWKV Time-Mixing (Receptance Weighted Key Value)](#7-rwkv-time-mixing)
8. [HGRN2 & Gated DeltaNet (Hierarchical Gated Linear Recurrence)](#8-hgrn2--gated-deltanet)
9. [Griffin & Hawk (Recurrent Gated Linear Attention with SSD)](#9-griffin--hawk)
10. [💻 Practical PyTorch Implementation: Linear Recurrent Attention Engine](#-practical-pytorch-implementation-linear-recurrent-attention-engine)
11. [🏁 Architectural Master Comparison Table](#-architectural-master-comparison-table)

---

## 1. Katharopoulos Linear Attention

```
Standard Quadratic Attention:        Linear Kernelized Attention:
    (Q @ K.T) @ V                          Q @ (K.T @ V)
   [N×d] [d×N]   [N×d]                    [N×d]   [d×N] [N×d]
   └──┬──────┘     │                        │     └───┬─────┘
    [N×N]          │                        │       [d×d]
      └─────┬──────┘                        └─────┬───┘
          [N×d]                                 [N×d]
   O(N²d) Complexity                     O(Nd²) Complexity
```

### Title & Subtitle
**Katharopoulos Linear Attention: Flipping the Matrix Multiplication Order with Kernel Associativity**  
*How swapping $(Q K^T) V$ to $Q (K^T V)$ via positive feature maps unlocks linear time training and constant-memory recurrent inference.*

### The Analogy
Imagine an auditorium filled with 10,000 historians. In standard attention, every single historian must walk across the room and conduct a private one-on-one conversation with all other 9,999 historians ($10,000 \times 10,000 = 100,000,000$ pairwise interactions). 

In Katharopoulos Linear Attention, nobody talks to each other directly. Instead, there is a single shared **blackboard** ($d \times d$) at the front of the stage. As each historian walks past, they write their key insight onto the blackboard ($S_t = S_{t-1} + k_t v_t^T$). Any historian can read the blackboard in one instant glance ($\phi(q_t) S_t$) without needing to interview the 9,999 people one by one!

```mermaid
flowchart LR
    subgraph StandardQuadratic["Standard Softmax Attention: O(N²d)"]
        Q1["Q (N×d)"] --> AttnMat["(Q @ Kᵀ) : [N×N] Softmax Matrix"]
        K1["K (N×d)"] --> AttnMat
        AttnMat --> Out1["AttnMat @ V : [N×d]"]
        V1["V (N×d)"] --> Out1
    end
    
    subgraph LinearAssociative["Linear Kernel Attention: O(Nd²)"]
        K2["ϕ(K) (N×d)"] --> State["(ϕ(K)ᵀ @ V) : [d×d] Global State S"]
        V2["V (N×d)"] --> State
        Q2["ϕ(Q) (N×d)"] --> Out2["ϕ(Q) @ S : [N×d]"]
        State --> Out2
    end
```

### The Problem
Standard scaled dot-product attention computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
The non-linear $\text{softmax}(\cdot)$ operation binds $Q$ and $K^T$ together *before* multiplying with $V$. This forces the hardware to materialize an explicit $N \times N$ attention matrix. When sequence length $N$ scales from $2\text{k} \to 32\text{k} \to 128\text{k}$, compute and memory costs skyrocket by $4\times, 256\times, 4096\times$, hitting an impassable memory wall and requiring an ever-expanding KV cache during token generation.

### How It Works (The Intuition)
1. **The Kernel Perspective**: We express the similarity between query $q_i$ and key $k_j$ as an inner product of non-linear feature representations: $\text{Sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j)$, where $\phi(x) = \text{elu}(x) + 1$ guarantees strictly positive weights.
2. **Associativity Magic**: Because matrix multiplication is associative ($A(BC) = (AB)C$), removing the row-wise Softmax allows us to change the computation order:
   $$\left( \phi(Q) \phi(K)^T \right) V = \phi(Q) \left( \phi(K)^T V \right)$$
3. **From Quadratic to Linear**:
   - Instead of computing $(N \times d) \times (d \times N) \to [N \times N]$, we compute $(d \times N) \times (N \times d) \to [d \times d]$ first!
   - For a fixed model dimension $d = 64$ and sequence length $N = 100,000$, computing a $[64 \times 64]$ matrix requires negligible memory compared to a $[100,000 \times 100,000]$ matrix ($4,096$ floats vs. $10\text{ billion}$ floats).
4. **The Dual Representation (Transformer $\leftrightarrow$ RNN)**:
   - **Parallel Training**: Compute the cumulative sum prefix scan of $\phi(K)^T V$ across all $N$ tokens simultaneously.
   - **Autoregressive Inference**: Maintain a running state matrix $S_t \in \mathbb{R}^{d \times d}$ and normalizer vector $z_t \in \mathbb{R}^d$. Generating token $t+1$ requires only a single matrix-vector multiply ($\mathcal{O}(1)$ memory, $\mathcal{O}(d^2)$ compute).

### The Math (Light)
$$\text{Feature Map: } \phi(x) = \text{elu}(x) + 1 \quad (\phi(x) > 0 \; \forall x)$$

$$\text{Parallel Matrix Form: } Y = \frac{\phi(Q) \left(\phi(K)^T V\right)}{\phi(Q) \left(\phi(K)^T \mathbf{1}_N\right)} \implies \mathcal{O}(N d^2)$$

$$\text{Autoregressive Recurrent Form: } S_t = S_{t-1} + \phi(k_t) v_t^T \in \mathbb{R}^{d \times d}, \quad z_t = z_{t-1} + \phi(k_t) \in \mathbb{R}^d$$

$$\text{Token Output: } y_t = \frac{\phi(q_t)^T S_t}{\phi(q_t)^T z_t} \in \mathbb{R}^d \implies \mathcal{O}(1) \text{ memory per step}$$

### Why It Matters
- **Strict Linear Scaling**: Time complexity drops from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N d^2)$. At $N = 64\text{k}$, execution is over **100x faster**.
- **Constant $O(1)$ KV Footprint**: Decouples inference memory from sequence length; generating token 1,000,000 takes the exact same 4KB of state memory as token 10.
- **Foundation for Modern SSMs**: Laid the theoretical foundation for RetNet, RWKV, Mamba, and Linear Transformers.

### Image Prompt
> 🎨 **Image Prompt:** A high-end 3D isometric scientific infographic on dark cybernetic slate `#0B0F19`. On the left side, an enormous, blazing red-orange quadratic $N \times N$ matrix grid glowing with excessive memory heat. A bright cyan laser `#00E5FF` passes through an optical crystal labeled "Associative Kernel $\phi(x)$", transforming the calculation into two ultra-compact, glowing emerald green matrices ($d \times d$) labeled "Memory State $S_t$". Sleek neon data streams bypass the giant red grid and flow through the compact state with glowing mathematical annotations $(\phi(Q)(\phi(K)^T V))$. 8k render, octane render style, futuristic HUD overlays.

---

## 2. Performer Attention

```
FAVOR+ Positive Orthogonal Random Features:
   x (d-dim) ──> [ W_orth (m × d) ] ──> Positive Exponential Kernel Map ──> ϕ(x) (m-dim)
   
Exact Softmax Kernel Approximation:
   E[ ϕ(q)ᵀ ϕ(k) ] = exp( qᵀk / √d )   without any N×N matrix!
```

### Title & Subtitle
**Performer Attention: Approximating the True Softmax Kernel with FAVOR+**  
*Unbiased, positive orthogonal random feature maps that emulate exact Softmax attention with provable linear complexity.*

### The Analogy
Imagine conducting a national census across 300 million citizens. Interviewing every single pair of citizens to measure cultural affinity would require $9 \times 10^{16}$ interviews. 

Instead, a world-class statistician selects 128 perfectly calibrated, mutually orthogonal demographic questions ($m$ orthogonal random feature probes). By projecting every citizen onto this standardized 128-dimensional questionnaire, the statistician calculates the exact nationwide cultural affinity matrix with provable mathematical accuracy and zero demographic bias.

```mermaid
flowchart TD
    InputQ["Query Vector q ∈ ℝᵈ"] --> ProjQ["Orthogonal Random Projection: W_orth @ q"]
    ProjQ --> KernelMapQ["Positive Exponential Map: h(q) · exp(...)"]
    KernelMapQ --> PhiQ["ϕ(q) ∈ ℝᵐ (m « N)"]
    
    InputK["Key Vector k ∈ ℝᵈ"] --> ProjK["Orthogonal Random Projection: W_orth @ k"]
    ProjK --> KernelMapK["Positive Exponential Map: h(k) · exp(...)"]
    KernelMapK --> PhiK["ϕ(k) ∈ ℝᵐ (m « N)"]
    
    PhiQ & PhiK --> InnerProd["Inner Product: ⟨ϕ(q), ϕ(k)⟩ ≈ exp(qᵀk / √d)"]
```

### The Problem
While Katharopoulos linear attention used $\text{elu}(x)+1$, it approximated an *arbitrary* polynomial kernel rather than the true **exponential Softmax kernel** ($\exp(q^T k / \sqrt{d})$). This kernel mismatch caused attention weights to "dilute" over long sequences, degrading perplexity on language tasks and preventing zero-shot replacement of attention in pretrained Transformers. 

Standard Random Fourier Features (RFF) could approximate exponential kernels, but trigonometric basis functions produce negative values ($\sin, \cos < 0$). In attention, negative kernel values cause catastrophic denominator cancellation ($z_t \to 0$), producing numerical infinities and model collapse.

### How It Works (The Intuition)
1. **FAVOR+ Mechanism**: Fast Attention Via Positive Orthogonal Random features decomposes the Softmax exponential kernel into positive expectations:
   $$\exp\left(\frac{q^T k}{\sqrt{d}}\right) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)} \left[ \phi(q)^T \phi(k) \right]$$
2. **Positive Random Features (PRF)**: The feature map $\phi(x)$ uses an exponential projection:
   $$\phi(x) = \frac{\exp(-\|x\|^2 / 2\sqrt{d})}{\sqrt{m}} \left[ \exp(\omega_1^T x), \exp(\omega_2^T x), \dots, \exp(\omega_m^T x) \right]^T$$
   Because $\exp(z) > 0$ for all real numbers, $\phi(x)$ is strictly non-negative, completely eliminating negative denominator collapse.
3. **Orthogonal Random Features (ORF)**: Instead of sampling random Gaussian vectors $\omega_i$ independently (which can cluster or align by chance), Performer enforces exact orthogonality across blocks of vectors ($W_{\text{orth}} W_{\text{orth}}^T = I$). This reduces the variance of the kernel approximation by over **10x** for the same number of features $m$.
4. **Linear Execution**: With $m \ll N$ (typically $m = 128$ or $256$), Performers compute $\phi(Q) \left( \phi(K)^T V \right)$ in $\mathcal{O}(N \cdot m \cdot d)$ time.

### The Math (Light)
$$\text{Softmax Kernel Factorization: } \exp\left(\frac{q^T k}{\sqrt{d}}\right) = \exp\left(-\frac{\|q\|^2}{2\sqrt{d}}\right) \cdot \exp\left(\frac{\|q+k\|^2}{2\sqrt{d}}\right) \cdot \exp\left(-\frac{\|k\|^2}{2\sqrt{d}}\right)$$

$$\text{FAVOR+ Feature Map: } \phi(x) = \frac{\exp\left(-\frac{\|x\|^2}{2\sqrt{d}}\right)}{\sqrt{m}} \left[ \exp\left(\omega_1^T x\right), \dots, \exp\left(\omega_m^T x\right) \right]^T \quad \text{where } \omega_i \in \mathbb{R}^d$$

$$\text{Orthogonal Constraint: } \Omega = \text{Gram-Schmidt}\left( G \right) \quad \text{such that } \Omega \Omega^T = I_m$$

$$\text{Performer Complexity: } \mathcal{O}(N \cdot m \cdot d) \quad (\text{Strictly linear in } N)$$

### Why It Matters
- **Drop-In Softmax Approximation**: Matches standard Transformer accuracy across language and vision benchmarks without modifying the Softmax loss objective.
- **Massive Context Scaling**: Enables processing ultra-long sequences ($N > 100\text{k}$) in genomics, AlphaFold-style protein folding, and high-resolution imaging.
- **Backward Compatible**: Pretrained standard Transformers can be converted to Performers with minimal fine-tuning steps.

### Image Prompt
> 🎨 **Image Prompt:** A 3D isometric scientific rendering of the FAVOR+ orthogonal kernel mechanism on dark navy slate `#0B0F19`. Glowing sphere showing multi-dimensional coordinate axes where unit vectors $\omega_1, \omega_2, \dots, \omega_m$ form perfectly perpendicular geometric laser beams in electric violet `#A855F7` and cyan `#00F5FF`. Raw token data passes through this spherical orthogonal filter, emerging as positive exponential energy waves that smoothly factorize a massive attention tensor into two slender hyper-efficient matrix flows. Clean mathematical formulas float in the ambient atmosphere.

---

## 3. Linformer Attention

```
Linformer Sequence Length Down-Projection:
   Q: [N × d]
   K: [N × d] ──[ Project E (k × N) ]──>  K_proj: [k × d]
   V: [N × d] ──[ Project F (k × N) ]──>  V_proj: [k × d]
   
   Attn Map = Softmax( Q @ K_proj.T / √d )  ──>  Shape: [N × k]  (k « N)
   Output   = Attn Map @ V_proj             ──>  Shape: [N × d]
```

### Title & Subtitle
**Linformer: Compressing Sequence Length via Low-Rank Space Projections**  
*Proving that attention matrices are intrinsically low-rank and down-projecting $N \times d$ keys and values to $k \times d$ landmarks.*

### The Analogy
Imagine writing a comprehensive index for a 10,000-page encyclopedia. Instead of cross-referencing every single word against all 10,000 pages, the editor creates a distilled table of **256 master topic landmarks** ($k=256$). 

Every page in the book ($N$) only needs to calculate its relevance to these 256 landmarks. The readers can quickly navigate anywhere in the entire 10,000-page encyclopedia by routing through this compact 256-landmark index.

```mermaid
flowchart LR
    subgraph FullAttention["Full Attention: N × N"]
        Q_full["Q: [N × d]"] --> Scores1["Q @ Kᵀ : [N × N]"]
        K_full["K: [N × d]"] --> Scores1
        Scores1 --> Out_full["Scores @ V : [N × d]"]
        V_full["V: [N × d]"] --> Out_full
    end
    
    subgraph LinformerAttention["Linformer: N × k"]
        K_in["K: [N × d]"] --> ProjE["Linear Proj E : [k × N]"] --> K_comp["K_proj: [k × d]"]
        V_in["V: [N × d]"] --> ProjF["Linear Proj F : [k × N]"] --> V_comp["V_proj: [k × d]"]
        Q_in["Q: [N × d]"] & K_comp --> Scores2["Q @ K_projᵀ : [N × k]"]
        Scores2 --> Softmax2["Softmax_k"] --> Out_comp["Attn @ V_proj : [N × d]"]
        V_comp --> Out_comp
    end
```

### The Problem
In standard Transformer layers, as sequence length $N$ grows, the rank of the $N \times N$ attention matrix does *not* grow proportionally. Singular value decomposition (SVD) of empirical attention maps reveals that the vast majority of spectral energy is concentrated in a tiny fraction of singular values. Computing full $N \times N$ pairwise interactions wastes massive compute on near-zero singular dimensions.

### How It Works (The Intuition)
1. **Mathematical Proof (Johnson-Lindenstrauss)**: Wang et al. proved that for any attention matrix $A \in \mathbb{R}^{N \times N}$, there exists a low-rank matrix $\widetilde{A} \in \mathbb{R}^{N \times k}$ with $k = \mathcal{O}(d / \epsilon^2)$ that approximates $A$ with error bounded by $\epsilon$, regardless of how large $N$ becomes!
2. **Dimension Reduction Projections**: Two learned linear projection matrices $E, F \in \mathbb{R}^{k \times N}$ project the sequence dimension of Keys and Values:
   $$\widetilde{K} = E K \in \mathbb{R}^{k \times d}, \quad \widetilde{V} = F V \in \mathbb{R}^{k \times d}$$
3. **Asymmetric Attention Computation**:
   - Queries $Q \in \mathbb{R}^{N \times d}$ multiply with compressed keys $\widetilde{K}^T \in \mathbb{R}^{d \times k}$, generating a small score matrix of shape $[N \times k]$.
   - Softmax is computed along the compact dimension $k$.
   - The $[N \times k]$ probability matrix multiplies compressed values $\widetilde{V} \in \mathbb{R}^{k \times d}$ to reconstruct the full $[N \times d]$ output.
4. **Parameter Sharing**: $E$ and $F$ can be shared across attention heads or across all layers to minimize parameter overhead.

### The Math (Light)
$$\widetilde{K}_h = E_h K_h \in \mathbb{R}^{k \times d_k}, \quad \widetilde{V}_h = F_h V_h \in \mathbb{R}^{k \times d_k} \quad (k \ll N)$$

$$\text{Linformer}(Q, K, V) = \text{Softmax}\left( \frac{Q (E K)^T}{\sqrt{d_k}} \right) (F V) \in \mathbb{R}^{N \times d}$$

$$\text{Time Complexity: } \mathcal{O}(N \cdot k \cdot d) \quad \text{vs.} \quad \text{Standard: } \mathcal{O}(N^2 \cdot d)$$

$$\text{Activation Memory: } \mathcal{O}(N \cdot k) \quad \text{vs.} \quad \text{Standard: } \mathcal{O}(N^2)$$

### Why It Matters
- **True $\mathcal{O}(N)$ Scaling**: Linear in sequence length $N$ while retaining exact row-wise Softmax normalization across dimension $k$.
- **85% Memory Reduction**: Enables handling $32\text{k}+$ token inputs on single GPU nodes for document classification and bidirectional encoding.
- **Ideal For Encoders & ViTs**: Excellent for Vision Transformers and BERT-style bidirectional encoders where sequence length is bounded at compile time.

### Image Prompt
> 🎨 **Image Prompt:** A 3D isometric diagram of the Linformer sequence compression architecture on a dark graphite background `#0B0F19`. A long translucent token ribbon of length $N = 8192$ enters from the left. Two glowing golden compression grates labeled "$E$" and "$F$" compress the length down into a sleek compact cylinder of size $k = 256$ in emerald neon `#00FF88`. Queries maintain their $N$ resolution, projecting onto the compact $k$ landmarks with precise luminous light beams. Crisp volumetric lighting, high-tech engineering schematic look.

---

## 4. Mamba & State Space Models

```
Continuous Dynamical System:          Discretized Selective Scan (ZOH):
   h'(t) = A h(t) + B x(t)               h_t = Ā_t h_{t-1} + B̄_t x_t
   y(t)  = C h(t)                        y_t = C_t h_t
                                      where (Δ_t, B_t, C_t) = f(x_t)  <-- Input Dependent!
```

### Title & Subtitle
**Mamba & Selective SSMs: Continuous Dynamical Systems with Input-Dependent Memory Gates**  
*How selective parameterization, hardware-aware parallel scans, and continuous state spaces dethroned Transformers on long-context modeling.*

### The Analogy
Imagine a courtroom stenographer. A standard Linear Time-Invariant (LTI) recorder writes down every background cough, door creak, and whispered comment at the exact same constant recording speed—quickly running out of paper. 

Mamba is an **intelligent selective stenographer**: when someone coughs or rambles about the weather, Mamba sets step-size $\Delta_t \approx 0$ (instantly ignoring the noise). But the moment the witness gives a critical testimony, Mamba cranks up $\Delta_t$, expands its selective memory $B_t$, and engraves the exact quote into the internal state matrix.

```mermaid
flowchart LR
    TokenInput["Input Token x_t"] --> GateDelta["Linear -> Softplus: Δ_t"]
    TokenInput --> GateB["Linear: B_t"]
    TokenInput --> GateC["Linear: C_t"]
    
    A_Matrix["Structured Matrix A (Diagonal/HiPPO)"] --> Discretize["Discretization: Ā_t = exp(Δ_t · A)"]
    GateDelta --> Discretize
    GateDelta & GateB --> DiscretizeB["B̄_t = Δ_t · B_t"]
    
    Discretize & DiscretizeB --> Recurrence["h_t = Ā_t · h_{t-1} + B̄_t · x_t"]
    PrevState["h_{t-1}"] --> Recurrence
    Recurrence --> Output["y_t = C_t · h_t"]
    GateC --> Output
```

### The Problem
Prior State Space Models (like S4 and H3) were **Linear Time-Invariant (LTI)**: their transition matrices $(A, B, C)$ were completely static and constant across all tokens. While LTI allowed S4 to train rapidly via global Fast Fourier Transform (FFT) 1D convolutions, it prevented the model from performing **content-aware reasoning** (e.g., selective token filtering, variable tracking, or the synthetic Associative Recall test).

### How It Works (The Intuition)
1. **The Selective Mechanism**: Gu & Dao made the SSM parameters dynamic functions of the current input token $x_t$:
   $$B_t = W_B x_t, \quad C_t = W_C x_t, \quad \Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta)$$
   - $\Delta_t$ acts as a continuous time-gate: large $\Delta_t$ causes the model to update state $h_t$ heavily with $x_t$; small $\Delta_t$ causes the model to ignore $x_t$ and retain past memory $h_{t-1}$.
2. **Discretization via Zero-Order Hold (ZOH)**: Converts continuous physical differential equations into discrete-time recurrences:
   $$\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = (\Delta_t A)^{-1}(\exp(\Delta_t A) - I) \cdot \Delta_t B_t \approx \Delta_t B_t$$
3. **Hardware-Aware Parallel Associative Scan**:
   - Because $(A_t, B_t, C_t)$ change at every token step, global FFT convolution is impossible.
   - Mamba solves this by fusing the recurrence loop into GPU **SRAM and Register memory**. Instead of writing intermediate hidden states $h_t \in \mathbb{R}^{B \times L \times D \times N}$ to slow High-Bandwidth Memory (HBM), the associative scan executes entirely in fast on-chip SRAM, yielding blistering training speeds.

### The Math (Light)
$$\text{Continuous State Equation: } \dot{h}(t) = A h(t) + B x(t), \quad y(t) = C h(t)$$

$$\text{ZOH Discretization: } \bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t$$

$$\text{Selective Recurrence: } h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t \in \mathbb{R}^N, \quad y_t = C_t h_t \in \mathbb{R}$$

$$\text{Complexity: } \mathcal{O}(L \cdot D \cdot N) \text{ time}, \quad \mathcal{O}(1) \text{ KV cache memory per generation step}$$

### Why It Matters
- **5x Inference Throughput**: Generates tokens at constant $O(1)$ memory with zero KV cache bloat.
- **Million-Token Contexts**: Linear scaling enables effortless scaling to context windows of $1,000,000+$ tokens on standard GPU hardware.
- **Matches Transformers at Scale**: First non-Transformer architecture to match or beat dense Transformer perplexity on large-scale language benchmarks.

### Image Prompt
> 🎨 **Image Prompt:** A stunning 3D isometric cutaway of the Mamba hardware-aware selective state space engine on dark slate `#0B0F19`. A stream of multi-colored glowing data tokens flows into a central silicon processor die. Dynamic glowing gate valves labeled "$\Delta_t$" and "$B_t$" open and close in real-time, selectively routing critical golden data particles into a crystalline high-speed SRAM accumulator chamber glowing with emerald neon `#00FF9D`. Faded unimportant tokens evaporate into cool blue vapor. High-contrast ray-traced lighting, futuristic chip architecture.

---

## 5. MambaSVD-Flash Hybrid

```
                      ┌──> Online SVD-SSM Recurrence [O(L·D·r)] ──> Y_ssm ──┐
                      │    A_t = U_{t,r} @ Σ_{t,r} @ V_{t,r}ᵀ               │
Input Tensor X ───────┤                                                     ├──> Fusion: γ ⊙ Y_ssm + (1-γ) ⊙ O_attn
                      │                                                     │
                      └──> FP8 FlashAttention-3 Path [High-Entropy] ──────> O_attn ─┘
```

### Title & Subtitle
**MambaSVD-Flash Hybrid: Fusing Selective State-Space Recurrence with Low-Rank Tensor Core Attention**  
*The dual-engine architecture combining the linear throughput of SVD-truncated SSM scans with the razor-sharp associative recall of FP8 FlashAttention-3.*

### The Analogy
Think of a high-performance Le Mans hybrid race car. On long, straight highway stretches, an ultra-efficient electric motor (the SVD-SSM path) delivers continuous linear torque with minimal fuel consumption. 

When entering tight, complex chicane curves requiring instantaneous explosive torque (complex multi-token syntactic lookups), the turbocharged combustion engine (FP8 FlashAttention-3) fires on all cylinders. A smart electronic ECU (learned gate $\gamma$) automatically distributes power between both engines in microseconds.

```mermaid
flowchart TD
    InputX["Input Sequence X ∈ ℝ^(B×L×D)"] --> SplitPath{"Dual Execution Split"}
    
    SplitPath --> PathSSM["SVD-SSM Linear Path"]
    PathSSM --> FactorA["Online SVD Factorization: A_t ≈ U_{t,r} Σ_{t,r} V_{t,r}ᵀ"]
    FactorA --> RecurrScan["Associative Low-Rank Scan (SRAM)"]
    RecurrScan --> Y_ssm["SSM Output: Y_ssm"]
    
    SplitPath --> PathFlash["High-Entropy Attention Path"]
    PathFlash --> QuantFP8["FP8 E4M3 Quantization"]
    QuantFP8 --> Flash3Kernel["Hopper FlashAttention-3 Kernel"]
    Flash3Kernel --> O_attn["Attention Output: O_attn"]
    
    InputX --> GateRouter["Gating Router: γ = σ(W_g X)"]
    
    Y_ssm & O_attn & GateRouter --> BlendedOut["Final Output: Y = γ ⊙ Y_ssm + (1 - γ) ⊙ O_attn"]
```

### The Problem
Pure State Space Models (Mamba) scale linearly, but they can experience memory dilution on complex "needle-in-a-haystack" associative retrieval tasks that dense Transformers solve effortlessly. Conversely, dense FlashAttention provides perfect retrieval but suffers from $\mathcal{O}(L^2)$ scaling. Furthermore, uncompressed SSM transition matrices introduce significant parameter footprint when hidden dimension $D$ expands to $8192+$.

### How It Works (The Intuition)
1. **Bifurcated Execution Pipeline**: The input sequence $X \in \mathbb{R}^{B \times L \times D}$ is fed in parallel to an SVD-compressed SSM path and a dense FP8 FlashAttention-3 path.
2. **SVD-Truncated SSM Recurrence**: The dense transition matrix $A_t \in \mathbb{R}^{D \times D}$ is dynamically factorized via online Singular Value Decomposition into rank-$r$ components ($r \ll D$):
   $$A_t = U_{t,r} \Sigma_{t,r} V_{t,r}^T$$
   State updates compute associative matrix-vector products through the low-rank bottleneck, reducing FLOPs from $\mathcal{O}(D^2)$ to $\mathcal{O}(D \cdot r)$.
3. **FP8 FlashAttention-3 Path**: For high-entropy tokens requiring exact retrieval, queries, keys, and values are quantized to 8-bit floating point (**FP8 E4M3**) and processed using NVIDIA Hopper asynchronous TMA (Tensor Memory Accelerator) clusters.
4. **Learned Gating Fusion**: A dynamic token-wise sigmoid gate $\gamma = \sigma(W_g X)$ arbitrates between the two streams:
   $$Y = \gamma \odot Y_{\text{ssm}} + (1 - \gamma) \odot O_{\text{attn}}$$

### The Math (Light)
$$\text{SVD State Factorization: } A_t = U_{t,r} \Sigma_{t,r} V_{t,r}^T \quad \text{where } U_{t,r} \in \mathbb{R}^{D \times r}, \; V_{t,r} \in \mathbb{R}^{r \times D}$$

$$\text{Low-Rank Recurrence: } h_t = U_{t,r} \left( \Sigma_{t,r} \left( V_{t,r}^T h_{t-1} \right) \right) + B_t x_t \in \mathbb{R}^D \implies \mathcal{O}(B \cdot L \cdot D \cdot r)$$

$$\text{FP8 Attention: } O_{\text{attn}} = \text{Softmax}\left(\frac{s_q s_k Q_{\text{fp8}} K_{\text{fp8}}^T}{\sqrt{d}}\right) (s_v V_{\text{fp8}})$$

$$\text{Gated Fusion: } Y = \sigma(W_g X) \odot (C_t h_t) + (1 - \sigma(W_g X)) \odot O_{\text{attn}}$$

### Why It Matters
- **7.75x Speedup at 8k Context**: Delivers $1.65\text{ ms}$ latency vs $12.8\text{ ms}$ for dense multi-head attention.
- **10x Memory Reduction**: Peak VRAM at $L=8192$ is only **320 MB** compared to **3,450 MB** for full MHA.
- **Lossless Retrieval**: Achieves $100\%$ accuracy on needle-in-a-haystack associative recall while operating at linear SSM scan speeds.

### Image Prompt
> 🎨 **Image Prompt:** A dual-engine 3D isometric architectural rendering on dark obsidian background `#0B0F19`. On the upper deck, a streamlined emerald neon pipeline represents the SVD-SSM path, passing through glowing crystalline prisms labeled "$U_r, \Sigma_r, V_r^T$". On the lower deck, an electric violet `#8B5CF6` Hopper microchip die executes FP8 FlashAttention-3 matrix tiles. In the center, a luminous holographic combiner gate labeled "$\gamma$-Router" fuses both data streams into an intense radiant output beam. Photorealistic cinematic lighting, 8k resolution.

---

## 6. RetNet: Retention Networks

```
RetNet's "Impossible Triangle" Solution:
                     [ Parallel Training ]
                          ▲         ▲
                         /           \
                        /   RetNet    \
                       /   Conquers   \
                      ▼     All 3!     ▼
     [ O(1) Inference ] ────────────── [ High Performance ]

Three Computation Paradigms:
1. Parallel:   R = (Q Kᵀ ⊙ D) V            (Training)
2. Recurrent:  S_n = γ S_{n-1} + K_nᵀ V_n  (Low-Latency Decoding)
3. Chunkwise:  Intra-Chunk Parallel + Inter-Chunk Recurrent (Long Context)
```

### Title & Subtitle
**RetNet: Conquering the Impossible Triangle with Retentive Networks**  
*Achieving parallel training, $O(1)$ inference cost, and Transformer-level performance through multi-scale decay retention.*

### The Analogy
Imagine a massive military communications network. During peacetime strategy sessions (training), all battle commanders analyze past history simultaneously via a satellite data feed (**Parallel Mode**). 

During frontline combat (inference), a radio operator carries a single compact tactical journal, updating it with each incoming transmission in 2 seconds flat (**Recurrent Mode**). 

When coordinating massive regional battalions (long context), each battalion processes its local troops in parallel while passing summary journals between battalions (**Chunkwise Mode**).

```mermaid
flowchart TD
    subgraph RetNetThreeParadigms["RetNet: Triple Execution Paradigms"]
        direction TB
        
        subgraph Mode1["1. Parallel Mode (Training)"]
            Q1["Q"] & K1["K"] --> Mult1["Q @ Kᵀ"]
            Mult1 & D_mat["Decay Matrix D (γ^(n-m))"] --> MaskedScores["(Q @ Kᵀ) ⊙ D"]
            MaskedScores & V1["V"] --> OutParallel["R = ((Q @ Kᵀ) ⊙ D) @ V"]
        end
        
        subgraph Mode2["2. Recurrent Mode (O(1) Inference)"]
            Kn["K_n"] & Vn["V_n"] --> OuterP["K_nᵀ @ V_n"]
            OuterP & PrevS["γ · S_{n-1}"] --> NewS["S_n = γ · S_{n-1} + K_nᵀ V_n"]
            Qn["Q_n"] & NewS --> OutRecurrent["R_n = Q_n @ S_n"]
        end
        
        subgraph Mode3["3. Chunkwise Mode (Long Context)"]
            ChunkParallel["Intra-Chunk Matrix Multiply (Parallel)"]
            ChunkRecurrent["Inter-Chunk State Propagation (Recurrent)"]
            ChunkParallel & ChunkRecurrent --> OutChunk["Fused Chunk Output"]
        end
    end
```

### The Problem
The fundamental trilemma of sequence modeling (the "Impossible Triangle") states that an architecture cannot simultaneously achieve:
1. **Parallel Training** (like Transformers via matrix multiplication).
2. **$O(1)$ Inference Cost** (like RNNs via constant memory state).
3. **Competitive Performance** (matching standard dense Transformer perplexity).

Transformers sacrifice (2); RNNs sacrifice (1); simple linear attentions sacrifice (3).

### How It Works (The Intuition)
1. **The Retention Mechanism**: RetNet replaces Multi-Head Attention with **MultiScale Retention (MSR)**, introducing an explicit scalar exponential decay parameter $\gamma \in (0, 1)$.
2. **Explicit Complex Positional Encoding**: Incorporates rotation angles $\theta$ and decay $\gamma$ directly into the key-query formulation:
   $$Q_n = (X_n W_Q) e^{i n \theta}, \quad K_n = (X_n W_K) e^{-i n \theta}$$
3. **Triple Representation**:
   - **Parallel View (Training)**: Evaluates full sequence interactions via masked matrix multiplication $(Q K^T \odot D) V$, where decay matrix $D_{n,m} = \gamma^{n-m}$ for $n \ge m$.
   - **Recurrent View (Inference)**: Decodes token-by-token in constant memory by accumulating state $S_n = \gamma S_{n-1} + K_n^T V_n \in \mathbb{R}^{d \times d}$.
   - **Chunkwise View (Serving)**: Breaks long sequences into chunks of size $B$. Calculates intra-chunk attention with fast Tensor Core GEMMs while propagating recurrent states across chunk boundaries.

### The Math (Light)
$$\text{Multi-Scale Decay: } \gamma_k = 1 - 2^{-5 - \text{arange}(0, H)} \quad \text{across heads } k \in [1, H]$$

$$\text{Parallel Form: } R = \left( Q K^T \odot D \right) V \quad \text{where } D_{n,m} = \begin{cases} \gamma^{n-m} & n \ge m \\ 0 & n < m \end{cases}$$

$$\text{Recurrent Form: } S_n = \gamma S_{n-1} + K_n^T V_n \in \mathbb{R}^{d \times d}, \quad \text{RetNet}(X)_n = Q_n S_n \in \mathbb{R}^d$$

$$\text{Chunkwise Form: } R_i = \underbrace{\left( Q_{[i]} K_{[i]}^T \odot D_{\text{intra}} \right) V_{[i]}}_{\text{Intra-Chunk (Parallel)}} + \underbrace{\left( Q_{[i]} S_{i-1} \right) \odot \xi}_{\text{Inter-Chunk (Recurrent)}}$$

### Why It Matters
- **8.4x Higher Inference Throughput**: Slashes decoding memory by **70%** and latency by **8x** compared to Transformers.
- **3.4x Faster Training**: Outperforms FlashAttention-2 during training on sequences exceeding 8k tokens.
- **Zero Performance Degradation**: Matches LLaMA-style dense Transformer scaling curves across 1.3B, 2.7B, and 6.7B parameters.

### Image Prompt
> 🎨 **Image Prompt:** A 3D isometric infographic illustrating the "Impossible Triangle" conquered by RetNet on a dark slate background `#0B0F19`. A glowing triangular pedestal of polished obsidian has three illuminated vertices labeled "Parallel Training", "O(1) Inference Cost", and "LLaMA-Grade Quality". In the center, a radiant glowing golden core labeled "RetNet MSR" emits three distinct cyan laser pulses connecting all three vertices simultaneously. Floating mathematical matrices with decay gradient waves $(\gamma^{n-m})$ shimmer in the surrounding space.

---

## 7. RWKV Time-Mixing

```
RWKV 4-Vector Architecture:
   r_t = σ( W_r · Mix_r(x_t, x_{t-1}) )   ──> Receptance (Dynamic Read Gate)
   w   = exp( -exp( W_w ) )               ──> Weight Decay (Continuous Time Decay)
   k_t = W_k · Mix_k(x_t, x_{t-1})        ──> Key (Information Attractor)
   v_t = W_v · Mix_v(x_t, x_{t-1})        ──> Value (Raw Information Content)

Recurrent WKV Operator:
   a_t = exp(-w) · a_{t-1} + exp(k_t) · v_t
   b_t = exp(-w) · b_{t-1} + exp(k_t)
   wkv_t = (a_{t-1} + exp(u + k_t) · v_t) / (b_{t-1} + exp(u + k_t))
   y_t   = r_t ⊙ wkv_t
```

### Title & Subtitle
**RWKV: Reinventing RNNs for the Transformer Era**  
*How Time-Mixing, token shifts, and channel-wise exponential decays deliver 100% Transformer expressiveness in a pure RNN form.*

### The Analogy
Imagine a master sound technician operating a professional audio mixing console. 

For each incoming audio channel, **Receptance ($R$)** acts as the volume fader that controls how much sound reaches the master output. **Weight ($W$)** is the reverb decay tail setting how long an echo lingers in memory. **Key ($K$)** sets the frequency filter, and **Value ($V$)** is the pure analog instrument signal. 

Before processing, the technician uses a **Time-Shift** dial to blend 30% of the previous token's acoustic vibration with the current note, ensuring seamless musical harmony across time.

```mermaid
flowchart TD
    CurrentX["Current Token x_t"] & PrevX["Previous Token x_{t-1}"] --> TimeMix["Time-Shift Mixing: μ ⊙ x_t + (1-μ) ⊙ x_{t-1}"]
    
    TimeMix --> R_Gate["Receptance: r_t = σ(W_r · x')"]
    TimeMix --> K_Proj["Key: k_t = W_k · x'"]
    TimeMix --> V_Proj["Value: v_t = W_v · x'"]
    TimeMix --> W_Decay["Time-Decay: w (channel-wise)"]
    
    K_Proj & V_Proj & W_Decay --> WKV_Scan["WKV State Scan: a_t = e^(-w)a_{t-1} + e^(k_t)v_t"]
    PrevState["State: (a_{t-1}, b_{t-1})"] --> WKV_Scan
    
    WKV_Scan --> WKV_Out["wkv_t Output"]
    R_Gate & WKV_Out --> MultiplyGate["Elementwise: y_t = r_t ⊙ (W_o · wkv_t)"]
```

### The Problem
Traditional RNNs (LSTM, GRU) suffer from **Backpropagation Through Time (BPTT)** bottlenecks: training cannot be parallelized across the time dimension because hidden state $h_t$ is entangled with non-linear activation functions ($\tanh, \sigma$) inside the recurrent loop. Conversely, Transformers parallelize training perfectly but suffer from massive, growing KV caches during generation.

### How It Works (The Intuition)
1. **Token Shift (Time-Mix)**: Every linear projection mixes the current token $x_t$ with the preceding token $x_{t-1}$ using a learned interpolation vector $\mu \in [0, 1]^D$:
   $$x'_t = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$$
2. **Channel-Wise Exponential Decay ($w$)**: Each feature channel in the hidden representation decays at an independent, learned exponential rate $e^{-w}$.
3. **The WKV Attention Mechanism**: Formulates attention as a channel-wise rational function:
   $$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u + k_t}}$$
   where $u$ is a special bonus weight prioritizing the current instant token.
4. **Receptance Gating**: The output is modulated by an input-dependent sigmoid gate: $y_t = \sigma(r_t) \odot (W_o \cdot wkv_t)$.
5. **Duality**: Parallel prefix scan during training; constant $\mathcal{O}(1)$ memory state $(a_t, b_t \in \mathbb{R}^D)$ during inference.

### The Math (Light)
$$\text{Linear Interpolation: } x'_{r,t} = \mu_r \odot x_t + (1 - \mu_r) \odot x_{t-1}$$

$$r_t = W_r x'_{r,t}, \quad k_t = W_k x'_{k,t}, \quad v_t = W_v x'_{v,t}$$

$$\text{State Updates: } a_t = e^{-w} \odot a_{t-1} + e^{k_t} \odot v_t \in \mathbb{R}^D, \quad b_t = e^{-w} \odot b_{t-1} + e^{k_t} \in \mathbb{R}^D$$

$$\text{Token Output: } wkv_t = \frac{a_{t-1} + e^{u + k_t} \odot v_t}{b_{t-1} + e^{u + k_t}}, \quad y_t = \sigma(r_t) \odot \left( W_o \cdot wkv_t \right)$$

### Why It Matters
- **Zero KV Cache Memory**: Runs multi-billion parameter foundation models on low-power devices, Raspberry Pis, and mobile phones with fixed RAM usage.
- **Infinite Generation Length**: Generates millions of continuous tokens with zero slowdown and zero memory footprint growth.
- **Battle-Tested Open Source Models**: Production-ready RWKV-v4, v5, v6 (Finch), and v7 architectures matching LLaMA benchmarks across multilingual datasets.

### Image Prompt
> 🎨 **Image Prompt:** A 3D isometric representation of the RWKV Time-Mixing neural architecture on dark navy slate `#0B0F19`. Four glowing fluorescent laser conduits labeled "$R$ (Receptance)", "$W$ (Decay)", "$K$ (Key)", and "$V$ (Value)" converge into a high-speed channel-wise accumulator die. A holographic wave visualization illustrates the Time-Shift mechanism blending the current token with a soft glowing ghost of the previous token $x_{t-1}$. Neon amber and cyan lighting, hyper-detailed circuit pathways, 8k ray-tracing.

---

## 8. HGRN2 & Gated DeltaNet

```
Gated DeltaNet Error-Driven Memory Update:
   Memory Error:   e_t = v_t - H_{t-1} @ k_t         (Difference from expectation!)
   State Update:   H_t = α_t · H_{t-1} + β_t · (e_t ⊗ k_t)
                       = α_t · H_{t-1} (I - (β_t/α_t) · k_t k_tᵀ) + β_t · (v_t ⊗ k_t)
   Query Readout:  y_t = q_t @ H_t
```

### Title & Subtitle
**HGRN2 & Gated DeltaNet: Error-Driven Associative Memory with Hierarchical Gating**  
*Combining the Delta Rule's selective forgetting with multi-layer hierarchical recurrence for superior associative recall.*

### The Analogy
Imagine maintaining a live whiteboard of company employee assignments. 

In simple linear attention, every time someone changes their department, you just write the new department over the board without erasing the old one (additive memory). Over time, the whiteboard becomes an illegible mess of conflicting text (eigenvalue saturation). 

The **Gated DeltaNet** algorithm looks at the whiteboard first, checks what department was previously listed for the employee, calculates the **memory prediction error** ($v_t - H_{t-1} k_t$), cleanly erases the obsolete assignment, and writes *only the net correction*.

```mermaid
flowchart TD
    QueryK["Current Key k_t"] & PrevMem["Past Memory Matrix H_{t-1}"] --> PredictVal["Memory Prediction: v̂_t = H_{t-1} @ k_t"]
    PredictVal & TrueVal["Target Value v_t"] --> ComputeErr["Error Vector: e_t = v_t - v̂_t"]
    
    ComputeErr & QueryK --> OuterCorrection["Error Outer Product: e_t ⊗ k_tᵀ"]
    
    PrevMem & ForgetGate["Retention Gate: α_t"] --> DecayState["α_t · H_{t-1}"]
    OuterCorrection & LearnRate["Update Rate: β_t"] --> ScaleCorrection["β_t · (e_t ⊗ k_tᵀ)"]
    
    DecayState & ScaleCorrection --> NewMem["Updated State: H_t = α_t H_{t-1} + β_t (e_t ⊗ k_tᵀ)"]
    QueryQ["Query q_t"] & NewMem --> Readout["Readout: y_t = q_t @ H_t"]
```

### The Problem
Simple linear attention and SSMs use **purely additive memory updates**:
$$H_t = H_{t-1} + k_t v_t^T$$
Because matrices add continuously without subtraction, memory states suffer from **eigenvalue explosion** over long sequences. More critically, additive memory cannot perform *destructive overwriting* (e.g., updating a variable `x = 5` to `x = 10` in code), causing linear models to fail on Multi-Query Associative Recall (MQAR) benchmarks.

### How It Works (The Intuition)
1. **The Delta Rule (Schlag et al.)**: Instead of blindly adding $k_t v_t^T$, DeltaNet tests its existing memory:
   $$\hat{v}_t = H_{t-1} k_t$$
   It computes the error vector $e_t = v_t - \hat{v}_t$. If the memory already knows the fact ($\hat{v}_t = v_t$), $e_t = 0$ and no memory capacity is wasted!
2. **Gated DeltaNet**: Augments the Delta update with input-dependent forget gating $\alpha_t \in (0, 1)$ and adaptive learning rate $\beta_t$:
   $$H_t = \alpha_t H_{t-1} + \beta_t (v_t - H_{t-1} k_t) k_t^T = H_{t-1} \left( \alpha_t I - \beta_t k_t k_t^T \right) + \beta_t v_t k_t^T$$
   The term $(I - \beta_t k_t k_t^T)$ acts as a **Householder projection** that directly subtracts old associations from memory before adding new ones.
3. **HGRN-2 (Hierarchical Gated Linear Recurrence)**: Employs 1D element-wise hierarchical gating with explicit state bounding:
   $$h_t = \text{Clamp}\left( f_t \odot h_{t-1} + i_t \odot x_t, -1.0, 1.0 \right)$$
   preventing activation explosion and supporting ultra-deep recurrent stacks.
4. **Hybrid Interleaving**: Stacks $R$ linear Gated DeltaNet/HGRN2 layers with 1 dense FlashAttention layer ($R:1$ ratio) to guarantee 100% precision on complex multi-hop retrieval.

### The Math (Light)
$$\text{Memory Prediction Error: } e_t = v_t - H_{t-1} k_t \in \mathbb{R}^d$$

$$\text{Gated Delta Update: } H_t = \alpha_t H_{t-1} + \beta_t e_t k_t^T = H_{t-1} \left( \alpha_t I - \beta_t k_t k_t^T \right) + \beta_t v_t k_t^T \in \mathbb{R}^{d \times d}$$

$$\text{HGRN-2 Recurrence: } f_t = \sigma(W_f x_t), \quad i_t = \sigma(W_i x_t), \quad h_t = \text{Clamp}\left( f_t \odot h_{t-1} + i_t \odot x_t, -1, 1 \right)$$

$$\text{Query Output: } y_t = q_t H_t \in \mathbb{R}^d \quad \text{or} \quad y_t = W_o (h_t \odot \sigma(W_g x_t))$$

### Why It Matters
- **Flawless Associative Recall**: Solves the MQAR and phone-book lookup tasks where conventional SSMs and linear attentions fall behind.
- **Accurate Code & State Tracking**: Dynamically updates and overwrites variable assignments and syntactic scope in long code repositories.
- **Chunkwise Parallel Training**: Executes in parallel via custom Triton chunk-scans, matching FlashAttention speeds during training.

### Image Prompt
> 🎨 **Image Prompt:** A 3D isometric scientific illustration of the Gated DeltaNet error-correction memory update on dark slate `#0B0F19`. A central crystalline memory slab labeled "$H_t$" glows with cyan internal matrices. An incoming key-value beam tests the slab, projecting a subtle red warning beam labeled "Prediction Error $e_t = v_t - H_{t-1}k_t$". A sleek emerald laser immediately performs an orthogonal Householder subtraction $(I - \beta k k^T)$, erasing the error before inscribing a pristine golden correction into the crystal. High-tech HUD annotations, crisp ambient lighting.

---

## 9. Griffin & Hawk

```
Google DeepMind's State Space Duality (SSD):
   1D Selective SSM Recurrence  <═══════════════>  2D Matrix-State Linear Attention
      h_t = α_t · h_{t-1} + B_t · x_t                  H_t = α_t · H_{t-1} + k_t ⊗ v_t
      y_t = C_t · h_t                                  y_t = q_t @ H_t
   
Griffin Hybrid Architecture Stack:
   Layer l:     [ Real-Gated LRU Recurrence ]  ──> O(L) Linear Scan
   Layer l+1:   [ Real-Gated LRU Recurrence ]  ──> O(L) Linear Scan
   Layer l+2:   [ Local Multi-Query Attention ] ──> Exact Local Syntax (Window = 1024)
```

### Title & Subtitle
**Griffin, Hawk & State Space Duality: Unifying SSMs and Linear Attention**  
*Google DeepMind's discovery of the mathematical bridge between 1D state spaces and 2D matrix linear attention.*

### The Analogy
Imagine two groups of astronomers studying the same distant galaxy. The radio astronomers (SSM camp) track the continuous radio frequency signal arriving over time (1D state recurrence). The optical astronomers (Attention camp) capture a high-resolution photographic plate showing all spatial stars at once (2D matrix attention). 

DeepMind's **State Space Duality (SSD)** proved that both groups were measuring the exact same physical reality through two different coordinate systems, unifying them into a single telescope!

```mermaid
flowchart LR
    subgraph SSD_Duality["State Space Duality (SSD) Framework"]
        direction TB
        SSM_View["SSM 1D Recurrence View:
        h_t = α_t · h_{t-1} + β_t · x_t
        y_t = C_t · h_t
        (O(1) Memory Inference)"]
        
        Equiv["⟺  MATHEMATICALLY EQUIVALENT  ⟺"]
        
        Attn_View["Linear Attention 2D State View:
        H_t = α_t · H_{t-1} + k_t ⊗ v_tᵀ
        y_t = q_t @ H_t
        (Parallel Tensor Core Training)"]
        
        SSM_View --- Equiv --- Attn_View
    end
```

### The Problem
For years, AI researchers treated **State Space Models (Mamba, S4)** and **Linear Attention (RetNet, Linear Transformers)** as two completely distinct, competing model families. SSMs operated on 1D vector states with continuous dynamical systems theory, while Linear Attentions operated on 2D outer-product matrix states. This fragmentation prevented researchers from combining the parallel training efficiency of Attention with the linear inference efficiency of SSMs.

### How It Works (The Intuition)
1. **State Space Duality (SSD)**: Dao & Gu (2024) and De et al. (DeepMind, 2024) proved that structured SSMs with diagonal/scalar transitions are **isomorphic to 1-semiseparable masked Linear Attention**:
   $$Y = \left( (Q K^T) \odot M_{\text{decay}} \right) V \Longleftrightarrow H_t = \alpha_t H_{t-1} + k_t v_t^T, \quad y_t = q_t H_t$$
2. **Hawk Architecture**: A pure non-transformer model replacing attention entirely with **Real-Gated Linear Recurrent Units (RG-LRU)**:
   - Recurrence: $h_t = \alpha_t \odot h_{t-1} + \sqrt{1 - \alpha_t^2} \odot (i_t \odot x_t)$
   - Where $\alpha_t = \sigma(W_a x_t)^c$ provides stable, bounded memory retention.
3. **Griffin Architecture**: A hybrid architecture that alternates **two RG-LRU recurrent blocks** with **one Local Multi-Query Attention block** (sliding window of 1024 tokens).
4. **Chunked 2D Scan Kernel**: Uses SSD to compute intra-chunk token interactions via high-throughput Tensor Core matrix multiplications (GEMM) while propagating scalar decays across chunk states.

### The Math (Light)
$$\text{RG-LRU Gate: } a_t = \sigma(W_a x_t), \quad \alpha_t = a_t^c \quad (c \text{ is a constant scalar, e.g. } 8)$$

$$\text{Input Modulation: } i_t = \sigma(W_i x_t), \quad \tilde{x}_t = W_x x_t$$

$$\text{Recurrent State Update: } h_t = \alpha_t \odot h_{t-1} + \sqrt{1 - \alpha_t^2} \odot \left( i_t \odot \tilde{x}_t \right) \in \mathbb{R}^D$$

$$\text{Griffin Hybrid Layout: } X^{(l+1)} = \begin{cases} \text{RG-LRU-Block}\left(X^{(l)}\right) & l \pmod 3 \in \{0, 1\} \\ \text{Local-MQA}\left(X^{(l)}, W=1024\right) & l \pmod 3 = 2 \end{cases}$$

### Why It Matters
- **Outperforms LLaMA-2**: Griffin achieves lower perplexity than LLaMA-2 models with significantly fewer training FLOPs.
- **Superior Context Extrapolation**: Trained on 8k contexts, Griffin extrapolates cleanly to **65,000+ tokens** without fine-tuning.
- **Unified Theoretical Framework**: Eliminates the division between SSMs and Attention, unlocking hybrid chunked GPU kernels that dominate modern LLM serving.

### Image Prompt
> 🎨 **Image Prompt:** A majestic 3D isometric scientific rendering of the State Space Duality (SSD) bridge on dark slate `#0B0F19`. On the left side, a glowing golden helical wave represents continuous 1D SSM dynamical systems. On the right side, a grid of luminous electric blue matrix tiles represents 2D Linear Attention. In the center, an elegant mobius strip labeled "State Space Duality (SSD)" bridges both representations into a single harmonious mathematical loop. Golden light particles transition fluidly between the wave and the matrix. 8k resolution, cinematic atmosphere.

---

## 💻 Practical PyTorch Implementation: Linear Recurrent Attention Engine

Here is a complete, modular, standalone PyTorch implementation demonstrating **Associative Linear Attention** with dual execution modes: **Parallel Training Scan** and **$O(1)$ Memory Autoregressive Step**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class KernelLinearAttention(nn.Module):
    """
    Katharopoulos-style Kernelized Linear Attention with dual execution:
    1. Parallel forward pass for training via cumulative associative prefix sum.
    2. Recurrent step-by-step forward pass for O(1) memory autoregressive decoding.
    """
    def __init__(self, d_model: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        # Projections for Q, K, V, and Output
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative kernel feature map: phi(x) = elu(x) + 1.0"""
        return F.elu(x) + 1.0

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel Training Forward Pass: O(N · d^2) FLOPs, O(N · d) memory.
        x: [Batch, SeqLen, Dim]
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project and reshape: [B, H, N, d]
        Q = self.feature_map(self.q_proj(x).view(B, N, H, d).transpose(1, 2))
        K = self.feature_map(self.k_proj(x).view(B, N, H, d).transpose(1, 2))
        V = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        # Compute outer product per token: K^T ⊗ V -> [B, H, N, d, d]
        KV = torch.einsum('bhni,bhnj->bhnij', K, V)
        
        # Causal prefix sum scan along sequence dimension N
        S = torch.cumsum(KV, dim=2) # Cumulative memory state [B, H, N, d, d]
        Z = torch.cumsum(K, dim=2)  # Cumulative normalizer [B, H, N, d]

        # Query state contraction: Q @ S -> [B, H, N, d]
        num = torch.einsum('bhni,bhnij->bhnj', Q, S)
        den = torch.einsum('bhni,bhni->bhn', Q, Z).unsqueeze(-1) + self.eps

        out = num / den # [B, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def forward_step(
        self, 
        x_t: torch.Tensor, 
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoregressive Single Token Step: O(1) Memory Footprint, O(d^2) Compute.
        x_t: [Batch, 1, Dim]
        state: Tuple of (S_t [B, H, d, d], z_t [B, H, d, 1])
        """
        B, _, D = x_t.shape
        H, d = self.num_heads, self.head_dim

        # Project single token
        q_t = self.feature_map(self.q_proj(x_t).view(B, H, d))
        k_t = self.feature_map(self.k_proj(x_t).view(B, H, d))
        v_t = self.v_proj(x_t).view(B, H, d)

        # Initialize state if first token
        if state is None:
            S_prev = torch.zeros(B, H, d, d, device=x_t.device, dtype=x_t.dtype)
            z_prev = torch.zeros(B, H, d, device=x_t.device, dtype=x_t.dtype)
        else:
            S_prev, z_prev = state

        # Recurrent state accumulation: S_t = S_{t-1} + k_t v_t^T
        S_t = S_prev + torch.einsum('bhi,bhj->bhij', k_t, v_t)
        z_t = z_prev + k_t

        # Output extraction: y_t = (q_t S_t) / (q_t z_t)
        num = torch.einsum('bhi,bhij->bhj', q_t, S_t)
        den = torch.einsum('bhi,bhi->bh', q_t, z_t).unsqueeze(-1) + self.eps

        out = (num / den).view(B, 1, D)
        return self.out_proj(out), (S_t, z_t)

# --- Verification & Equivalence Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, D = 2, 8, 32
    layer = KernelLinearAttention(d_model=D, num_heads=4)
    x = torch.randn(B, N, D)

    # 1. Parallel Training Pass
    parallel_out = layer.forward_parallel(x)

    # 2. Sequential Recurrent Step-by-Step Pass
    recurrent_outs = []
    state = None
    for t in range(N):
        x_t = x[:, t:t+1, :]
        out_t, state = layer.forward_step(x_t, state)
        recurrent_outs.append(out_t)
    recurrent_out = torch.cat(recurrent_outs, dim=1)

    # 3. Check Mathematical Equivalence
    max_diff = (parallel_out - recurrent_out).abs().max().item()
    print(f"Parallel vs. Recurrent Max Absolute Difference: {max_diff:.8e}")
    assert max_diff < 1e-5, "Parallel and Recurrent outputs must match!"
    print("✅ Mathematical equivalence between parallel scan and recurrent step confirmed!")
```

---

## 🏁 Architectural Master Comparison Table

| Architecture | Time Complexity | Memory Complexity | Parallel Training? | Inference State Size | Decay Mechanism | Primary Innovation & Best Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard Softmax MHA** | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(N^2)$ | ✅ Full Matrix | $\mathcal{O}(N \cdot d)$ (Expanding) | None (Dense Softmax) | Exact global attention; short contexts ($<4\text{k}$) |
| **Katharopoulos Linear** | $\mathcal{O}(N d^2)$ | $\mathcal{O}(N d)$ | ✅ Prefix Scan | $\mathcal{O}(d^2)$ (Constant) | None (Unweighted Sum) | Kernel feature map associativity; fast generation |
| **Performer (FAVOR+)** | $\mathcal{O}(N m d)$ | $\mathcal{O}(N m)$ | ✅ Matrix Product | $\mathcal{O}(m \cdot d)$ (Constant) | None (Softmax Approx) | Positive orthogonal random features; biology & genomics |
| **Linformer** | $\mathcal{O}(N k d)$ | $\mathcal{O}(N k)$ | ✅ Matrix Product | $\mathcal{O}(k \cdot d)$ (Constant) | None (Low-Rank Proj) | Sequence down-projection $N \to k$; BERT & ViT encoders |
| **Mamba (Selective SSM)** | $\mathcal{O}(N \cdot D \cdot N_{\text{state}})$ | $\mathcal{O}(1)$ | ✅ Fused Scan | $\mathcal{O}(D \cdot N_{\text{state}})$ | Input-Dependent $\Delta_t$ | Selective time-varying ODEs; million-token LLMs |
| **MambaSVD-Flash Hybrid** | $\mathcal{O}(N D r)$ | $\mathcal{O}(1)$ | ✅ Dual-Engine | $\mathcal{O}(D \cdot r)$ | Truncated SVD $\Sigma_r$ | SVD SSM scan + FP8 FlashAttention-3; high-entropy LLMs |
| **RetNet (Retention)** | $\mathcal{O}(N d^2)$ | $\mathcal{O}(d^2)$ | ✅ Triple-Mode | $\mathcal{O}(d^2)$ (Constant) | Multi-Scale $\gamma^{n-m}$ | "Impossible Triangle" solver; parallel + chunkwise + $O(1)$ |
| **RWKV (v4–v7)** | $\mathcal{O}(N d)$ | $\mathcal{O}(1)$ | ✅ WKV Scan | $\mathcal{O}(d)$ (Constant) | Learned Channel $e^{-w}$ | Time-Shift token mixing + pure RNN form; Edge AI & Phones |
| **HGRN2 & Gated DeltaNet** | $\mathcal{O}(N d^2)$ | $\mathcal{O}(d^2)$ | ✅ Chunk Scan | $\mathcal{O}(d^2)$ (Constant) | Error-Driven Delta Rule | Destructive memory overwriting; code & multi-query recall |
| **Griffin & Hawk** | $\mathcal{O}(N d)$ | $\mathcal{O}(1)$ | ✅ SSD Scan | $\mathcal{O}(d)$ (Constant) | Real-Gated $\alpha_t^c$ | State Space Duality bridge + Local MQA hybrid; LLaMA alternative |

---

*Authored by the SVD Linear Attention Research Group. Next in the series: **Part 3: Sparse, Chunked, Local & Block-Diagonal Attentions**.*
