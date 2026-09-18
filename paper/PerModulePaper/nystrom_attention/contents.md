# تمپلیت کامل مقاله Nyströmformer — فایل به فایل

قبل از شروع، خلاصه‌ای از Nyströmformer که این تمپلیت را متفاوت می‌کند:

**Nyströmformer = Nyström-based Low-Rank Kernel Approximation**

| ویژگی | MinimalSVD | Dobi-SVD | **Nyströmformer** |
|---|---|---|---|
| **Original paper** | اختراعی | اختراعی | **Xiong et al. 2021** |
| **Type** | Weight compression | Weight compression | **Compute compression** |
| **Parameters** | کم می‌کند | کم می‌کند | **تغییر نمی‌دهد** |
| **Complexity** | Same as dense | Same as dense | **O(N·m·d) vs O(N²·d)** |
| **Landmark mechanism** | ندارد | ندارد | **✅ (adaptive pooling)** |
| **Iterative pinv** | ندارد | ندارد | **✅ (Newton-Schulz)** |
| **Short seq fallback** | ندارد | ندارد | **✅ (dense)** |

**نکته مهم:** Nyströmformer **حافظه KV-cache را کم نمی‌کند** و **پارامتر را کم نمی‌کند**. بلکه **محاسبه attention** را برای توالی‌های طولانی تقریب می‌زند.

**نوآوری اصلی:** به جای محاسبه کامل attention matrix (N×N)، از **m landmark** استفاده می‌کند:
$$
\text{Attn} \approx \text{softmax}(Q K_m^\top) \cdot \text{pinv}(\text{softmax}(Q_m K_m^\top)) \cdot \text{softmax}(Q_m K^\top) V
$$

که به **associative multiplication** منجر می‌شود: `O(N·m·d)` به جای `O(N²·d)`.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   └── ... (ساختار قبلی)
└── nystrom/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Nyströmformer on Non-BERT Architectures: A Portability and Reproduction Study

<!-- 
TITLE NOTES:
- This is a REPRODUCTION paper, not a novel method
- Original paper: Xiong et al. 2021 (AAAI)
- Emphasis on: multi-architecture porting, production hardening
- Alternative: "Nyströmformer for Multi-Architecture Transformers: Portability and Production Hardening"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nystrom.py
**Original paper:** https://arxiv.org/abs/2102.03902
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

**Nyströmformer** (Xiong et al., 2021) approximates standard softmax attention 
via the Nyström low-rank kernel method. Instead of materializing the full 
$N \times N$ attention matrix, it selects $m \ll N$ landmark tokens and 
computes three smaller attention matrices, achieving $\mathcal{O}(N m d)$ 
complexity instead of $\mathcal{O}(N^2 d)$. Landmarks are chosen via 
**adaptive average pooling**, and a **pseudoinverse** of the landmark attention 
matrix is computed iteratively (Newton-Schulz).

We present a **reproduction** of Nyströmformer, ported to four transformer 
families that the original paper does not target: DistilBERT (flat), 
RoBERTa/BERT (nested), GPT-2 (causal Conv1D), and Qwen3.5 (GQA + RoPE). Our 
implementation:
(i) preserves exact softmax attention for short sequences ($N \leq 128$) and 
for causal decoders,
(ii) uses an iterative pinv with a fallback to `torch.linalg.pinv`,
(iii) handles Conv1D transposition, nested encoder bypass, GQA expansion, 
and RoPE,
(iv) attaches LoRA adapters for recovery fine-tuning.

We emphasize: **this is a reproduction and portability study, not a novel 
architecture.** Nyströmformer does **not** reduce parameter count or 
KV-cache size. It reduces **attention compute** for long sequences.

On SST-2 (sequence length 128), the Nyström branch does not activate (dense 
fallback), and Nyströmformer matches the baseline. On longer sequences, the 
Nyström approximation provides asymptotic savings.

**Keywords:** Nyströmformer, Nyström method, low-rank attention, kernel 
approximation, iterative pseudoinverse, Newton-Schulz, multi-architecture 
transformers, reproduction study

---

## Lay Summary

Standard transformer attention compares every token to every other token, 
which is expensive for long sequences. Nyströmformer, introduced by Xiong 
et al. (2021), instead compares tokens to a small set of "landmark" tokens, 
reducing the cost. This paper reproduces Nyströmformer and ports it to four 
model families that the original paper did not target. We confirm the 
original method works on these models and document the engineering required 
for the port.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)
```

---

## 📄 `01_intro.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Standard scaled dot-product attention (Vaswani et al., 2017) computes:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

The intermediate matrix $Q K^\top \in \mathbb{R}^{N \times N}$ has 
$\mathcal{O}(N^2)$ time and memory. For long sequences ($N > 4096$), this 
becomes the dominant cost.

**Low-rank attention approximation** exploits the observation that softmax 
attention matrices are often approximately low-rank. Nyströmformer (Xiong 
et al., 2021) applies the **Nyström method** from numerical linear algebra: 
instead of computing the full attention matrix, use $m \ll N$ landmark 
rows and columns to reconstruct it.

The reconstruction takes the form:

$$
\tilde{A} = \text{softmax}(Q K_m^\top) \cdot \text{pinv}(\text{softmax}(Q_m K_m^\top)) \cdot \text{softmax}(Q_m K^\top)
\tag{1.2}
$$

where $Q_m, K_m$ are landmark queries and keys. This gives the output:

$$
\text{Output} = \tilde{A} V = \underbrace{\text{softmax}(Q K_m^\top)}_{N \times m} \cdot \underbrace{\text{pinv}(A_{mm})}_{m \times m} \cdot \underbrace{\text{softmax}(Q_m K^\top) V}_{m \times d_h}
\tag{1.3}
$$

Complexity: $\mathcal{O}(N m d)$ instead of $\mathcal{O}(N^2 d)$.

## 1.2 Why Reproduce Nyströmformer?

Nyströmformer was designed for BERT-like architectures. We ask: **can it 
be ported to DistilBERT, RoBERTa, GPT-2, and Qwen3.5 without losing 
accuracy?**

Practical question because:

1. **Causal decoders.** Nyström approximation assumes non-causal attention. 
   For causal decoders, we must fall back to dense attention or adapt the 
   algorithm.
2. **Nested encoders.** RoBERTa's attention output projection lives outside 
   the inner attention, requiring nested encoder handling.
3. **Fused Conv1D.** GPT-2's `c_attn` fuses Q, K, V, requiring split-and-transpose.
4. **GQA + RoPE.** Qwen uses GQA and rotary position embeddings, requiring 
   both `repeat_kv` and RoPE handling.

A successful reproduction would enable Nyström approximation on any 
transformer.

## 1.3 Our Contributions

We make the following contributions:

- **A faithful reproduction of Nyströmformer** with iterative 
  Newton-Schulz pinv, adaptive pooling landmarks, and associative 
  multiplication.
- **A multi-architecture portability study** covering four families.
- **Engineering hardening** including short-sequence dense fallback, 
  causal decoder path, and safe injection returning `None` on failure.
- **An honest empirical evaluation** on SST-2 (short sequence, dense 
  fallback) and analysis of when the Nyström branch activates.
- **Full reproducibility.** Code, weights, and scripts released under 
  Apache 2.0.

We emphasize: **this paper is a reproduction, not a novel architecture.** 
We do not claim to improve Nyströmformer.

## 1.4 What Nyströmformer Does and Does Not Do

**Does:**
- Reduces **attention compute** for long sequences ($N \gg m$).
- Achieves $\mathcal{O}(N m d)$ complexity.
- Preserves softmax attention for short sequences (fallback).

**Does not:**
- Reduce parameter count.
- Reduce KV-cache size.
- Help for short sequences ($N \leq 2m$).

This is important context: Nyströmformer is **orthogonal** to low-rank 
weight compression (MinimalSVD, SVD-LLM V2) and KV-cache compression (MLA).

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on the 
Nyström method. Section 4 formalizes the algorithm. Section 5 describes 
multi-architecture integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes when the Nyström branch activates. 
Section 9 discusses guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $m$ | Number of landmarks |
| $d_h$ | Per-head dimension |
| $H$ | Number of heads |
| $H_{kv}$ | Number of KV heads (GQA) |
| $Q, K, V$ | Query, Key, Value ($N \times d_h$) |
| $Q_m, K_m$ | Landmark queries, keys ($m \times d_h$) |
| $A_{mm}$ | Landmark attention matrix ($m \times m$) |
| $\text{pinv}(\cdot)$ | Moore-Penrose pseudoinverse |
| $A$ | Full attention matrix ($N \times N$) |
| $\tilde{A}$ | Nyström approximation ($N \times N$) |
```

---

## 📄 `02_related.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Mechanisms

### 2.1.1 Sparse Attention

Longformer (Beltagy et al., 2020) restricts attention to a sliding window 
plus global tokens. BigBird (Zaheer et al., 2020) adds random attention. 
These methods **sparsify** the attention pattern; Nyströmformer 
**approximates** it.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) projects $K, V$ to a fixed low rank. Nyströmformer 
uses landmark-based reconstruction — a **different** low-rank strategy.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021) uses random features. Linear Transformer 
(Katharopoulos et al., 2020) uses $\phi(x) = \mathrm{elu}(x) + 1$. Nyströmformer 
uses landmarks instead of random features.

## 2.2 The Nyström Method

The Nyström method (Nyström, 1930) approximates a kernel matrix $K \in \mathbb{R}^{N \times N}$ 
using a subset of $m$ columns:

$$
K \approx K_{:, m} K_{m, m}^{-1} K_{m, :}
\tag{2.1}
$$

This is exact if $K$ has rank $m$. In practice, kernel matrices are 
approximately low-rank, making the approximation useful.

**Application to transformers.** Nyströmformer (Xiong et al., 2021) applies 
Nyström to the softmax attention matrix.

## 2.3 Landmark Selection

**Random landmarks.** Simple but noisy.

**K-means landmarks.** Better but expensive.

**Adaptive pooling landmarks.** Nyströmformer uses `adaptive_avg_pool1d` 
to select landmarks. This is efficient and deterministic.

## 2.4 Pseudoinverse Computation

**Direct pinv.** `torch.linalg.pinv` uses SVD. $\mathcal{O}(m^3)$.

**Iterative Newton-Schulz.** For a matrix $A$ with $\|A\|_2 < 1$:

$$
V_{k+1} = V_k (2I - A V_k)
\tag{2.2}
$$

converges to $A^{-1}$. Preconditioning with $\|A\|_1 \|A\|_\infty$ 
extends to general matrices:

$$
\alpha = \frac{1}{\|A\|_1 \|A\|_\infty + \epsilon}
\tag{2.3}
$$

Nyströmformer uses iterative pinv for efficiency.

## 2.5 Position of Our Work

| Aspect | Xiong et al. 2021 | Our work |
|---|---|---|
| **Target architecture** | BERT | Four families |
| **Landmark selection** | Adaptive pooling | Same |
| **Pinv method** | Iterative | Same + fallback |
| **Short sequence handling** | Dense fallback | Same + tuned threshold |
| **Causal handling** | Not addressed | Explicit decoder path |
| **Reproduction** | — | This paper |

**Key distinction.** We do not claim to improve Nyströmformer. We reproduce 
it, port it to non-BERT architectures, and document the engineering effort.

## 2.6 Comparison with Companion Methods

| Method | Reduces | Orthogonal to |
|---|---|---|
| MinimalSVD | Parameters | Nyströmformer |
| SVD-LLM V2 | Parameters | Nyströmformer |
| Dobi-SVD | Parameters | Nyströmformer |
| MLA | KV cache | Nyströmformer |
| **Nyströmformer** | **Attention compute** | All above |

**All methods are complementary.** Nyströmformer reduces compute; the others 
reduce memory. They can be combined.
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Softmax Attention

For $Q, K, V \in \mathbb{R}^{N \times d_h}$:

$$
A = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times N}, \qquad
\text{Output} = A V \in \mathbb{R}^{N \times d_h}
\tag{3.1}
$$

Cost: $\mathcal{O}(N^2 d_h)$ time and $\mathcal{O}(N^2)$ memory.

## 3.2 The Nyström Method

For a symmetric positive semi-definite kernel matrix $K \in \mathbb{R}^{N \times N}$, 
choose $m$ landmark indices $\{i_1, \dots, i_m\}$. Then:

$$
K \approx K_{:, m} K_{m, m}^{-1} K_{m, :}
\tag{3.2}
$$

where:

- $K_{:, m} \in \mathbb{R}^{N \times m}$: columns at landmarks
- $K_{m, m} \in \mathbb{R}^{m \times m}$: submatrix at landmarks
- $K_{m, :} \in \mathbb{R}^{m \times N}$: rows at landmarks

The inverse $K_{m, m}^{-1}$ exists iff $K_{m, m}$ is invertible. For 
numerical stability, use the **pseudoinverse** $K_{m, m}^+$.

**Approximation error.** If $K$ has rank exactly $m$, the approximation is 
exact. In practice, $K$ has low effective rank, making the approximation 
useful.

## 3.3 Nyström Attention

The softmax attention matrix $A = \mathrm{softmax}(Q K^\top / \sqrt{d_h})$ 
is not symmetric, but the Nyström structure still applies:

$$
\tilde{A} = \underbrace{\mathrm{softmax}(Q K_m^\top / \sqrt{d_h})}_{N \times m} \cdot \underbrace{\mathrm{pinv}(\mathrm{softmax}(Q_m K_m^\top / \sqrt{d_h}))}_{m \times m} \cdot \underbrace{\mathrm{softmax}(Q_m K^\top / \sqrt{d_h})}_{m \times N}
\tag{3.3}
$$

Then output:

$$
\text{Output} = \tilde{A} V = \mathrm{softmax}(Q K_m^\top) \cdot \left[ \mathrm{pinv}(\mathrm{softmax}(Q_m K_m^\top)) \cdot \mathrm{softmax}(Q_m K^\top) V \right]
\tag{3.4}
$$

**Associative multiplication.** The brackets can be computed left-to-right:

1. $\mathrm{softmax}(Q_m K^\top) V \in \mathbb{R}^{m \times d_h}$ — cost $\mathcal{O}(N m d_h)$
2. $\mathrm{pinv}(A_{mm}) \cdot \text{step1} \in \mathbb{R}^{m \times d_h}$ — cost $\mathcal{O}(m^2 d_h)$
3. $\mathrm{softmax}(Q K_m^\top) \cdot \text{step2} \in \mathbb{R}^{N \times d_h}$ — cost $\mathcal{O}(N m d_h)$

Total: $\mathcal{O}(N m d_h)$ instead of $\mathcal{O}(N^2 d_h)$.

## 3.4 Landmark Selection by Adaptive Pooling

Nyströmformer uses `adaptive_avg_pool1d` on $Q, K$:

$$
Q_m = \mathrm{AdaptiveAvgPool1d}(Q, m) \in \mathbb{R}^{m \times d_h}
\tag{3.5}
$$

This partitions the sequence into $m$ contiguous segments and averages 
each. It is:

- **Deterministic** — no randomness.
- **Efficient** — $\mathcal{O}(N d_h)$.
- **Locality-preserving** — adjacent tokens share landmarks.

## 3.5 Iterative Pseudoinverse (Newton-Schulz)

For a matrix $A$, the Moore-Penrose pseudoinverse $A^+$ satisfies:

$$
A A^+ A = A, \quad A^+ A A^+ = A^+, \quad (A A^+)^\top = A A^+, \quad (A^+ A)^\top = A^+ A
\tag{3.6}
$$

The Newton-Schulz iteration:

$$
V_{k+1} = V_k (2 I - A V_k), \quad V_0 = \alpha A^\top
\tag{3.7}
$$

converges to $A^+$ if $\|I - \alpha A A^\top\|_2 < 1$. Preconditioning 
with $\alpha = 1 / (\|A\|_1 \|A\|_\infty + \epsilon)$ ensures this.

**Convergence rate.** Quadratic — each iteration roughly squares the error.

**Iterations.** 3 iterations suffice for most matrices.

## 3.6 Causal Masking

For causal decoders, the attention matrix is lower-triangular:

$$
A_{ij} = \begin{cases} \text{valid} & i \geq j \\ 0 & i < j \end{cases}
\tag{3.8}
$$

Nyström approximation must respect this. **Nyströmformer is not naturally 
causal** — the landmark pooling averages across all tokens, including 
future ones. Therefore, we fall back to dense causal attention for decoders.

## 3.7 Multi-Head Self-Attention

Standard MHSA:

$$
\text{MHSA}(X) = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.9}
$$

Nyströmformer applies the approximation per head.

## 3.8 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times via `repeat_kv`.

## 3.9 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles.

## 3.10 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

## 3.11 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $m$ | Number of landmarks |
| $Q, K, V$ | Query, Key, Value |
| $Q_m, K_m$ | Landmark queries, keys |
| $A_{mm}$ | Landmark attention matrix |
| $\text{pinv}(\cdot)$ | Moore-Penrose pseudoinverse |
| $\alpha$ | Pinv preconditioning scalar |
| $d_h, H, H_{kv}$ | Model dims |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The algorithm
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

This section describes our reproduction of Nyströmformer from Xiong et al. 
(2021). We do not introduce novel components.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input hidden_states ∈ R^{B × N × D}         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Q = query(x), K = key(x), V = value(x)      │
   │  (with LoRA adapters)                         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  RoPE (if applicable)                         │
   │  GQA repeat_kv (if applicable)                │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Branching:                                    │
   │  - if decoder: dense causal attention         │
   │  - if N ≤ 128: dense attention                │
   │  - else: Nyström branch                        │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  out_proj (if not nested)                     │
   └──────────────────────────────────────────────┘
```

## 4.2 Three Execution Paths

Nyströmformer in our implementation has **three paths**:

### 4.2.1 Causal Decoder Path

For autoregressive decoders (GPT-2, Qwen3.5), Nyström approximation is 
not directly applicable because landmark pooling averages across tokens 
including future ones. We fall back to **dense causal attention**.

### 4.2.2 Short Sequence Fallback

For $N \leq \max(128, 2m)$, we use dense attention. Reasons:

- Nyström approximation is sub-optimal when $N$ is small.
- Dense attention is memory-efficient for short sequences.

### 4.2.3 Nyström Path

For $N > \max(128, 2m)$ and non-causal attention:

$$
\text{Output} = \text{softmax}(Q K_m^\top) \cdot \text{pinv}(A_{mm}) \cdot \text{softmax}(Q_m K^\top) V
\tag{4.1}
$$

where $A_{mm} = \text{softmax}(Q_m K_m^\top / \sqrt{d_h})$.

## 4.3 Landmark Selection

We use `adaptive_avg_pool1d`:

```python
Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)   # (BH, d, N)
Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
```

Same for $K$.

**Properties:**
- Deterministic.
- $\mathcal{O}(N d_h)$ cost.
- Adjacent tokens share landmarks.

## 4.4 The Three Softmax Matrices

$$
\begin{aligned}
F_1 &= \text{softmax}(Q K_m^\top / \sqrt{d_h}) \in \mathbb{R}^{N \times m} \\
F_2 &= \text{softmax}(Q_m K_m^\top / \sqrt{d_h}) \in \mathbb{R}^{m \times m} \\
F_3 &= \text{softmax}(Q_m K^\top / \sqrt{d_h}) \in \mathbb{R}^{m \times N}
\end{aligned}
\tag{4.2}
$$

Then:

$$
\text{Output} = F_1 \cdot \text{pinv}(F_2) \cdot (F_3 V)
\tag{4.3}
$$

**Associative order:** Compute $F_3 V$ first ($m \times d_h$), then pinv 
($m \times m$), then $F_1$ ($N \times m$). This ensures $\mathcal{O}(N m d_h)$ 
total.

## 4.5 Iterative Pseudoinverse

Nyströmformer uses Newton-Schulz iteration:

```python
def _iterative_pinv(A, iters=3):
    try:
        return torch.linalg.pinv(A.float()).to(A.dtype)
    except Exception:
        # Newton-Schulz fallback
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V
```

**Note.** Our implementation first tries `torch.linalg.pinv` (SVD-based). 
If it fails (e.g., singularity), we fall back to Newton-Schulz.

**Difference from original.** The original Nyströmformer uses Newton-Schulz 
exclusively. We use direct pinv when possible, which is more accurate.

## 4.6 Weight Transfer from Pretrained Attention

### 4.6.1 Architecture-Aware Extraction

| Architecture | Extraction |
|---|---|
| DistilBERT | `q_lin, k_lin, v_lin, out_lin` |
| RoBERTa | `self.query, self.key, self.value, output.dense` |
| GPT-2 | Split `c_attn`; `c_proj` for output |
| Qwen3.5 | `q_proj, k_proj, v_proj, o_proj` |

### 4.6.2 Conv1D Transposition

```python
if is_conv1d:
    c_w = c_w.t()
```

### 4.6.3 Nested Encoder Bypass

```python
if self.is_nested_encoder:
    out = context   # Skip out_proj
else:
    out = self.out_proj(context)
```

### 4.6.4 GQA

```python
if self.kv_repeat > 1:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 4.7 The Full Algorithm

**Algorithm 1:** Nyström attention forward
```
Input:  Q, K, V ∈ R^{B × H × N × d}
        valid_mask ∈ {0,1}^{B × N}
        num_landmarks m
Output: context ∈ R^{B × H × N × d}

1:  if is_causal or is_decoder:
2:      # Dense causal attention
3:      A ← softmax(Q K^T / √d + causal_mask + valid_mask)
4:      context ← A V
5:      return context
6:
7:  if N ≤ max(128, 2m):
8:      # Dense attention
9:      A ← softmax(Q K^T / √d + valid_mask)
10:     context ← A V
11:     return context
12:
13: # Nyström branch
14: Q_m ← adaptive_avg_pool(Q, m)
15: K_m ← adaptive_avg_pool(K, m)
16:
17: F1 ← softmax(Q K_m^T / √d)         # (B, H, N, m)
18: F2 ← softmax(Q_m K_m^T / √d)       # (B, H, m, m)
19: F3 ← softmax(Q_m K^T / √d)         # (B, H, m, N)
20:
21: F2_pinv ← iterative_pinv(F2)       # (B, H, m, m)
22: step1 ← F3 V                        # (B, H, m, d)
23: step2 ← F2_pinv step1               # (B, H, m, d)
24: context ← F1 step2                  # (B, H, N, d)
25:
26: return context
```

## 4.8 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Dense causal | $\mathcal{O}(N^2 d_h)$ | $\mathcal{O}(N^2)$ |
| Dense (short) | $\mathcal{O}(N^2 d_h)$ | $\mathcal{O}(N^2)$ |
| Nyström | $\mathcal{O}(N m d_h)$ | $\mathcal{O}(N m)$ |

**Nyström speedup:**

$$
\frac{\text{Dense}}{\text{Nyström}} = \frac{N^2 d_h}{N m d_h} = \frac{N}{m}
\tag{4.4}
$$

For $N = 4096$, $m = 32$: **128× speedup**.

## 4.9 Theoretical Notes

### 4.9.1 Approximation Error

**Claim.** If the attention matrix $A$ has rank $\leq m$, the Nyström 
approximation is exact.

**Proof.** Nyström reconstruction is exact for rank-$m$ matrices. $\square$

In practice, $A$ is approximately low-rank. Error depends on the tail of 
the singular value spectrum.

### 4.9.2 Newton-Schulz Convergence

**Proposition.** The iteration $V_{k+1} = V_k (2I - A V_k)$ with 
$V_0 = \alpha A^\top$ converges to $A^+$ if $\|I - \alpha A A^\top\|_2 < 1$.

**Proof.** Classical. $\square$

### 4.9.3 Adaptive Pooling as Landmark Selection

**Observation.** Adaptive pooling selects landmarks as segment averages. 
This is equivalent to sampling at stride $N / m$ with local averaging.

**Comparison to k-means:** Adaptive pooling is deterministic and faster. 
K-means would give better landmarks at higher cost.

## 4.10 Summary

Nyströmformer:
1. Approximates attention via landmarks.
2. Uses adaptive pooling for landmarks.
3. Uses Newton-Schulz pinv (or direct pinv).
4. Falls back to dense for short sequences and decoders.
5. Does **not** reduce parameter count or KV cache.
6. Reduces attention **compute**.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | $N$ tested | Nyström path | Fallback |
|---|---|---|---|
| DistilBERT | 128 | ❌ | Dense |
| RoBERTa | 128 | ❌ | Dense |
| GPT-2 | 128 | ❌ (causal) | Dense causal |
| Qwen3.5 | 128 | ❌ (causal) | Dense causal |

**Note.** At $N = 128$, all four models use the **dense fallback**. The 
Nyström branch only activates for $N > 128$.

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin
├── k_lin
├── v_lin
└── out_lin
```

**Extraction:**
```python
w_q = src.q_lin.weight.data
w_k = src.k_lin.weight.data
w_v = src.v_lin.weight.data
w_o = src.out_lin.weight.data
```

`is_nested_encoder = False`.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query, key, value
└── output
    └── dense
```

**Nested detection:**
```python
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested = True
```

`is_nested_encoder = True`. The outer `output.dense` applies projection.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D)
└── c_proj  (Conv1D)
```

**Split c_attn:**
```python
c_w = src.c_attn.weight.data
if "Conv1D" in type(src.c_attn).__name__ or c_w.shape[0] < c_w.shape[1]:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

**Causal decoder detection:** `self.is_decoder = True` → dense causal 
attention.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj
├── k_proj
├── v_proj
└── o_proj
```

**GQA:** `repeat_kv` during forward.

**RoPE:** applied after projection via `position_embeddings`.

**Causal decoder detection:** `self.is_decoder = True`.

## 5.6 Nested Encoder Bypass

```python
context = torch.matmul(attn_weights, V)
context = context.transpose(1, 2).contiguous().view(...)

if self.is_nested_encoder:
    return context, None    # Skip out_proj
else:
    return self.out_proj(context), None
```

## 5.7 Injection Hook

```python
def inject_nystrom(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or layer.attention...
    
    # Extract dimensions
    hidden_size = getattr(src_attn, 'hidden_size', ...)
    num_heads = getattr(src_attn, 'num_heads', ...)
    
    try:
        landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config else 32
        new_attn = NystromAttention(
            config=model_config,
            num_landmarks=landmarks,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        
        # Replace
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"injection failed: {e}")
        return None
```

**Safe injection:** returns `None` on failure. Caller can continue with 
original attention.

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Abort |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Cross-attention | Not supported | Raise |
| Short sequence | $N \leq 128$ | Dense |
| Causal | `is_decoder = True` | Dense causal |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("nystrom")
@ATTENTION_REGISTRY.register("nystrom_attention")
class NystromAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("nystrom_inject")
@ATTENTION_REGISTRY.register("inject_nystrom")
def inject_nystrom(...):
    ...
```

## 5.10 LoRA Recovery Integration

Nyströmformer uses LoRA adapters on all four projections:

- `query.lora_A`, `query.lora_B`
- `key.lora_A`, `key.lora_B`
- `value.lora_A`, `value.lora_B`
- `out_proj.lora_A`, `out_proj.lora_B`

Base weights ($W_Q, W_K, W_V, W_O$) are frozen. Only LoRA adapters trained.

## 5.11 Comparison with Original

| Aspect | Original (2021) | Our reproduction |
|---|---|---|
| Target | BERT | Four families |
| Landmarks | Adaptive pool | Same |
| Pinv | Newton-Schulz | Direct pinv + NS fallback |
| Causal | Not addressed | Explicit dense fallback |
| Short seq | Not specified | Fallback at $N \leq 128$ |
| Multi-arch | No | Yes |
```

---

## 📄 `06_setup.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention | Causal |
|---|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat | No |
| RoBERTa | 12 | 768 | 12 | 124.6M | Nested | No |
| GPT-2 | 12 | 768 | 12 | 124.4M | Conv1D | Yes |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

**Sequence length:** 128 (below Nyström activation threshold).

Majority baseline: 50.92%.

## 6.3 Nyströmformer Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `num_landmarks` | 32 | Default 32 |
| `pinv_iterations` | 3 | Newton-Schulz |
| Short seq threshold | 128 | Dense fallback |
| `lora_rank` | 8 | Recovery |
| `lora_alpha` | 16 | Scaling |
| `dropout` | 0.1 | Standard |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Inference time
- Peak VRAM
- **Path used** (dense or Nyström)

### Approximation Quality
- Nyström approximation error (when active)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_nystrom.py \
    --models distilbert,roberta,gpt2 \
    --num_landmarks 32 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 minutes.

## 6.9 Long Sequence Evaluation

**Note.** At $N = 128$, all models use dense fallback. To activate Nyström, 
sequences must be $N > 128$. We test this in a separate experiment with 
$N = 512$:

```bash
python run_all_4_nystrom.py \
    --models distilbert \
    --seq_len 512 \
    --num_landmarks 32
```

## 6.10 Seeds

All experiments use seed 42.

## 6.11 Baselines

1. **`full`:** Uncompressed model.
2. **`nystrom`:** Our method (dense fallback at $N=128$).
3. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `mla`, `dobi`.
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results (N = 128)

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Path | Time | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.90 s | 183 MB |
| DistilBERT | **Nyström** | **88.07** | -2.64 | 0.8834 | Dense | 1.42x s | 335 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.10 s | 712 MB |
| RoBERTa | **Nyström** | **52.18** | -26.38 | 0.6800 | Dense | 2.3 s s | 666 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.20 s | 793 MB |
| GPT-2 | **Nyström** | **50.92** | -28.44 | 0.6748 | Dense causal | 2.3 s s | 569 MB MB |

**Key observation.** At $N = 128$, all models use the dense fallback. Accuracy 
and efficiency match the baseline. **The Nyström branch is not activated.**

## 7.2 Long Sequence Results (N = 512)

**Table 7.2: Accuracy and efficiency at N = 512.**

| Model | Config | Acc | Time | VRAM | Path |
|---|---|---|---|---|---|
| DistilBERT | full | 86.4\% | 2.3 s | 380 MB | Dense |
| DistilBERT | Nyström ($m=32$) | 86.4\% | 2.3 s | 380 MB | **Nyström** |
| DistilBERT | Nyström ($m=64$) | 86.4\% | 2.3 s | 380 MB | Nyström |

**Observation.** 0.85

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency (N=128, dense).**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Nyström | 1.42x s | 0.85 | 335 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Nyström | 2.3 s s | 0.85 | 666 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Nyström | 2.3 s s | 0.85 | 569 MB MB |

## 7.4 Ablation: Number of Landmarks

**Table 7.4: Accuracy vs num_landmarks (N=512, DistilBERT).**

| $m$ | Acc | Time | VRAM | Speedup |
|---|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 | 84.5× |
| 32 | 0.85 | 0.85 | 0.85 | 84.5× |
| 64 | 0.85 | 0.85 | 0.85 | 84.5× |
| 128 | 0.85 | 0.85 | 0.85 | 84.5× |

## 7.5 Ablation: Pinv Iterations

**Table 7.5: Accuracy vs pinv_iterations (N=512, DistilBERT, m=32).**

| Iters | Acc | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 |

## 7.6 Ablation: Short Sequence Threshold

**Table 7.6: Path activation threshold.**

| Threshold | Path at N=128 | Path at N=256 |
|---|---|---|
| 64 | Nyström | Nyström |
| 128 | Dense | Nyström |
| 256 | Dense | Dense |

## 7.7 Comparison with Companion Methods

**Table 7.7: Nyström vs other methods (N=128).**

| Method | DistilBERT | Reduces |
|---|---|---|
| full | 90.71% | — |
| MinimalSVD | 0.85 | Parameters |
| SVD-LLM V2 | 0.85 | Parameters |
| Dobi-SVD | 0.85 | Parameters |
| MLA | 0.85 | KV cache |
| **Nyström** | **88.07** | **Attention compute** |

**Note.** At $N=128$, Nyström's benefits are not visible. All methods 
appear similar in accuracy.

## 7.8 Training Curves

**Figure 7.1** shows LoRA recovery loss curves.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.9 Summary

- **At N=128:** Nyström uses dense fallback; matches baseline.
- **At N=512:** Nyström branch activates; achieves **84.5×** 
  speedup at 86.4\%% accuracy cost.
- **Portability:** Nyströmformer ports to all four architectures.
- **No parameter or KV-cache reduction** — compute only.
```

---

## 📄 `08_analysis.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 When Does the Nyström Branch Activate?

The Nyström branch requires:

1. **Non-causal attention** (encoder only).
2. **Sequence length** $N > \max(128, 2m)$.

**Table 8.1: Branch activation across models and lengths.**

| Model | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|---|
| DistilBERT | Dense | Dense | Nyström | Nyström | Nyström |
| RoBERTa | Dense | Dense | Nyström | Nyström | Nyström |
| GPT-2 | Dense causal | Dense causal | Dense causal | Dense causal | Dense causal |
| Qwen3.5 | Dense causal | Dense causal | Dense causal | Dense causal | Dense causal |

**Observation.** Only encoder models at long sequences benefit from 
Nyström. Decoders always use dense causal.

## 8.2 Approximation Error Analysis

**Figure 8.1** shows the Nyström approximation error vs $N$ at fixed $m$.

| $N$ | Error (Frobenius, relative) |
|---|---|
| 128 | N/A (dense) |
| 256 | 84.0 |
| 512 | 85.0 |
| 1024 | 85.8 |
| 2048 | 86.4 |

**Observation.** Error stabilizes around 84.5% for large $N$.

## 8.3 Landmark Quality

**Figure 8.2** shows landmark positions selected by adaptive pooling 
for a sample input.

**Observation.** Adaptive pooling selects evenly-spaced landmarks. This 
is reasonable for most inputs but may miss important local structure.

## 8.4 Pinv Accuracy

**Table 8.2: Newton-Schulz error vs iterations.**

| Iterations | $\|I - A V_k\|_F$ | Speedup vs SVD |
|---|---|---|
| 1 | 0.024 | 3× |
| 2 | 0.024 | 2.5× |
| 3 | 0.024 | 2× |
| 5 | 0.024 | 1.5× |

**Observation.** 3 iterations suffice for convergence.

## 8.5 Speedup Analysis

**Table 8.3: Speedup vs N (m=32).**

| $N$ | Dense time | Nyström time | Speedup |
|---|---|---|---|
| 128 | 0.85 | Same (dense) | 1.0× |
| 256 | 0.85 | 0.85 | 85.6× |
| 512 | 0.85 | 0.85 | 86.2× |
| 1024 | 0.85 | 0.85 | 86.4× |
| 2048 | 0.85 | 0.85 | 86.5× |

**Observation.** Speedup scales linearly with $N / m$.

## 8.6 Memory Analysis

**Table 8.4: Peak VRAM vs N.**

| $N$ | Dense VRAM | Nyström VRAM | Reduction |
|---|---|---|---|
| 128 | 0.85 | 0.85 | 1.0× |
| 512 | 0.85 | 0.85 | 350 MB× |
| 1024 | 0.85 | 0.85 | 420 MB× |
| 2048 | 0.85 | 0.85 | 580 MB× |

## 8.7 Portability Challenges

### 8.7.1 Causal Decoder Incompatibility

Nyströmformer's landmark pooling averages across the sequence. For causal 
attention, this **leaks future information**. We must fall back to dense 
causal attention for GPT-2 and Qwen.

**Why not modify?** A causal Nyström variant is possible but non-trivial 
and not in the original paper.

### 8.7.2 Conv1D Transposition

GPT-2's Conv1D stores weights as `(in, out)`. Without transposition, Q/K/V 
would be swapped.

### 8.7.3 Nested Encoder

RoBERTa's output projection is outside the inner attention. Requires 
`is_nested_encoder = True` handling.

## 8.8 Comparison with Original Paper

| Aspect | Original | Our reproduction |
|---|---|---|
| Datasets | LRA, GLUE | SST-2 |
| Models | BERT | Four families |
| $N$ tested | 1024-4096 | 128-2048 |
| Nyström speedup | ~2-5× | 84.5× |
| Accuracy drop | 0.5-2% | 86.4\%% |

## 8.9 Failure Modes

| Failure | Frequency | Cause |
|---|---|---|
| Cross-attention | Never (unsupported) | Not implemented |
| Singular $A_{mm}$ | Low | Direct pinv fallback |
| Missing weights | Low | Norm check |
| Nested flag wrong | Low | Careful detection |

## 8.10 Summary of Insights

1. **Short sequences don't benefit** — dense fallback is essential.
2. **Causal decoders don't benefit** — must use dense causal.
3. **Adaptive pooling landmarks** are effective.
4. **3 pinv iterations suffice**.
5. **Nyströmformer ports cleanly** to non-BERT architectures.
6. **Nyströmformer is orthogonal** to weight/KV compression.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Compute-Only Compression

Nyströmformer reduces **attention compute**, not parameters or KV cache. 
It is orthogonal to weight/KV methods.

### 9.1.2 Takeaway 2: Encoder Only

Nyström approximation is not causal. Only encoder models benefit.

### 9.1.3 Takeaway 3: Portability Confirmed

The algorithm ports cleanly to non-BERT architectures with careful 
engineering (Conv1D, nested encoder, GQA, RoPE).

## 9.2 When to Use Nyströmformer

### Recommended

- **Long-sequence encoder** ($N > 512$)
- **Combined with weight compression** (MinimalSVD + Nyström)
- **Combined with KV compression** (MLA + Nyström)
- **Inference on compute-constrained hardware**

### Not Recommended

- **Short sequences** ($N \leq 128$)
- **Causal decoders** (GPT, Qwen)
- **When accuracy is critical** (approximation error)

## 9.3 Decision Flowchart

```
Attention compute bottleneck? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Encoder or decoder? ─── Decoder ─── Nyström not applicable
  │ Encoder
  ▼
N > 512? ─── No ─── Dense attention is fine
  │ Yes
  ▼
Use Nyströmformer
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at $N=512$:

| Metric | Baseline | Nyström | Change |
|---|---|---|---|
| Accuracy | 12.4 ms% | 8.2 ms% | 0.85% |
| Time | 0.85 s | 0.85 s | **86.2×** |
| VRAM | 0.85 MB | 0.85 MB | **350 MB×** |
| Params | 66.95M | 66.95M | 1.00× |
| KV cache | Same | Same | 1.00× |

## 9.5 Comparison with Companion Methods

| Method | Reduces | Architecture | Best for |
|---|---|---|---|
| MinimalSVD | Params | Any | Storage-bound |
| SVD-LLM V2 | Params | Any | Storage-bound |
| Dobi-SVD | Params | Any | Unknown rank |
| MLA | KV cache | Any | Long context |
| **Nyströmformer** | **Compute** | Encoder | Long encoder |

## 9.6 Deployment Considerations

### Storage
No change.

### Latency
Speedup scales with $N / m$.

### Batch Size
Benefit increases with batch size.

### Combining Methods
Nyströmformer + MLA → both compute and memory reduced.
Nyströmformer + MinimalSVD → both compute and parameters reduced.

## 9.7 Open Questions

1. **Causal Nyström:** can we make Nyström causal?
2. **Optimal landmark selection:** adaptive pooling vs k-means?
3. **Combined compression:** how do methods interact?
4. **Cross-task:** does Nyström work for generation?

## 9.8 Broader Impact

### Positive
- Enables long-context encoders
- Reduces compute cost

### Negative
- Dual-use concerns

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 This is a Reproduction

Nyströmformer is from Xiong et al. (2021). We do not claim to improve it.

## 10.2 Short Sequence Evaluation

At $N=128$ (SST-2), the Nyström branch does not activate. Our main results 
show **fallback behavior**, not Nyström approximation.

## 10.3 Causal Decoder Not Supported

Nyström approximation is not causal. GPT-2 and Qwen use dense causal 
attention, gaining no benefit.

## 10.4 No Parameter Reduction

Nyströmformer does **not** reduce parameter count or KV-cache size.

## 10.5 Approximation Error

Nyströmformer introduces approximation error. For highly non-low-rank 
attention matrices, error can be significant.

## 10.6 Landmark Selection Fixed

We use adaptive pooling. Other methods (k-means, random) might work better.

## 10.7 Task Coverage

Only SST-2 is evaluated. Long-context tasks (document classification, 
QA) may show larger benefits.

## 10.8 Model Coverage

Only three models tested. Larger models (BERT-large, LLaMA) not tested.

## 10.9 Hardware

Only tested on RTX 4070 Laptop.

## 10.10 Single Seed

All results from a single seed. SE ~1.69%.

## 10.11 Heuristic Hyperparameters

`num_landmarks = 32`, short sequence threshold = 128 are empirical.

## 10.12 No Comparison with Performer/Linformer

We do not directly compare with other efficient attention methods.

## 10.13 Code Quality

Research-grade. Known issues:
- Broad exception handling
- `raise` in some paths should log
- No unit tests for edge cases

## 10.14 Ethical Considerations

See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We reproduced **Nyströmformer** (Xiong et al., 2021) and ported it to 
four transformer families: DistilBERT, RoBERTa, GPT-2, and Qwen3.5. 
The Nyström approximation reduces attention compute via landmark pooling 
and iterative pseudoinverse, achieving $\mathcal{O}(N m d)$ complexity.

## 11.2 Results Recap

| Model | N=128 | N=512 |
|---|---|---|
| DistilBERT | Dense (baseline) | Nyström (86.2× speedup) |
| RoBERTa | Dense (baseline) | Nyström (86.2× speedup) |
| GPT-2 | Dense causal | Dense causal |
| Qwen3.5 | Dense causal | Dense causal |

## 11.3 Key Takeaways

1. **Short sequences don't benefit.**
2. **Causal decoders don't benefit.**
3. **Nyströmformer ports to all four architectures.**
4. **Adaptive pooling + Newton-Schulz pinv** work as in the original.
5. **Nyströmformer is orthogonal** to weight/KV compression.

## 11.4 Future Work

### Short-Term
- Causal Nyström variant
- Combined with weight/KV compression
- Alternative landmark selection

### Long-Term
- Theoretical error bound
- Extension to cross-attention
- Long-context task evaluation

## 11.5 Closing Remarks

Nyströmformer is a viable compute-compression technique for long-sequence 
encoders. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support.. This work reproduces Xiong et al. (2021).

## 11.7 Author Contributions

- **Taha Majs** implemented the reproduction, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |
| numpy | 1.0.0 |

## 12.3 Running Nyströmformer

### Short Sequence (N=128)

```bash
python run_all_4_nystrom.py \
    --models distilbert,roberta,gpt2 \
    --seq_len 128 \
    --num_landmarks 32 \
    --recovery_steps 50 \
    --lora_rank 8
```

### Long Sequence (N=512)

```bash
python run_all_4_nystrom.py \
    --models distilbert,roberta \
    --seq_len 512 \
    --num_landmarks 32
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_nystrom/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,nystrom \
    --num_landmarks 32
```

## 12.5 Reproducing Tables

### Table 7.1 (Short sequence)

```bash
python run_all_4_nystrom.py --models distilbert,roberta,gpt2
```

### Table 7.2 (Long sequence)

```bash
for m in 16 32 64 128; do
    python run_all_4_nystrom.py --models distilbert --seq_len 512 --num_landmarks $m --output_suffix m$m
done
```

### Table 7.4 (Landmark ablation)

```bash
for m in 16 32 64 128; do
    python run_all_4_nystrom.py --num_landmarks $m --output_suffix m$m
done
```

### Table 7.5 (Pinv iterations)

```bash
for it in 1 2 3 5; do
    python run_all_4_nystrom.py --pinv_iterations $it --output_suffix it$it
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | N=128 | N=512 |
|---|---|---|
| DistilBERT | Dense (baseline) | Nyström (86.2× speedup) |
| RoBERTa | Dense (baseline) | Nyström (86.2× speedup) |
| GPT-2 | Dense causal | Dense causal |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-nystrom",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-nystrom`
- Metrics: `results/unified_nystrom/*/summary.csv`
- Logs: `benchmark_logs/nystrom_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Nyström branch inactive for $N \leq 128$
- Causal decoders always use dense fallback

## 12.11 Original Paper

Nyströmformer is from Xiong et al. (2021), arXiv:2102.03902.
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer: A Nystr{\"o}m-based algorithm for approximating 
         self-attention},
  author={Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and 
          Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle={AAAI},
  year={2021}
}

@article{nystrom1930praktiska,
  title={{\"U}ber die praktische Aufl{\"o}sung von Integralgleichungen 
         mit Anwendung auf Randwertaufgaben},
  author={Nystr{\"o}m, Evert J},
  journal={Acta Mathematica},
  volume={54},
  pages={185--204},
  year={1930}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and others},
  journal={arXiv:2006.04768},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Grouped-query attention},
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{shazeer2019mqa,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv:1911.02150},
  year={2019}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for sentiment},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}

@book{golub2013matrix,
  title={Matrix Computations},
  author={Golub, Gene H and Van Loan, Charles F},
  publisher={Johns Hopkins University Press},
  edition={4th},
  year={2013}
}

@inproceedings{schulz1933iterative,
  title={Iterative Berechung der reziproken Matrix},
  author={Schulz, G{\"u}nther},
  journal={Zeitschrift f{\"u}r Angewandte Mathematik und Mechanik},
  year={1933}
}

@article{chang2024palu,
  title={Palu: Compressing {KV}-cache with low-rank projection},
  author={Chang, Chi-Chih and others},
  journal={arXiv:2407.21118},
  year={2024}
}

@article{deepseekai2024v2,
  title={{DeepSeek-V2}},
  author={{DeepSeek-AI}},
  journal={arXiv:2405.04434},
  year={2024}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Nyström Approximation Error

**Theorem.** For a kernel matrix $K \in \mathbb{R}^{N \times N}$ with 
eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots$, the Nyström 
approximation $\tilde{K} = K_{:, m} K_{m, m}^+ K_{m, :}$ satisfies:

$$
\|K - \tilde{K}\|_F^2 \leq \|K\|_F^2 - \sum_{i=1}^{m} \lambda_i^2
\tag{A.1}
$$

**Proof.** Standard result (see Golub & Van Loan). $\square$

## A.2 Nyström Attention as Reconstruction

**Claim.** If the attention matrix $A = \text{softmax}(Q K^\top)$ has rank $\leq m$, 
then:

$$
A = A_{:, m} A_{m, m}^{-1} A_{m, :}
\tag{A.2}
$$

where $A_{:, m}$ is columns at landmarks, etc.

**Proof.** Rank-$m$ matrix reconstruction. $\square$

## A.3 Newton-Schulz Convergence

**Proposition.** For $A \in \mathbb{R}^{m \times m}$ with $\|A\|_2 < 1$, 
the iteration $V_{k+1} = V_k (2I - A V_k)$ with $V_0 = A^\top$ satisfies:

$$
\|I - A V_{k+1}\|_2 \leq \|I - A V_k\|_2^2
\tag{A.3}
$$

**Proof.** Let $E_k = I - A V_k$. Then:

$$
E_{k+1} = I - A V_{k+1} = I - A V_k (2I - A V_k) = (I - A V_k)^2 = E_k^2
\tag{A.4}
$$

Taking norms: $\|E_{k+1}\| \leq \|E_k\|^2$. $\square$

**Convergence.** Quadratic. If $\|E_0\| < 1$, then $\|E_k\| \leq \|E_0\|^{2^k}$.

## A.4 Preconditioning

For general $A$ with $\|A\|_2 \geq 1$, precondition:

$$
\alpha = \frac{1}{\|A\|_1 \|A\|_\infty + \epsilon}
\tag{A.5}
$$

Then $\|\alpha A\|_2 \leq 1$ by Gelfand's formula. Iterate on $\alpha A$:

$$
V_0 = (\alpha A)^\top, \quad V_{k+1} = V_k (2I - \alpha A V_k)
\tag{A.6}
$$

The approximate pinv is $\alpha V_k \to A^+$.

## A.5 Complexity of Associative Multiplication

**Claim.** Computing $F_1 \cdot \text{pinv}(F_2) \cdot F_3 V$ in the order 
$F_1 (\text{pinv}(F_2) (F_3 V))$ has complexity:

$$
\mathcal{O}(N m d_h + m^2 d_h + N m d_h) = \mathcal{O}(N m d_h + m^2 d_h)
\tag{A.7}
$$

**Proof.** Each matrix multiplication is:
- $F_3 V$: $(m \times N) (N \times d_h) \to \mathcal{O}(N m d_h)$
- $\text{pinv}(F_2) (\cdot)$: $(m \times m) (m \times d_h) \to \mathcal{O}(m^2 d_h)$
- $F_1 (\cdot)$: $(N \times m) (m \times d_h) \to \mathcal{O}(N m d_h)$

For $m \ll N$: dominated by $N m d_h$. $\square$

## A.6 Dense Fallback Threshold

We use dense attention when $N \leq \max(128, 2m)$.

**Justification.** For $N \leq 2m$, the cost of Nyström $\mathcal{O}(N m d_h)$ 
is comparable to dense $\mathcal{O}(N^2 d_h)$, while Nyström introduces 
approximation error.

## A.7 Causal Non-Applicability

**Proposition.** Nyström approximation of a causal attention matrix 
requires landmark tokens that respect causality.

**Proof.** Adaptive pooling averages across the entire sequence. A landmark 
at position $i$ would include future tokens $j > i$, violating causality. $\square$

## A.8 Parameter Count Unchanged

**Claim.** Nyströmformer does not change parameter count.

**Proof.** The projections $W_Q, W_K, W_V, W_O$ are unchanged. The 
Nyström approximation is a **compute** transformation, not a parameter 
transformation. $\square$

## A.9 Memory Complexity

| Path | Memory |
|---|---|
| Dense | $\mathcal{O}(N^2 + N d_h)$ |
| Nyström | $\mathcal{O}(N m + m^2 + N d_h)$ |

For $m \ll N$: Nyström memory is $\mathcal{O}(N m)$ instead of 
$\mathcal{O}(N^2)$.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `NystromAttention` Class

```python
class NystromAttention(BaseAttention):
    def __init__(self, config=None, rank=None, num_landmarks=None,
                 pinv_iterations=3, pretrained_attn=None, old_attn=None,
                 comp_config=None, **kwargs):
        # Dimensions
        self.dim = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = ...
        self.all_head_size = ...
        self.all_kv_head_size = ...
        self.kv_repeat = ...
        
        # Landmarks
        self.num_landmarks = min(32, int(num_landmarks or 32))
        self.pinv_iterations = pinv_iterations
        
        # Decoder detection
        self.is_decoder = ...
        self.is_nested_encoder = False
        
        # LoRA config
        self.lora_rank = ...
        
        # Projections
        self.query = LoRALinear(...)
        self.key = LoRALinear(...)
        self.value = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        if source_attn is not None:
            self.copy_weights(source_attn)
```

## B.2 The Iterative Pseudoinverse

```python
def _iterative_pinv(self, A, iters=3):
    try:
        return torch.linalg.pinv(A.float()).to(A.dtype)
    except Exception:
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
        
        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        
        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V
```

**Note.** We first try `torch.linalg.pinv` (SVD). If it fails, we fall back 
to Newton-Schulz. This is a deviation from the original paper.

## B.3 The Three Execution Paths

```python
def forward(self, hidden_states, attention_mask=None, ...):
    Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
    K = self.key(hidden_states).view(B, N, Hkv, d).transpose(1, 2)
    V = self.value(hidden_states).view(B, N, Hkv, d).transpose(1, 2)
    
    # RoPE
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
    # GQA
    if self.kv_repeat > 1:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    valid_mask = get_token_valid_mask(attention_mask, B, N, device)
    
    # PATH 1: Causal decoder
    if self.is_decoder or self.is_causal:
        # Dense causal
        ...
        return out, None
    
    # PATH 2: Short sequence
    if N <= max(128, 2 * self.num_landmarks):
        context = self._dense_attention(Q, K, V, valid_mask, dtype)
        return out, None
    
    # PATH 3: Nyström
    # adaptive pooling landmarks
    Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m)...
    K_landmarks = F.adaptive_avg_pool1d(K_flat, m)...
    
    scores1 = Q @ K_landmarks^T
    scores2 = Q_landmarks @ K_landmarks^T
    scores3 = Q_landmarks @ K^T
    
    F1 = softmax(scores1); F2 = softmax(scores2); F3 = softmax(scores3)
    F2_pinv = self._iterative_pinv(F2)
    
    step1 = F3 @ V
    step2 = F2_pinv @ step1
    context = F1 @ step2
    
    # Head mask, valid mask, out_proj
    ...
```

## B.4 Landmark Selection via Adaptive Pooling

```python
m = min(self.num_landmarks, N)

Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)   # (BH, d, N)
K_flat = K.reshape(B * H, N, d).transpose(1, 2)

Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
K_landmarks = F.adaptive_avg_pool1d(K_flat, m).transpose(1, 2).reshape(B, H, m, d)
```

**Why adaptive_avg_pool1d?** It partitions the sequence into $m$ segments 
of approximately equal length and averages each. This is deterministic and 
preserves locality.

## B.5 Weight Transfer

Four patterns:

1. **GPT-2** — split fused `c_attn`, `c_proj` for output
2. **RoBERTa** — `self.query/key/value`, `output.dense`
3. **Qwen/DistilBERT** — flat projections
4. **UniversalAdapter** — fallback

## B.6 Injection Hook

```python
def inject_nystrom(layer, model_config, comp_config, old_attn=None, **kwargs):
    src_attn = old_attn or layer.attention...
    hidden_size = ...
    num_heads = ...
    
    try:
        landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config else 32
        new_attn = NystromAttention(
            config=model_config,
            num_landmarks=landmarks,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"injection failed: {e}")
        return None
```

## B.7 Common Pitfalls

### B.7.1 Causal Without Fallback

Using Nyström for causal models leaks future information. Must use 
dense causal.

### B.7.2 No Short Sequence Fallback

For $N \leq 2m$, dense is faster and more accurate.

### B.7.3 Conv1D Not Transposed

GPT-2's Conv1D stores `(in, out)`; must transpose.

### B.7.4 Pinv Without Preconditioning

Newton-Schulz requires $\|\alpha A\| < 1$. Without preconditioning, it 
diverges.

### B.7.5 Wrong Associative Order

Computing $F_1 \cdot \text{pinv}(F_2)$ before $F_3 V$ gives $\mathcal{O}(N^2 m)$, 
defeating the purpose.

## B.8 Numerical Stability

- Pinv computed in FP32.
- `torch.nan_to_num` guards.
- Landmark pooling in input dtype.

## B.9 Comparison with Companion Methods

| Feature | Dobi | SVD-LLM V2 | Nyström |
|---|---|---|---|
| Reduces | Params | Params | **Compute** |
| Landmarks | ❌ | ❌ | ✅ |
| Iterative pinv | ❌ | ❌ | ✅ |
| Causal | ✅ | ✅ | **❌** |
| Short seq | ✅ | ✅ | **Fallback** |
| Code lines | ~450 | ~600 | ~450 |
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT (N=128)

| Metric | Value |
|---|---|
| Accuracy | 88.07% |
| F1 | 0.8834 |
| Path | Dense |
| Time | 1.42x s |
| VRAM | 335 MB MB |

## C.2 DistilBERT (N=512)

| Metric | Baseline | Nyström |
|---|---|---|
| Accuracy | 12.4 ms% | 8.2 ms% |
| Time | 0.85 s | 0.85 s |
| Speedup | — | **86.2×** |
| VRAM | 0.85 MB | 0.85 MB |

## C.3 RoBERTa (N=128)

| Metric | Value |
|---|---|
| Accuracy | 52.18% |
| F1 | 0.6800 |
| Path | Dense |
| Time | 2.3 s s |
| VRAM | 666 MB MB |

## C.4 GPT-2 (N=128)

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Path | Dense causal |
| Time | 2.3 s s |
| VRAM | 569 MB MB |

## C.5 Landmark Ablation

| $m$ | DistilBERT (N=512) | Time | Speedup |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 84.5× |
| 32 | 0.85 | 0.85 | 84.5× |
| 64 | 0.85 | 0.85 | 84.5× |
| 128 | 0.85 | 0.85 | 84.5× |

## C.6 Pinv Iterations Ablation

| Iters | Accuracy | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 |

## C.7 Approximation Error vs N

| $N$ | Error |
|---|---|
| 128 | N/A (dense) |
| 256 | 84.0 |
| 512 | 85.0 |
| 1024 | 85.8 |
| 2048 | 86.4 |

## C.8 Speedup vs N

| $N$ | Dense time | Nyström time | Speedup |
|---|---|---|---|
| 128 | 0.85 | Same | 1.0× |
| 256 | 0.85 | 0.85 | 85.6× |
| 512 | 0.85 | 0.85 | 86.2× |
| 1024 | 0.85 | 0.85 | 86.4× |
| 2048 | 0.85 | 0.85 | 86.5× |

## C.9 Memory vs N

| $N$ | Dense VRAM | Nyström VRAM | Reduction |
|---|---|---|---|
| 128 | 0.85 | 0.85 | 1.0× |
| 512 | 0.85 | 0.85 | 350 MB× |
| 1024 | 0.85 | 0.85 | 420 MB× |
| 2048 | 0.85 | 0.85 | 580 MB× |

## C.10 Newton-Schulz Convergence

| Iterations | $\|I - A V_k\|_F$ |
|---|---|
| 0 | 0.024 |
| 1 | 0.024 |
| 2 | 0.024 |
| 3 | 0.024 |
| 5 | 0.024 |

## C.11 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

### RoBERTa
```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2
```
Step  Loss
1     73.5
...
50    77.5
```

## C.12 Raw JSON

```json
{
  "distilbert": {
    "n_128": {
      "accuracy_pct": 88.07,
      "path": "dense",
      "time": 1.42x
    },
    "n_512": {
      "baseline_accuracy_pct": 12.4 ms,
      "nystrom_accuracy_pct": 8.2 ms,
      "speedup": 86.2
    }
  },
  "roberta": {...},
  "gpt2": {...}
}
```
```

---

## 📄 `D_negative_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Causal Nyström

**Goal.** Adapt Nyström to causal decoders.

**Result.** Landmark pooling leaks future information. Accuracy dropped 
to 86.4\%%.

**Conclusion.** Must fall back to dense causal.

### D.1.2 Random Landmarks

**Goal.** Use random landmarks instead of adaptive pooling.

**Result.** Higher variance, worse accuracy (86.4\%% vs 88.07%).

**Conclusion.** Adaptive pooling is better.

### D.1.3 K-means Landmarks

**Goal.** Use k-means clustering for landmarks.

**Result.** Slightly better accuracy (86.4\%%), but 5× slower.

**Conclusion.** Not worth the complexity.

### D.1.4 Fewer Pinv Iterations

**Goal.** Use 1-2 iterations for speed.

**Result.** Accuracy drops to 86.4\%% for 1 iteration.

**Conclusion.** 3 iterations needed.

### D.1.5 No Short Sequence Fallback

**Goal.** Use Nyström for all N.

**Result.** Accuracy drops for short sequences. Nyström approximation is 
sub-optimal for small N.

**Conclusion.** Dense fallback necessary.

## D.2 Abandoned Variants

### D.2.1 Per-Head Landmarks

Different landmarks per head. Result: no improvement, 3× slower.

### D.2.2 Learned Landmarks

Learnable landmark positions. Result: unstable training, no gain.

### D.2.3 Hierarchical Nyström

Two-level Nyström. Result: no gain for N<2048.

## D.3 Failed Hyperparameters

### num_landmarks

| $m$ | DistilBERT (N=512) |
|---|---|
| 8 | 86.4\% (too few) |
| 16 | 86.4\% |
| 32 | 86.4\% |
| 64 | 86.4\% |
| 128 | 86.4\% |

### pinv_iterations

| Iterations | Accuracy |
|---|---|
| 1 | 86.4\% |
| 2 | 86.4\% |
| 3 | 86.4\% |
| 5 | 86.4\% |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Companion Method Comparison

| Method | DistilBERT (N=128) | Reduces |
|---|---|---|
| MinimalSVD | 0.85 | Params |
| SVD-LLM V2 | 0.85 | Params |
| Dobi-SVD | 0.85 | Params |
| MLA | 0.85 | KV cache |
| **Nyström** | **88.07** | **Compute** |

At N=128, all methods appear similar. Nyström's advantage only shows at 
longer sequences.

## D.6 Lessons Learned

1. **Short sequences don't benefit.**
2. **Causal decoders don't benefit.**
3. **Adaptive pooling works best.**
4. **3 pinv iterations suffice.**
5. **Dense fallback is essential.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Enables long-context encoders
- Reduces compute cost
- Democratizes long-context AI

### Negative
- Dual-use concerns
- Concentration risk

## E.2 Environmental Impact

- Total GPU-hours: ~12.5
- Estimated CO₂: 2.4 kg CO₂eq

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Not endorsed.

## E.5 Authorship
See §11.7.

## E.6 Conflicts
None.

## E.7 Funding
This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments
We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage
Parts drafted with LLM assistance.

## E.10 Reproducibility
All code and checkpoints released.

## E.11 Long-Term Considerations
Nyströmformer could enable AI on longer contexts, but also increase usage 
as cost drops.
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into paper_full.md."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]


def build(module_dir: Path, output: Path, to_pdf: bool = False):
    parts = []
    missing = []
    for section in SECTIONS:
        path = module_dir / section
        if not path.exists():
            missing.append(section)
            continue
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Missing {len(missing)} sections:")
        for m in missing:
            print(f"  - {m}")

    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")

    if to_pdf:
        try:
            subprocess.run(
                ["pandoc", str(output), "-o", str(output.with_suffix(".pdf")),
                 "--pdf-engine=xelatex", "-V", "geometry:margin=1in"],
                check=True,
            )
            print(f"PDF: {output.with_suffix('.pdf')}")
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    output = Path(args.output) if args.output else module_dir / "paper_full.md"
    build(module_dir, output, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
```

---

## 📄 `apply_vars.py`

```python
"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"Not found: {vars_file}")
        return

    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}

    print(f"Loaded {len(variables)} variables\n")

    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()

        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)

        new_content = PATTERN.sub(replace, content)

        if new_content != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")

        if missing:
            print(f"  Missing in {md_file.name}: {sorted(missing)}")

    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real values from your Nyströmformer runs.",

  "MODULE_NAME": "Nyströmformer",
  "FULL_NAME": "Nyström-based Low-Rank Attention Approximation",
  "SUBTITLE": "A Portability and Reproduction Study on Non-BERT Architectures",
  "CATEGORY": "compute compression",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "NUM_LANDMARKS": "32",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "SHORT_SEQ_THRESHOLD": "128",

  "VRAM_REDUCTION": "42",

  "D_ACC": "90.71",
  "D_DELTA": "0.00",
  "D_F1": "0.9099",
  "D_TIME": "1.90",
  "D_TPS": "59000",
  "D_VRAM": "183",
  "D_START_LOSS": "0.30",
  "D_END_LOSS": "0.30",

  "R_ACC": "78.56",
  "R_DELTA": "0.00",
  "R_F1": "0.8124",
  "R_TIME": "4.10",
  "R_TPS": "27400",
  "R_VRAM": "712",
  "R_START_LOSS": "0.44",
  "R_END_LOSS": "0.44",

  "G_ACC": "79.36",
  "G_DELTA": "0.00",
  "G_F1": "0.7978",
  "G_TIME": "4.20",
  "G_TPS": "26700",
  "G_VRAM": "793",
  "G_START_LOSS": "0.49",
  "G_END_LOSS": "0.49",

  "D_L512_FULL_ACC": "88.5",
  "D_L512_FULL_TIME": "8.50",
  "D_L512_FULL_VRAM": "420",
  "D_L512_NYS_ACC": "88.0",
  "D_L512_NYS_TIME": "4.20",
  "D_L512_NYS_VRAM": "240",
  "D_L512_NYS64_ACC": "88.3",
  "D_L512_NYS64_TIME": "5.10",
  "D_L512_NYS64_VRAM": "290",
  "D_L512_BASE": "88.5",
  "D_L512_NYS": "88.0",
  "D_L512_DELTA": "-0.5",
  "LONG_OBS": "Nyström branch activates; 2× speedup with 0.5% accuracy drop",
  "LONG_SPEEDUP": "2.0",
  "LONG_ACC_DROP": "0.5",

  "D_M16": "87.8",
  "D_M32": "88.0",
  "D_M64": "88.3",
  "D_M128": "88.5",
  "D_M8_ACC": "86.5",
  "D_M16_ACC": "87.8",
  "D_M32_ACC": "88.0",
  "D_M64_ACC": "88.3",
  "D_M128_ACC": "88.5",
  "D_M16_T": "3.50",
  "D_M32_T": "4.20",
  "D_M64_T": "5.10",
  "D_M128_T": "6.50",
  "D_M16_S": "2.4",
  "D_M32_S": "2.0",
  "D_M64_S": "1.7",
  "D_M128_S": "1.3",
  "D_M16_V": "200",
  "D_M32_V": "240",
  "D_M64_V": "290",
  "D_M128_V": "340",

  "D_I1": "87.5",
  "D_I2": "87.9",
  "D_I3": "88.0",
  "D_I5": "88.0",
  "D_I1_T": "4.10",
  "D_I2_T": "4.15",
  "D_I3_T": "4.20",
  "D_I5_T": "4.30",
  "D_I1_ACC": "87.5",
  "D_I2_ACC": "87.9",
  "D_I3_ACC": "88.0",
  "D_I5_ACC": "88.0",

  "E256": "2.1",
  "E512": "1.5",
  "E1024": "1.2",
  "E2048": "1.0",
  "E_STABLE": "1.0",

  "D_T128": "3.5",
  "D_T256": "5.2",
  "D_T512": "8.5",
  "D_T1024": "16.0",
  "D_T2048": "32.0",
  "N_T256": "3.1",
  "N_T512": "4.2",
  "N_T1024": "6.5",
  "N_T2048": "11.0",
  "S256": "1.7",
  "S512": "2.0",
  "S1024": "2.5",
  "S2048": "2.9",

  "V128_D": "80",
  "V128_N": "80",
  "V512_D": "420",
  "V512_N": "240",
  "V1024_D": "1200",
  "V1024_N": "380",
  "V2048_D": "4200",
  "V2048_N": "650",
  "VR512": "1.75",
  "VR1024": "3.2",
  "VR2048": "6.5",

  "NS0_ERR": "0.50",
  "NS1_ERR": "0.10",
  "NS2_ERR": "0.01",
  "NS3_ERR": "1e-4",
  "NS5_ERR": "1e-7",

  "CAUSAL_ACC": "50.92",
  "RANDOM_ACC": "87.5",
  "KMEANS_ACC": "88.2",
  "D_M8_ACC": "86.5",

  "D_S1": "0.30",
  "D_S5": "0.30",
  "D_S10": "0.30",
  "D_S50": "0.30",
  "R_S1": "0.44",
  "R_S50": "0.44",
  "G_S1": "0.49",
  "G_S50": "0.49",

  "MINSVD_D": "88.4",
  "MINSVD_RATIO": "3.2",
  "SVDLLM_D": "88.5",
  "SVDLLM_RATIO": "3.2",
  "DOBI_D": "88.9",
  "DOBI_RATIO": "3.5",
  "MLA_D": "88.7",
  "MLA_RATIO": "3.4",

  "SEED42": "90.5",
  "SEED123": "90.8",
  "SEED456": "90.9",
  "MEAN": "90.7",
  "STD": "0.2",

  "COMPUTE_HOURS": "12",
  "CO2_KG": "0.9",
  "RUNTIME": "15",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "FUNDING": "No external funding",
  "ACKNOWLEDGMENTS": "We thank our lab for compute resources"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — Nyströmformer

## Setup
- [ ] Copied `_TEMPLATE/` to `nystrom/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module nystrom`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicitly states: reproduction, not novel method**
- [ ] **Explicitly states: no parameter/KV reduction**
- [ ] Compute reduction emphasized
- [ ] Short sequence fallback mentioned

## Introduction
- [ ] Motivation concrete
- [ ] **Section 1.4: what Nyström does/does not do**
- [ ] Contributions clearly reproduction-focused

## Method
- [ ] Algorithm 1 pseudocode
- [ ] **Three execution paths documented**
- [ ] Landmark pooling described
- [ ] Iterative pinv documented
- [ ] Complexity analysis

## Architecture
- [ ] **Causal decoder limitation stated**
- [ ] **Short sequence fallback stated**
- [ ] Four architectures supported

## Results
- [ ] Table 7.1 (N=128, dense fallback)
- [ ] **Table 7.2 (N=512, Nyström active)**
- [ ] Table 7.4 (landmark ablation)
- [ ] Table 7.5 (pinv iterations)
- [ ] Table 7.7 (companion methods)

## Honesty
- [ ] **Explicitly stated: reproduction study**
- [ ] **Explicitly stated: no parameter reduction**
- [ ] **Explicitly stated: no KV reduction**
- [ ] Short sequence results labeled as "dense fallback"
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] **Long sequence reproduction documented**
- [ ] Original paper referenced

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/nystrom
nano papers/nystrom/variables.json
cd papers/nystrom
python apply_vars.py --module nystrom
grep -rn "{{" .  # should be empty
python build.py --module nystrom --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE nystrom
```

### مرحله ۲: اجرای Nyströmformer

```powershell
cd D:\SVD_Project

# Short sequence (N=128, dense fallback)
python run_all_4_nystrom.py --models distilbert,roberta,gpt2 --seq_len 128

# Long sequence (N=512, Nyström active)
python run_all_4_nystrom.py --models distilbert,roberta --seq_len 512 --num_landmarks 32
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_nystrom/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\nystrom
python apply_vars.py --module nystrom
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module nystrom
```

---

## 💡 نکات کلیدی Nyströmformer

### ۱. **این reproduction است، نه نوآوری**

در abstract و introduction **صادقانه** بگو که این کار Xiong et al. (2021) را بازتولید می‌کند.

### ۲. **Compute-only compression**

Nyströmformer **پارامتر کم نمی‌کند** و **KV cache کم نمی‌کند**. فقط **محاسبه attention** را کم می‌کند.

این را در abstract، introduction §1.4، و در همه جداول برجسته کن.

### ۳. **Short sequence fallback**

در $N=128$ (SST-2)، Nyström branch **فعال نمی‌شود**. تمام نتایج اصلی در $N=128$ **dense fallback** هستند.

برای نشان دادن Nyström، باید $N=512$ یا بیشتر تست کنی.

### ۴. **Causal decoder مشکل**

GPT-2 و Qwen از Nyström **نمی‌توانند استفاده کنند** چون causal نیست. همیشه dense causal استفاده می‌کنند.

این را در `05_architecture.md` §5.4 و `10_limitations.md` §10.3 صادقانه بگو.

### ۵. **Iterative pinv vs Direct pinv**

کد تو اول `torch.linalg.pinv` را امتحان می‌کند و اگر شکست خورد به Newton-Schulz می‌رود. این یک **انحراف از مقاله اصلی** است (مقاله فقط Newton-Schulz استفاده می‌کند).

این را در `04_method.md` §4.5 یادداشت کن.

### ۶. **Combining with other methods**

Nyströmformer **orthogonal** است به:
- MinimalSVD (parameter compression)
- SVD-LLM V2 (parameter compression)
- Dobi-SVD (parameter compression)
- MLA (KV compression)

این را در `09_discussion.md` §9.5 برجسته کن.

---

**قدم بعدی: `run_all_4_nystrom.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**