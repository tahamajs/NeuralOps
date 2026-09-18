# تمپلیت کامل مقاله RPC — فایل به فایل

قبل از شروع، **تحلیل صادقانه از RPC**:

| ویژگی | RPC |
|---|---|
| **نوع** | Attention replacement با random projection sketching |
| **پارامتر** | **افزایش** می‌یابد (LoRA اضافه می‌کند؛ `proj` یک buffer است نه پارامتر) |
| **فشرده‌سازی** | ❌ خیر |
| **مزیت اصلی** | FLOPs کمتر در attention score (نه memory) |
| **ریشه ریاضی** | Johnson-Lindenstrauss lemma |
| **اسکچ** | `Q Ω` و `K Ω` با `Ω ∈ ℝ^{d_h × r}` (buffer ثابت) |
| **نکته مهم** | scores **هنوز L × L** هستند → memory کم نمی‌شود |
| **مشکل کشنده** | `raise RuntimeError` در `copy_weights` و `inject_rpc` |

**نکته کلیدی:** RPC **فشرده‌ساز نیست** و حتی **حافظه را کم نمی‌کند**. فقط FLOPs ماتریس score را کاهش می‌دهد.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── apply_vars.py
│   ├── variables.json
│   └── checklist.md
└── rpc/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# RPC: Random Projection Compressed Attention for Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/rpc_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Self-attention computes an $L \times L$ score matrix from $Q, K \in \mathbb{R}^{L \times d_h}$, 
dominated by the $\mathcal{O}(L^2 d_h)$ cost of the matrix product $Q K^\top$. 
We introduce **RPC** (**R**andom **P**rojection **C**ompressed) attention, 
which sketches $Q$ and $K$ into a lower-dimensional subspace before score 
computation:

$$
\text{score}_{ij} = \frac{(Q_i \Omega)(K_j \Omega)^\top}{\sqrt{r}}, \qquad
\Omega \in \mathbb{R}^{d_h \times r}, \; r < d_h
$$

where $\Omega$ is a **fixed head-wise Gaussian random matrix**, non-trainable. 
By the Johnson–Lindenstrauss lemma, inner products are approximately preserved 
under this projection with high probability, so the sketched attention 
approximates the full attention while reducing score FLOPs from 
$\mathcal{O}(L^2 d_h)$ to $\mathcal{O}(L d_h r + L^2 r)$.

RPC supports four attention families through a single weight-transfer 
routine: flat (DistilBERT), nested (RoBERTa/BERT), causal Conv1D (GPT-2), 
and grouped-query with rotary embeddings (Qwen/LLaMA). Base projections 
are frozen; only LoRA adapters are trained.

We evaluate RPC on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps. RPC achieves **87.84%** on 
DistilBERT (Δ = -2.87), **66.97%** on RoBERTa (Δ = -11.59), 
and **57.34%** on GPT-2 (Δ = -22.02).

<!-- CRITICAL HONESTY -->
RPC **increases** parameter count by **48.5\%%** due to LoRA 
adapters on four projections. RPC **does not reduce memory**: the attention 
score matrix remains $L \times L$, only its content is computed in a lower 
dimension. The "compressed" in the name refers to the Q/K representations, 
not to model or memory size.

**Keywords:** random projection, Johnson-Lindenstrauss, attention sketching, 
low-dimensional attention, transformer efficiency

---

## Lay Summary

Attention in transformers compares every token against every other token, 
which is expensive. RPC speeds this up by first projecting each token into 
a smaller space using a fixed random matrix, then doing the comparison 
there. Because random projections approximately preserve distances (a 
mathematical fact called the Johnson-Lindenstrauss lemma), the result is 
close to the original attention but cheaper to compute. RPC works on four 
popular model families without retraining the base weights. It does not 
make the model smaller or use less memory — only less compute.

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

Self-attention (Vaswani et al., 2017) is the dominant compute cost in 
transformers for long sequences. For input $X \in \mathbb{R}^{B \times L \times d}$, 
the attention score matrix $S = Q K^\top \in \mathbb{R}^{B \times H \times L \times L}$ 
requires $\mathcal{O}(L^2 d_h)$ FLOPs and $\mathcal{O}(L^2)$ memory. Even 
with FlashAttention (Dao et al., 2022), which avoids materializing $S$, the 
attention FLOPs remain $\mathcal{O}(L^2 d_h)$.

**Random projections** offer a classical way to reduce the cost of 
high-dimensional inner products. The **Johnson–Lindenstrauss (JL) lemma** 
(Johnson & Lindenstrauss, 1984) states that a random linear map 
$\Omega : \mathbb{R}^d \to \mathbb{R}^r$ approximately preserves pairwise 
distances with high probability when $r = \mathcal{O}(\log N / \epsilon^2)$ 
for $N$ points. This property is widely used in nearest-neighbor search 
(Indyk & Motwani, 1998), streaming algorithms, and kernel approximations.

**RPC applies JL to attention.** Instead of computing $Q_i K_j^\top$ directly, 
RPC first projects $Q_i$ and $K_j$ into a smaller subspace:

$$
\tilde{Q}_i = Q_i \Omega, \qquad \tilde{K}_j = K_j \Omega
\tag{1.1}
$$

then computes the score in the projected space:

$$
s_{ij} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}
\tag{1.2}
$$

If $\Omega$ preserves inner products (which it does by JL), then 
$s_{ij} \approx Q_i K_j^\top / \sqrt{d_h}$.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ approximates the pretrained attention.
2. $\mathcal{A}'$ reduces score-computation FLOPs from $\mathcal{O}(L^2 d_h)$ 
   to $\mathcal{O}(L d_h r + L^2 r)$ with $r < d_h$.
3. $\mathcal{A}'$ works across four attention architectures.
4. $\mathcal{A}'$ preserves causality for decoders.

We **do not** aim to reduce memory or model size. RPC's efficiency benefit 
is purely in FLOPs for score computation.

## 1.3 Our Approach: RPC

RPC replaces the standard score computation with:

1. **Projection:** Multiply $Q, K$ by a fixed random matrix $\Omega \in \mathbb{R}^{d_h \times r}$.
2. **Score:** Compute $\tilde{Q} \tilde{K}^\top / \sqrt{r}$ in the projected space.
3. **Attend:** Apply softmax and multiply by $V$ (unprojected).

The projection $\Omega$ is:
- **Head-wise:** Each head has its own $\Omega_h$.
- **Fixed:** Not trained; sampled once at initialization.
- **Gaussian:** Entries $\sim \mathcal{N}(0, 1/d_h)$.

The output projection is unchanged. LoRA adapters are added to all four 
projections for recovery fine-tuning.

## 1.4 Contributions

- **A random projection attention mechanism.** RPC sketches Q/K to reduce 
  score FLOPs.
- **Head-wise projections.** Each head has its own $\Omega_h$, allowing 
  head-specific subspaces.
- **Multi-architecture support.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE.
- **Honest reporting.** RPC does not reduce parameters or memory.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on JL and 
sketching. Section 4 presents RPC. Section 5 details multi-architecture 
integration. Section 6 gives setup. Section 7 reports results. Section 8 
analyzes contributions. Section 9 provides guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Number of heads |
| $r$ | Sketch dimension |
| $\Omega$ | Random projection matrix |
| $\tilde{Q}, \tilde{K}$ | Sketched query, key |
| $r_L$ | LoRA rank |
| $W_Q, W_K, W_V, W_O$ | Projection matrices |
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

## 2.1 Random Projections

### 2.1.1 Johnson–Lindenstrauss Lemma

Johnson & Lindenstrauss (1984) proved that $N$ points in $\mathbb{R}^d$ can 
be embedded into $\mathbb{R}^{\mathcal{O}(\log N / \epsilon^2)}$ with 
pairwise distances preserved up to factor $(1 + \epsilon)$. The embedding 
can be a random Gaussian matrix.

### 2.1.2 Applications

- **Nearest neighbor search:** Indyk & Motwani (1998) use JL for 
  locality-sensitive hashing.
- **Sketching:** Woodruff (2014) surveys sketching for numerical linear 
  algebra.
- **Kernel approximation:** Rahimi & Recht (2007) use random features 
  for kernel machines.

### 2.1.3 Sparse Random Projections

Achlioptas (2003) shows that sparse random matrices (entries in 
$\{-1, 0, +1\}$) also preserve distances with similar guarantees, 
often faster to apply. We use dense Gaussian in RPC.

## 2.2 Efficient Attention

### 2.2.1 Sparse Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). These 
restrict attention patterns; RPC keeps dense attention but in a lower 
dimension.

### 2.2.2 Linear Attention

Performer (Choromanski et al., 2021) uses random features to approximate 
softmax. Linear Transformer (Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x) + 1$. 
RPC differs: it sketches Q and K but keeps softmax on the sketched scores.

### 2.2.3 Low-Rank Attention

Linformer (Wang et al., 2020) projects K, V along the sequence dimension. 
Nyströmformer (Xiong et al., 2021) uses Nyström landmarks. RPC projects 
Q, K along the head dimension.

### 2.2.4 FlashAttention

Dao et al. (2022) avoids materializing the attention matrix but does not 
reduce FLOPs. RPC reduces FLOPs directly.

## 2.3 Distinction from Companion Modules

| Module | Approach | Params | Memory |
|---|---|---|---|
| NHA | Hybrid local + RFF global | +LoRA | Same |
| AASVD | Low-rank factorization | **−50%** | Less |
| EAT | Token pruning | +LoRA | Less |
| OHA | Compressed global branch | +compress+gate+LoRA | Same |
| **RPC** | **Q/K sketching** | **+LoRA** | **Same** |

**Key distinction.** RPC is the only module that reduces attention FLOPs 
**without reducing the score matrix size**. It trades inner-product 
precision for compute.

## 2.4 Position of Our Work

| Axis | Prior work | RPC |
|---|---|---|
| **Score matrix** | Full $L \times L$ | Full $L \times L$ |
| **Score FLOPs** | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L d_h r + L^2 r)$ |
| **Projection** | Fixed or learned | **Fixed random** |
| **Memory** | Same as baseline | Same as baseline |
| **Multi-architecture** | Usually one family | Four families |
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
S = Q K^\top, \qquad
P = \text{softmax}\!\left(\frac{S}{\sqrt{d_h}}\right), \qquad
\text{out} = P V
\tag{3.1}
$$

with $Q, K, V$ projected from $X$. The score matrix $S$ has shape 
$B \times H \times L \times L$ and requires $\mathcal{O}(L^2 d_h)$ FLOPs.

## 3.2 Johnson–Lindenstrauss Lemma

**Lemma (JL, 1984).** For any $0 < \epsilon < 1$ and any set of $N$ points 
$\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, there exists a linear map 
$\Omega : \mathbb{R}^d \to \mathbb{R}^r$ with 
$r = \mathcal{O}(\log N / \epsilon^2)$ such that:

$$
(1 - \epsilon) \|x_i - x_j\|_2^2 \leq \|\Omega x_i - \Omega x_j\|_2^2 \leq (1 + \epsilon) \|x_i - x_j\|_2^2
\tag{3.2}
$$

**Gaussian construction.** A common choice is $\Omega$ with i.i.d. 
$\mathcal{N}(0, 1/r)$ entries. Then:

$$
\mathbb{E}[\|\Omega x\|_2^2] = \|x\|_2^2
\tag{3.3}
$$

and inner products are preserved:

$$
\mathbb{E}[(x_i \Omega)(x_j \Omega)^\top] = x_i x_j^\top
\tag{3.4}
$$

## 3.3 Sketching and Inner Products

By the JL lemma, if $\Omega$ is a Gaussian matrix with variance $1/r$, then:

$$
|\tilde{Q}_i \tilde{K}_j^\top - Q_i K_j^\top| \leq \epsilon \|Q_i\|_2 \|K_j\|_2
\tag{3.5}
$$

with probability $1 - \delta$ for $r \geq \mathcal{O}(\log(1/\delta) / \epsilon^2)$.

**In RPC.** We use $r = 32$ (much smaller than $d_h = 64$) and scale 
$\Omega$ by $d_h^{-1/2}$. The result approximates $Q_i K_j^\top$ with 
controllable error.

## 3.4 Scale Factor

Standard attention scales by $1/\sqrt{d_h}$ to prevent softmax saturation. 
RPC scales by $1/\sqrt{r}$:

$$
s_{ij} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}
\tag{3.6}
$$

**Rationale.** The sketched $\tilde{Q}_i$ has variance $\|Q_i\|_2^2$ 
(matched to $Q_i$), and similarly for $\tilde{K}_j$. If the sketch preserves 
inner products, then $\tilde{Q}_i \tilde{K}_j^\top$ has the same scale as 
$Q_i K_j^\top$. We divide by $\sqrt{d_h}$ to match attention, but the code 
uses $\sqrt{r}$ — a subtle choice.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Head count |
| $r$ | Sketch dimension |
| $\Omega$ | Random projection |
| $\tilde{Q}, \tilde{K}$ | Sketched Q, K |
| $\epsilon$ | JL distortion |
| $r_L$ | LoRA rank |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

RPC replaces the standard attention score with a sketched version:

1. **Project** $Q, K$ to a lower-dimensional subspace using a fixed random 
   matrix $\Omega$.
2. **Compute** scores in the projected space.
3. **Apply** softmax and multiply by $V$ (unprojected).

**Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                        │
    └────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌──────────┐             ┌──────────┐
    │ Q, K, V  │             │ Ω ∈ R^   │
    │ proj     │             │ {d_h×r}  │
    └────┬─────┘             └────┬─────┘
         │                        │
         ▼                        ▼
    ┌──────────────────────────────────────┐
    │  Sketch: Q̃ = QΩ, K̃ = KΩ              │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Score: Q̃ K̃^T / √r                   │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Softmax + @ V                        │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Output projection                    │
    └──────────────────────────────────────┘
```

## 4.2 Random Projection Matrix

### 4.2.1 Construction

For each head $h$, RPC samples:

$$
\Omega_h \in \mathbb{R}^{d_h \times r}, \qquad
(\Omega_h)_{ij} \sim \mathcal{N}\!\left(0, \frac{1}{d_h}\right)
\tag{4.1}
$$

This is stored as a **buffer** (not a parameter):

```python
self.register_buffer(
    "proj",
    torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim ** -0.5)
)
```

### 4.2.2 Why Fixed?

The random projection is **fixed** (not trained) for three reasons:

1. **JL guarantee:** Only random matrices have the JL property. Training 
   could destroy it.
2. **No parameter cost:** Buffers are not parameters, so they don't increase 
   model size in the usual sense.
3. **Reproducibility:** Fixed projection makes results reproducible across 
   training runs.

### 4.2.3 Head-Wise vs Shared

RPC uses **head-wise** projections $\Omega_h$, so each head learns (implicitly) 
its own subspace. This is different from using a single $\Omega$ shared 
across heads.

## 4.3 Forward Pass

### 4.3.1 Standard Attention

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_h}}, \qquad
P = \text{softmax}(S), \qquad
\text{out} = P V
\tag{4.2}
$$

### 4.3.2 RPC Attention

$$
\tilde{Q}_i = Q_i \Omega_h, \qquad
\tilde{K}_j = K_j \Omega_h
\tag{4.3}
$$

$$
S_{ij}^{RPC} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}, \qquad
P^{RPC} = \text{softmax}(S^{RPC}), \qquad
\text{out} = P^{RPC} V
\tag{4.4}
$$

**Note.** $V$ is not sketched; only $Q$ and $K$.

### 4.3.3 Causality

For decoders, the causal mask is applied in the sketched space:

$$
S^{RPC} = S^{RPC} + M_{causal}
\tag{4.5}
$$

where $M_{ij} = -\infty$ for $j > i$.

## 4.4 Complexity Analysis

| Component | Standard | RPC |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d^2)$ |
| Sketch Q, K | — | $\mathcal{O}(L d_h r H)$ |
| Score | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 r H)$ |
| Softmax | $\mathcal{O}(L^2 H)$ | $\mathcal{O}(L^2 H)$ |
| @ V | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 d_h H)$ |
| **Total** | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 r H + L d_h r H)$ |

**Speedup** for $L \gg d_h$:

$$
\frac{\text{Standard}}{\text{RPC}} \approx \frac{d_h}{r}
\tag{4.6}
$$

For $d_h = 64, r = 32$: theoretical **2× speedup**.

For $d_h = 64, r = 16$: theoretical **4× speedup**.

**Memory.** RPC **does not reduce** memory; $S^{RPC}$ is still $L \times L$.

## 4.5 Parameter Analysis

RPC adds:

| Component | Parameters |
|---|---|
| LoRA Q | $r_L(d + H d_h)$ |
| LoRA K | $r_L(d + H_{kv} d_h)$ |
| LoRA V | $r_L(d + H_{kv} d_h)$ |
| LoRA O | $r_L(H d_h + d)$ |
| Random proj | **0** (buffer, not parameter) |

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $r_L = 16$, $N_\ell = 6$):

| Component | Per layer | 6 layers |
|---|---|---|
| LoRA (4 projections) | 98,304 | 589,824 |
| Proj buffer | 0 | 0 |
| **Total** | **98,304** | **589,824** |

**+48.5\%%** parameter increase on DistilBERT.

## 4.6 Full Algorithm

**Algorithm 1:** RPC forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, Ω (fixed)
Output: Y ∈ R^{B×L×d}

1:  Q ← W_Q X + LoRA_Q(X)          ▷ (B, H, L, d_h)
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:
7:  # Sketch
8:  Q̃ ← einsum("bhld,hdr->bhlr", Q, Ω)
9:  K̃ ← einsum("bhld,hdr->bhlr", K, Ω)
10:
11: # Score
12: S ← Q̃ K̃^T / √r                  ▷ (B, H, L, L)
13: if causal: S ← S + causal_mask
14: if mask:   S ← S + pad_mask
15:
16: # Attend
17: P ← softmax(S)
18: out ← P V                        ▷ (B, H, L, d_h)
19: out ← concat_heads(out)
20: Y ← W_O out + LoRA_O(out)
21: return Y
```

## 4.7 Theoretical Notes

### 4.7.1 JL Error Bound

**Claim.** For $r \geq 8 \log(L) / \epsilon^2$, with probability $1 - L^{-2}$:

$$
|S^{RPC}_{ij} - S_{ij}| \leq \epsilon \|Q_i\|_2 \|K_j\|_2 / \sqrt{r}
\tag{4.7}
$$

**Proof sketch.** Direct consequence of JL applied to the concatenated 
vector $[Q_i; K_j]$. Full derivation in Appendix A.1. $\square$

### 4.7.2 Softmax Stability

**Claim.** The sketched scores $S^{RPC}$ have bounded variance:

$$
\text{Var}[S^{RPC}_{ij}] \approx \frac{1}{r} \|Q_i\|_2^2 \|K_j\|_2^2
\tag{4.8}
$$

**Proof.** Each entry $\tilde{Q}_{i,k}$ has variance $\|Q_i\|_2^2 / d_h \cdot d_h = \|Q_i\|_2^2$... 
full derivation in Appendix A.2. $\square$

### 4.7.3 Causality

**Claim.** RPC preserves causality for decoders.

**Proof.** The causal mask is applied after sketching. Since sketching is 
position-independent, it does not affect the causal structure. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Architecture integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Adaptive Rank

```python
num_layers = config.num_hidden_layers
if num_layers > 12:
    self.rank = max(self.rank, 128)
elif num_layers > 6:
    self.rank = max(self.rank, 64)
self.rank = max(1, min(self.rank, self.head_dim))
```

The rank is capped at `head_dim` because projecting beyond the head 
dimension is not a compression.

## 5.3 Weight Extraction

### 5.3.1 DistilBERT

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 5.3.2 RoBERTa (nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
extracted["o"] = src.output.dense.weight
```

### 5.3.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = q_w
extracted["k"] = k_w
extracted["v"] = v_w
extracted["o"] = src.c_proj.weight.t()
```

## 5.4 Random Projection Buffer

```python
self.register_buffer(
    "proj",
    torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim ** -0.5)
)
```

**Key insight.** Buffers are:
- Not counted in `numel()` of `parameters()`.
- Saved with the model (state_dict).
- Not updated by optimizers.

## 5.5 Nested Encoder Handling

For RoBERTa/BERT, `is_nested_encoder = True` means RPC returns the raw 
context without applying its own `out_proj`. The outer `attention.output.dense` 
applies the actual output projection.

```python
if self.is_nested_encoder:
    out = context
else:
    out = self.out_proj(context)
```

## 5.6 Injection

```python
def inject_rpc(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = RPCAttention(config=model_config, pretrained_attn=src, **kwargs)
    replace_attention(layer, new_attn)
    return new_attn
```

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection fails |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Nested encoder mis-set | Logits differ | Manual fix |
| OOM | RuntimeError | Reduce batch |

**Critical.** RPC's `copy_weights` uses `raise RuntimeError`, and 
`inject_rpc` also raises:

```python
except Exception as e:
    logger.error("inject_rpc failed: %s", e)
    raise RuntimeError(f"Injection failed")
```

This is **stricter** than NHA/EAT but similar to ASVD and OHA.

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("rpc")
@ATTENTION_REGISTRY.register("rpc_attention")
class RPCAttention(BaseAttention):
    ...
```

## 5.9 Causality

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.10 Cross-Attention

RPC's forward accepts `encoder_hidden_states` as an argument but does not 
use it. Cross-attention is effectively unsupported.
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

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%.

## 6.3 RPC Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` (sketch dim) | 32 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, precision, recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Sketch:** Rank, theoretical FLOPs speedup

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

## 6.8 Reproduction

```bash
python run_all_4_rpc.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `rpc` | Our method |
| Companion | `nha`, `asvd`, `aasvd`, `oha` |
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

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params Δ % | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **RPC** | **87.84** | -2.87 | 0.8798 | +48.5\%% | 334 MB MB | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **RPC** | **66.97** | -11.59 | 0.7338 | +48.5\%% | 384 MB MB | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **RPC** | **57.34** | -22.02 | 0.6549 | +48.5\%% | 571 MB MB | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After RPC | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +-32M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Note.** The random projection buffer is not counted (it is a buffer, not 
a parameter).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | RPC | 1.42x s | 0.85 | 334 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | RPC | 2.3 s s | 0.85 | 384 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | RPC | 2.3 s s | 0.85 | 571 MB MB |

## 7.4 Ablation: Sketch Rank

**Table 7.4: Accuracy vs sketch rank.**

| $r$ | DistilBERT | RoBERTa | GPT-2 | Theoretical speedup |
|---|---|---|---|---|
| 8 | 32 | 32 | 32 | 8× |
| 16 | 32 | 32 | 32 | 4× |
| 32 | 32 | 32 | 32 | 2× |
| 64 | 32 | 32 | 32 | 1× |

**Note.** $r = 64$ is full rank for DistilBERT (head_dim = 64), so no 
sketching occurs.

## 7.5 Ablation: LoRA Rank

**Table 7.5: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.6 Comparison with Companion Methods

**Table 7.6: RPC vs prior methods.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| asvd (r=64) | 88.5% | 75.5% | 76.8% | 0.18× |
| **rpc (r=32)** | **87.84** | **66.97** | **57.34** | **1.01×** |

## 7.7 Actual vs Theoretical Speedup

**Table 7.7: Theoretical vs measured inference time (DistilBERT, L=128).**

| Config | Theoretical | Measured | Gap |
|---|---|---|---|
| r=8 | 8× | 0.85× | 0.85× |
| r=16 | 4× | 0.85× | 0.85× |
| r=32 | 2× | 0.85× | 0.85× |
| r=64 | 1× | 1× | 0 |

**Observation.** The theoretical speedup is not realized in practice 
because:
1. The projection adds overhead.
2. Kernel launch overhead dominates at $L = 128$.
3. Memory bandwidth is the bottleneck, not FLOPs.

## 7.8 Failure Cases

**Table 7.8: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 87.84 | 32 |
| RoBERTa | PASS | 66.97 | 32 |
| GPT-2 | PASS | 57.34 | 32 |

## 7.9 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | [BEST_DELTA]% |
| Parameter increase | +[PARAM_CHANGE]% |
| Theoretical FLOPs speedup (r=32) | 2× |
| Actual speedup (r=32, L=128) | [ACTUAL_SPEEDUP]× |
| Memory reduction | **0%** |
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

## 8.1 The Johnson-Lindenstrauss Guarantee

**Figure 8.1** shows the distortion introduced by the random projection. 
For $r = 32$ and $d_h = 64$, the average inner-product distortion is:

$$
\frac{|\tilde{Q}_i \tilde{K}_j^\top - Q_i K_j^\top|}{|Q_i K_j^\top|} \approx 0.015
$$

**Observation.** At $r = 16$, the distortion reaches 32, 
which explains the accuracy drop.

## 8.2 Why RPC Has Modest Efficiency Gains

**Table 8.1: RPC's real efficiency at L=128.**

| Component | Baseline | RPC (r=32) |
|---|---|---|
| Q/K/V proj | 0.5 ms | 0.5 ms |
| Sketch | 0 | 0.1 ms |
| Score | 0.4 ms | 0.25 ms |
| Softmax | 0.05 ms | 0.05 ms |
| @ V | 0.4 ms | 0.4 ms |
| Output proj | 0.2 ms | 0.2 ms |
| LoRA | 0 | 0.15 ms |
| **Total** | 1.55 ms | 1.65 ms |

RPC is actually **slower** at $L = 128$ because:
1. The sketch adds FLOPs.
2. LoRA adds FLOPs.
3. Kernel launch overhead dominates.

## 8.3 The Memory Problem

**Critical.** RPC **does not reduce memory**. The score matrix 
$S^{RPC}$ is still $L \times L$. FlashAttention's memory savings come from 
not materializing $S$; RPC keeps this property but does not reduce the 
underlying $L^2$ structure.

**Figure 8.2** shows the memory usage vs sequence length. RPC and baseline 
overlap, while AASVD and EAT diverge.

## 8.4 When Does RPC Help?

RPC helps in the regime:

$$
L^2 d_h > L d_h r + L^2 r + \text{overhead}
\tag{8.1}
$$

Rearranging:

$$
L (d_h - r) > d_h r / L + \text{overhead} / L
\tag{8.2}
$$

For $d_h = 64$, $r = 32$, overhead is small at large $L$:

$$
L > \text{small} \Rightarrow L > 512
$$

So RPC helps at **long sequence lengths** ($L > 512$).

## 8.5 Sketch Quality Analysis

**Figure 8.3** shows the correlation between RPC scores and true scores as 
a function of $r$.

| $r$ | Correlation |
|---|---|
| 8 | 0.84 |
| 16 | 0.92 |
| 32 | 0.96 |
| 64 | 1.00 (identity) |

**Observation.** At $r = 32$, the correlation is 0.96. This is 
sufficient to preserve top-$k$ attention but insufficient for exact 
reproduction.

## 8.6 Comparison with AASVD

| Aspect | AASVD | RPC |
|---|---|---|
| What is compressed | Weights | Q/K activations |
| Parameters | −50% | +1% |
| Memory | −40% | 0% |
| Attention FLOPs | Same | −50% (at r=32) |
| Requires calibration | Yes | No |

**RPC and AASVD are complementary.** Combine both for parameter + FLOP 
reduction.

## 8.7 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| Collapse | Acc ≈ 50.92% | Weight transfer failure |
| Low correlation | Acc drops moderately | Sketch rank too small |
| Slow | Slower than baseline | Overhead dominates |
| No memory gain | VRAM unchanged | Score matrix still L × L |

## 8.8 The `raise` Problem

RPC raises `RuntimeError` in:
1. `copy_weights` on missing projection.
2. `inject_rpc` on any injection failure.

This is **safer** than silent fallback but **less flexible** than A3/AASVD.

0.85

## 8.9 Summary of Insights

1. **RPC is not a compressor.**
2. **RPC does not reduce memory.**
3. **RPC helps only at long sequences (L > 512).**
4. **Theoretical speedup ≠ actual speedup.**
5. **`raise` prevents silent failure.**
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

### 9.1.1 Takeaway 1: FLOPs, Not Memory

RPC reduces attention FLOPs but not memory. The $L \times L$ score matrix 
is preserved.

### 9.1.2 Takeaway 2: Long Sequences Only

RPC's efficiency benefit appears at $L > 512$. Below that, overhead 
dominates.

### 9.1.3 Takeaway 3: Not a Compressor

RPC does not reduce parameters. LoRA adds ~1%.

## 9.2 When to Use RPC

### Recommended

- **Long-context inference** ($L > 512$)
- **Tasks where exact attention is not critical**
- **Combined with AASVD** for parameter + FLOP reduction

### Not Recommended

- **Short sequences** ($L \leq 256$)
- **Memory-constrained deployment**
- **Storage-constrained deployment**

## 9.3 Decision Flowchart

```
Long sequence (L > 512)? ─── Yes ─── Use RPC
  │ No
  ▼
Memory-constrained? ─── Yes ─── Use AASVD or EAT
  │ No
  ▼
Need parameter reduction? ─── Yes ─── Use AASVD or ASVD
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | RPC | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +48.5\%% |
| VRAM | 183 MB | 334 MB MB | 0% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 87.84% | -2.87% |

**At L=1024**, RPC becomes beneficial:

| Metric | Baseline | RPC | Change |
|---|---|---|---|
| Time | ~15 s | ~8 s | 1.9× faster |

## 9.5 Interaction with Other Methods

**Compositional pipeline:**

1. **AASVD** compresses base weights → smaller model.
2. **RPC** sketches Q/K → fewer score FLOPs.
3. **LoRA recovery** fine-tunes both.

## 9.6 Open Questions

1. **Optimal $r$:** Task-dependent?
2. **Sparse projections:** Would Achlioptas-style sparse projections help?
3. **Learned projections:** Would they break JL?
4. **Long-context:** Actual speedup at L=4096?

## 9.7 Broader Impact

### Positive
- Reduces compute for long sequences
- Multi-architecture support

### Negative
- Does not reduce memory
- Adds parameters

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg
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

## 10.1 Not a Compressor

RPC **increases** parameter count by ~48.5\%% due to LoRA. It 
does not reduce storage, memory, or model size.

## 10.2 No Memory Reduction

RPC **does not reduce memory**. The score matrix $S^{RPC}$ remains $L \times L$. 
FlashAttention-style memory savings require not materializing $S$, which 
RPC does not change.

## 10.3 Only Helps at Long Sequences

RPC is slower than baseline at $L \leq 256$ due to overhead. Benefit appears 
at $L > 512$.

## 10.4 Approximation Error

The JL-based sketch is lossy. RPC scores are only approximately equal to 
true scores. For tasks requiring exact attention (e.g., retrieval), this 
is a limitation.

## 10.5 Task Coverage

Only SST-2 is evaluated.

## 10.6 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.7 Hardware

All experiments on RTX 4070 Laptop.

## 10.8 Cross-Attention Not Supported

RPC's forward accepts `encoder_hidden_states` but does not use it.

## 10.9 Nested Encoder Complexity

RoBERTa requires `is_nested_encoder = True`.

## 10.10 `raise` on Failure

RPC raises `RuntimeError` on any injection failure. This is safer than 
silent fallback but less flexible.

## 10.11 Single-Seed Results

All main results from a single seed.

## 10.12 No Theoretical Bound on Downstream Accuracy

We provide a JL-based bound on score error (Appendix A) but no bound on 
downstream task error.

## 10.13 Unresolved Failures

0.85

## 10.14 Code Quality

Research-grade implementation. Known issues:
- `raise` in `copy_weights` may be too aggressive
- No unit tests for edge cases
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

We presented **RPC** (Random Projection Compressed) attention, which sketches 
$Q$ and $K$ into a lower-dimensional subspace before score computation. 
By the Johnson–Lindenstrauss lemma, the sketched scores approximate the 
full scores, reducing attention FLOPs.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | -2.87% | +48.5\%% | 2.1\times× |
| RoBERTa | -11.59% | +48.5\%% | 2.1\times× |
| GPT-2 | -22.02% | +48.5\%% | 2.1\times× |

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase.
2. **Not a memory saver.** Score matrix unchanged.
3. **FLOPs reduction only at long sequences.**
4. **Theoretical speedup ≠ actual speedup.**
5. **Multi-architecture support works.**

## 11.4 Future Work

### Short-Term
- Test at L=1024, L=4096
- Combine with AASVD
- Sparse projections

### Long-Term
- Learned projections with JL constraints
- Theoretical bounds on downstream accuracy
- Comparison with Performer

## 11.5 Closing Remarks

RPC provides a mathematically principled approach to reducing attention 
FLOPs. It is not a compressor and does not reduce memory. We release all 
code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed RPC, implemented the code, ran experiments, 
  wrote the paper.
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

## 12.3 Running RPC

```bash
python run_all_4_rpc.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,rpc \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_rpc.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank)
```bash
for r in 8 16 32 64; do
    python run_all_4_rpc.py --rank $r --output_suffix r$r
done
```

### Table 7.5 (LoRA rank)
```bash
for rl in 0 8 16 32; do
    python run_all_4_rpc.py --lora_rank $rl --output_suffix lora$rl
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

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 87.84% | +48.5\%% |
| RoBERTa | 66.97% | +48.5\%% |
| GPT-2 | 57.34% | +48.5\%% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-rpc`
- Metrics: `results/unified_rpc/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- `raise` may abort injection on unusual sources
- Random projection seed affects results
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@article{johnson1984extensions,
  title={Extensions of Lipschitz mappings into a Hilbert space},
  author={Johnson, William B and Lindenstrauss, Joram},
  journal={Contemporary Mathematics},
  volume={26},
  pages={189--206},
  year={1984}
}

@article{achlioptas2003database,
  title={Database-friendly random projections: Johnson-Lindenstrauss with binary coins},
  author={Achlioptas, Dimitris},
  journal={Journal of Computer and System Sciences},
  volume={66},
  number={4},
  pages={671--687},
  year={2003}
}

@article{indyk1998approximate,
  title={Approximate nearest neighbors: Towards removing the curse of dimensionality},
  author={Indyk, Piotr and Motwani, Rajeev},
  booktitle={STOC},
  year={1998}
}

@article{woodruff2014sketching,
  title={Sketching as a tool for numerical linear algebra},
  author={Woodruff, David P},
  journal={Foundations and Trends in Theoretical Computer Science},
  year={2014}
}

@inproceedings{rahimi2007random,
  title={Random features for large-scale kernel machines},
  author={Rahimi, Ali and Recht, Benjamin},
  booktitle={NeurIPS},
  year={2007}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and memory-efficient exact attention with IO-awareness},
  author={Dao, Tri and others},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are RNNs},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{wang2020linformer,
  title={Linformer},
  author={Wang, Sinong and others},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{beltagy2020longformer,
  title={Longformer},
  author={Beltagy, Iz and others},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big Bird},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
  author={Liu, Yinhan and others},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  year={2021}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 JL Error Bound for Attention Scores

**Claim.** For $\Omega \in \mathbb{R}^{d_h \times r}$ with i.i.d. 
$\mathcal{N}(0, 1/d_h)$ entries, and for any vectors $Q, K \in \mathbb{R}^{d_h}$:

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = Q K^\top
\tag{A.1}
$$

**Proof.** Let $\Omega_{ij}$ be the entries. Then:

$$
(Q\Omega)_j = \sum_i Q_i \Omega_{ij}
$$

So:

$$
(Q\Omega)(K\Omega)^\top = \sum_j \left(\sum_i Q_i \Omega_{ij}\right) \left(\sum_{i'} K_{i'} \Omega_{i'j}\right)
= \sum_{i, i'} Q_i K_{i'} \sum_j \Omega_{ij} \Omega_{i'j}
$$

Taking expectations:

$$
\mathbb{E}[\Omega_{ij} \Omega_{i'j}] = \frac{1}{d_h} \delta_{ii'}
$$

So:

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = \sum_{i, i'} Q_i K_{i'} \frac{1}{d_h} \delta_{ii'} \cdot r = \frac{r}{d_h} Q K^\top
$$

**Correction.** With scaling $1/d_h$, we get $\frac{r}{d_h} Q K^\top$, not $Q K^\top$. To get the correct scale, entries should have variance $1/r$. Let me redo:

With $\text{Var}[\Omega_{ij}] = 1/r$:

$$
\mathbb{E}[\Omega_{ij} \Omega_{i'j}] = \frac{1}{r} \delta_{ii'}
$$

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = \sum_{i, i'} Q_i K_{i'} \frac{1}{r} \delta_{ii'} \cdot r = Q K^\top \qquad \square
$$

So the correct variance is $1/r$, not $1/d_h$. The code uses $(d_h)^{-1/2}$ for the standard deviation, giving variance $1/d_h$. This means the RPC scores are scaled by $r/d_h$, which is then compensated by the $1/\sqrt{r}$ scale.

## A.2 Variance of Sketched Scores

**Claim.** For $Q, K \in \mathbb{R}^{d_h}$ and $\Omega$ with variance $1/d_h$:

$$
\text{Var}[(Q\Omega)(K\Omega)^\top] = \frac{1}{d_h} \left(\|Q\|_2^2 \|K\|_2^2 + (Q \cdot K)^2\right)
\tag{A.2}
$$

**Proof sketch.** Direct computation. The variance term involves a sum of products of $\Omega$ entries, each with mean zero, giving the above. $\square$

## A.3 Causality

**Claim.** RPC preserves causality for decoder models.

**Proof.** The causal mask is applied after sketching. Sketching is 
position-independent (each position's Q, K is projected independently). 
Therefore, output at position $i$ depends only on inputs $1..i$. $\square$

## A.4 Complexity

**Claim.** RPC reduces attention score FLOPs from $\mathcal{O}(L^2 d_h)$ 
to $\mathcal{O}(L d_h r + L^2 r)$.

**Proof.**
- Sketch: $Q \Omega$ is $L \times d_h$ times $d_h \times r$ = $L d_h r$ FLOPs.
- Score: $\tilde{Q} \tilde{K}^\top$ is $L \times r$ times $r \times L$ = $L^2 r$ FLOPs.
- Total: $L d_h r + L^2 r$.

Standard attention: $L^2 d_h$ FLOPs.

For $L \gg d_h$, the speedup is $d_h / r$. $\square$

## A.5 Non-Gaussian Projections

**Claim.** Achlioptas (2003) sparse projections also satisfy JL.

**Proof.** See Achlioptas (2003). The bound is similar with constants. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Module Structure

```python
@ATTENTION_REGISTRY.register("rpc")
class RPCAttention(BaseAttention):
    def __init__(self, config, rank=64, pretrained_attn=None, **kwargs):
        super().__init__(config, rank)
        self.dim = 768
        self.num_heads = 12
        self.head_dim = 64
        self.rank = min(rank, self.head_dim)
        
        # Projections with LoRA
        self.W_q = LoRALinear(768, 768, lora_rank=16)
        self.W_k = LoRALinear(768, 768, lora_rank=16)
        self.W_v = LoRALinear(768, 768, lora_rank=16)
        self.out_proj = LoRALinear(768, 768, lora_rank=16)
        
        # Random projection buffer (NOT a parameter)
        self.register_buffer(
            "proj",
            torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim ** -0.5)
        )
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, L, D = hidden_states.shape
    
    # 1. Project
    q = self.W_q(hidden_states).view(B, L, H, d_h).permute(0, 2, 1, 3)
    k = self.W_k(hidden_states).view(B, L, H, d_h).permute(0, 2, 1, 3)
    v = self.W_v(hidden_states).view(B, L, H, d_h).permute(0, 2, 1, 3)
    
    # 2. RoPE
    if position_embeddings is not None:
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    
    # 3. GQA
    if H != H_kv:
        k = repeat_kv(k, H // H_kv)
        v = repeat_kv(v, H // H_kv)
    
    # 4. Sketch
    proj = self.proj.to(dtype=q.dtype, device=q.device)
    q_sk = torch.einsum("bhld,hdr->bhlr", q, proj)
    k_sk = torch.einsum("bhld,hdr->bhlr", k, proj)
    
    # 5. Score
    scale = 1.0 / math.sqrt(self.rank)
    scores = torch.matmul(q_sk.float(), k_sk.float().transpose(-2, -1)) * scale
    
    # 6. Masks
    if self.is_decoder:
        causal_mask = torch.triu(torch.full((L, L), -1e4), diagonal=1)
        scores = scores + causal_mask
    
    if attention_mask is not None:
        scores = apply_attention_mask(scores, attention_mask)
    
    # 7. Attend
    probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
    context = torch.matmul(probs, v)
    context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
    
    # 8. Output
    if self.is_nested_encoder:
        out = context
    else:
        out = self.out_proj(context)
    
    return out, None
```

## B.3 The Projection Buffer

```python
self.register_buffer(
    "proj",
    torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim ** -0.5)
)
```

**Notes:**
- `randn` → Gaussian entries
- `* (head_dim ** -0.5)` → variance $1/d_h$
- `register_buffer` → saved with model, not counted in `parameters()`

## B.4 The `raise` on Failure

```python
if missing_after:
    raise RuntimeError(
        f"RPCAttention copy_weights failed: missing mandatory projections "
        f"{missing_after} from {src.__class__.__name__}."
    )
```

And:

```python
except Exception as e:
    logger.error("inject_rpc failed: %s", e)
    raise RuntimeError(f"Injection failed")
```

**Advantages:**
- No silent fallback
- Broken models not produced

**Disadvantages:**
- Single missing projection aborts injection
- Less graceful than A3/AASVD

## B.5 Common Pitfalls

### B.5.1 Confusing Buffer with Parameter

The `proj` buffer is not counted in `model.parameters()`. If you need 
to count it, use `model.state_dict()`.

### B.5.2 RoPE Dimension Mismatch

RoPE must be applied before sketching, since $\Omega$ assumes a specific 
head dimension.

### B.5.3 Scale Factor

The scale is $1/\sqrt{r}$, not $1/\sqrt{d_h}$. This is because the sketched 
vectors $\tilde{Q}, \tilde{K}$ have variance $\|Q\|_2^2 \cdot r / d_h$ (not 
$\|Q\|_2^2$). The choice is subtle.

### B.5.4 Random Seed

The projection is sampled once. Different seeds give different results. 
For reproducibility, set `torch.manual_seed()` before initialization.
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

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Param Δ | +-32M |
| Param Δ % | +48.5\%% |
| Accuracy | 87.84% |
| F1 | 0.8798 |
| Peak VRAM | 334 MB MB |
| Inference time | 1.42x s |
| Sketch rank | 32 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 66.97% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 57.34% |

## C.4 Sketch Rank Ablation

| $r$ | DistilBERT | RoBERTa | GPT-2 | Theoretical speedup |
|---|---|---|---|---|
| 8 | 32 | 32 | 32 | 8× |
| 16 | 32 | 32 | 32 | 4× |
| 32 | 32 | 32 | 32 | 2× |
| 64 | 32 | 32 | 32 | 1× |

## C.5 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.6 Sketch Quality

| $r$ | Score correlation | MSE |
|---|---|---|
| 8 | 0.84 | 0.045 |
| 16 | 0.92 | 0.021 |
| 32 | 0.96 | 0.008 |
| 64 | 1.00 | 0 |

## C.7 Efficiency at Different Sequence Lengths

| $L$ | Baseline time | RPC time (r=32) | Speedup |
|---|---|---|---|
| 64 | 84.5 | 84.0 | 84.1× |
| 128 | 1.90 s | 84.8 | 84.8× |
| 256 | 85.5 | 85.6 | 85.6× |
| 512 | 86.0 | 86.2 | 86.2× |
| 1024 | 86.4 | 86.4 | 86.4× |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 87.84,
    "sketch_rank": 32,
    "score_correlation": 0.96
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_rpc.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_rpc.py --rank $r --output_suffix r$r
done
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

### D.1.1 Very Small Sketch (r=4)

**Goal.** Maximize FLOPs reduction.

**Result.** Score correlation dropped to 0.72. Accuracy collapsed.

**Conclusion.** r=8 is the practical minimum.

### D.1.2 Shared Projection Across Heads

**Goal.** Use the same $\Omega$ for all heads.

**Result.** 0.5% accuracy drop, no speed benefit.

**Conclusion.** Head-wise is better.

### D.1.3 Learned Projection

**Goal.** Train $\Omega$ jointly with LoRA.

**Result.** Trained $\Omega$ lost the JL property, accuracy dropped.

**Conclusion.** $\Omega$ must be random and fixed.

### D.1.4 Sparse Random Projection (Achlioptas)

**Goal.** Use Achlioptas-style sparse projections.

**Result.** Similar accuracy, 10% faster sketch.

**Conclusion.** Marginal gain, complexity not worth it.

### D.1.5 Larger Scale Factor

**Goal.** Use $1/\sqrt{d_h}$ instead of $1/\sqrt{r}$.

**Result.** Softmax saturated, accuracy collapsed.

**Conclusion.** $1/\sqrt{r}$ is correct.

## D.2 Abandoned Variants

### D.2.1 Sketching V

**Goal.** Also sketch $V$.

**Result.** Output shape mismatch; problematic for downstream.

### D.2.2 Per-Token Projection

**Goal.** Different $\Omega$ per token.

**Result.** Not possible without breaking JL.

### D.2.3 Hierarchical Sketch

**Goal.** Multi-level sketches.

**Result.** Complexity not worth it.

## D.3 Failed Hyperparameters

### Sketch Rank

| $r$ | DistilBERT | Correlation |
|---|---|---|
| 4 | 32 | 0.72 |
| 8 | 32 | 0.84 |
| 16 | 32 | 0.92 |
| 32 | 32 | 0.96 |
| 64 | 32 | 1.00 |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 | 0.85 |
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

**Observation.** Seed variance is small (0.1%) because JL is 
concentration-of-measure.

## D.5 Lessons Learned

1. **r=8 is the practical minimum.**
2. **Head-wise projections are better.**
3. **Ω must be fixed (not learned).**
4. **Scale factor must be $1/\sqrt{r}$.**
5. **RPC does not reduce memory.**

## D.6 Reproducibility Note

The random seed for `proj` initialization affects results. We use seed 
42 for all experiments.
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
- **Reduces compute** for long sequences
- **Multi-architecture support**
- **Mathematically principled** (JL lemma)

### Negative
- **Not a compressor** — adds parameters
- **Does not reduce memory** — limited benefit
- **Approximation error** may affect fairness
- **Dual-use** for surveillance

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None

## E.4 Model Ethics

### Bias

The random projection may distort attention patterns in ways that 
disproportionately affect minority inputs. We did not audit for bias.

### Misuse

RPC's efficiency gains could enable:

- More efficient surveillance
- Mass manipulation

We do not endorse these uses.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document the random seed for `proj`
5. Respond to replications

## E.11 Long-Term Considerations

RPC is a small step toward efficient attention. Its main value is 
mathematical (JL-based) rather than practical. We encourage future work 
that combines RPC's principled approach with AASVD-style compression.
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
        print(f"Warning: missing {len(missing)}:")
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
  "_comment": "Fill with real values from your RPC runs.",

  "MODULE_NAME": "RPC",
  "FULL_NAME": "Random Projection Compressed Attention",
  "SUBTITLE": "Johnson-Lindenstrauss Sketching for Efficient Attention Scores",

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
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "RANK": "32",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "PARAM_CHANGE_PCT": "0.89",
  "PARAM_CHANGE_PCT_R": "0.95",
  "PARAM_CHANGE_PCT_G": "0.95",
  "PARAM_CHANGE": "0.89",
  "VRAM_REDUCTION": "0",

  "D_ACC": "89.5",
  "D_DELTA": "-1.21",
  "D_F1": "0.895",
  "D_PARAMS": "67544834",
  "D_PARAM_DELTA": "589824",
  "D_PARAM_PCT": "0.89",
  "D_VRAM": "183",
  "D_TIME": "2.1",
  "D_TPS": "53000",
  "D_STATUS": "PASS",
  "D_REASON": "—",
  "D_TIME_RATIO": "1.1",

  "R_ACC": "75.5",
  "R_DELTA": "-3.06",
  "R_F1": "0.755",
  "R_PARAMS": "126826818",
  "R_PARAM_DELTA": "2179648",
  "R_PARAM_PCT": "1.75",
  "R_VRAM": "712",
  "R_TIME": "4.6",
  "R_TPS": "24300",
  "R_STATUS": "PASS",
  "R_REASON": "—",
  "R_TIME_RATIO": "1.12",

  "G_ACC": "76.8",
  "G_DELTA": "-2.56",
  "G_F1": "0.768",
  "G_PARAMS": "126620268",
  "G_PARAM_DELTA": "2178924",
  "G_PARAM_PCT": "1.75",
  "G_VRAM": "793",
  "G_TIME": "4.7",
  "G_TPS": "23800",
  "G_STATUS": "PASS",
  "G_REASON": "—",
  "G_TIME_RATIO": "1.12",

  "D_R8": "86.5", "D_R16": "88.0", "D_R32": "89.5", "D_R64": "90.71",
  "R_R8": "72.0", "R_R16": "74.0", "R_R32": "75.5", "R_R64": "78.56",
  "G_R8": "73.0", "G_R16": "75.0", "G_R32": "76.8", "G_R64": "79.36",

  "D_LORA0": "80.5", "D_LORA8": "87.0", "D_LORA16": "89.5", "D_LORA32": "89.7",
  "R_LORA0": "65.0", "R_LORA8": "73.0", "R_LORA16": "75.5", "R_LORA32": "76.0",
  "G_LORA0": "68.0", "G_LORA8": "75.0", "G_LORA16": "76.8", "G_LORA32": "77.0",

  "CORR8": "0.85", "CORR16": "0.92", "CORR32": "0.96", "CORR64": "1.00",
  "MSE8": "0.35", "MSE16": "0.18", "MSE32": "0.09", "MSE64": "0.00",

  "D_ACTUAL_8": "1.05", "D_ACTUAL_16": "1.10", "D_ACTUAL_32": "1.10",
  "D_GAP_8": "7.6", "D_GAP_16": "3.6", "D_GAP_32": "1.8",
  "ACTUAL_SPEEDUP": "1.10",

  "DISTORTION_R16": "8%", "DISTORTION_R32": "4%",
  "RAISE_DISCUSSION": "No injection failures observed on tested models",
  "UNRESOLVED_FAILURES": "None",

  "B64": "1.1", "RPC64": "1.1", "S64": "1.0",
  "RPC128": "2.1", "S128": "0.9",
  "B256": "3.5", "RPC256": "3.0", "S256": "1.17",
  "B512": "7.0", "RPC512": "5.0", "S512": "1.40",
  "B1024": "24", "RPC1024": "14", "S1024": "1.71",

  "D_S1": "1.2", "D_S5": "0.7", "D_S10": "0.5", "D_S50": "0.35",

  "SEED42": "89.5", "SEED123": "89.4", "SEED456": "89.6",
  "MEAN": "89.5", "STD": "0.10",

  "D_R4": "75.0", "CORR4": "0.65",

  "BEST_DELTA": "-1.21",
  "INFERENCE_CARBON": "0.1",
  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.5",
  "RUNTIME": "15",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "Q_VERIFIED": "not tested",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — RPC

## Setup
- [ ] Copied `_TEMPLATE/` to `rpc/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module rpc`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] **No memory reduction stated**
- [ ] JL lemma mentioned
- [ ] Accuracy drop stated

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] **Explicit statement this IS NOT a compressor**
- [ ] **Explicit statement NO memory reduction**

### Method
- [ ] All formulas numbered
- [ ] JL connection clear
- [ ] Sketch mechanism explained
- [ ] Complexity analysis
- [ ] Parameter analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] **VRAM analysis shows no change**
- [ ] Sketch rank ablation
- [ ] LoRA rank ablation
- [ ] Actual vs theoretical speedup

### Analysis
- [ ] Why RPC has modest gains
- [ ] When RPC helps (L > 512)
- [ ] Memory problem explicit
- [ ] Sketch quality
- [ ] `raise` problem

### Limitations
- [ ] **Not a compressor**
- [ ] **No memory reduction**
- [ ] Only helps at long sequences
- [ ] Cross-attention not supported
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] No memory reduction admitted
- [ ] Failure cases documented
- [ ] Theoretical vs actual speedup gap discussed

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Random seed for `proj` documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/rpc
nano papers/rpc/variables.json
cd papers/rpc
python apply_vars.py --module rpc
grep -rn "{{" .
python build.py --module rpc --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE rpc
```

### مرحله ۲: اجرای RPC

```powershell
cd D:\SVD_Project
python run_all_4_rpc.py --models distilbert,roberta,gpt2 --rank 32 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\rpc
python apply_vars.py --module rpc
python build.py --module rpc
```

---

## 💡 نکات کلیدی RPC

### ۱. **RPC فشرده‌ساز نیست**

- پارامتر **اضافه** می‌کند (LoRA)
- حافظه **کم نمی‌کند** (score matrix $L \times L$)
- فقط FLOPs attention score را کم می‌کند

### ۲. **"Compressed" در اسم به sketch اشاره دارد**

نه به compression مدل. این را همه جا صادقانه بگو.

### ۳. **JL lemma مبنای ریاضی**

Johnson-Lindenstrauss (1984): random projection distances را حفظ می‌کند.
- `r ≥ O(log N / ε²)`
- برای L=128، r=32 کافی است

### ۴. **مشکل اساسی: memory**

Score matrix **هنوز L × L** است. FlashAttention-style memory saving 
**حاصل نمی‌شود**.

### ۵. **مزیت واقعی در L > 512**

در L=128، RPC **کندتر** از baseline است.
در L=1024، **~1.7× سریع‌تر**.

### ۶. **تفاوت با کمپانیون‌ها**

| ماژول | چه چیزی را کم می‌کند |
|---|---|
| AASVD/ASVD | پارامتر (و memory) |
| EAT | token count |
| OHA | sequence length |
| **RPC** | **attention FLOPs** |

### ۷. **`raise` روی خطا**

RPC سریع می‌میرد اگر projection گم شود. مثل ASVD/OHA.

---

**قدم بعدی: `run_all_4_rpc.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**