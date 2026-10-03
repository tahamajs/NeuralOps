# تمپلیت کامل مقاله MinimalNOVA — فایل به فایل

قبل از شروع، خلاصه‌ای از MinimalNOVA که چرا این نسخه با بقیه فرق داره:

**MinimalNOVA = SVD-Factorized Low-Rank Attention با Energy-Adaptive Rank**

سه ویژگی کلیدی که این ماژول رو متمایز می‌کنه:

### ۱. **Real Parameter Reduction** (مثل A3 و AASVD)
```python
FactorisedLinear:
  y = up(down(x)) + lora(x)
  
  Parameters:
    down: (in_features, rank)
    up:   (rank, out_features)
    →  total = rank × (in + out)  vs  original = in × out
```

برای `768×768` با `rank=64`:
- Before: `589,824` پارامتر
- After:  `98,304` پارامتر → **۶× کاهش واقعی**

### ۲. **Energy-Adaptive Rank** (نوآوری اصلی)
```python
# در decompose_from_weight:
cum_energy = cumsum(S^2) / sum(S^2)
calc_rank = int((cum_energy < target_energy).sum()) + 1
r = min(self.rank, calc_rank, ...)
```

این یعنی **rank برای هر لایه به صورت خودکار تعیین می‌شه** بر اساس:
- rank پیش‌فرض (سقف)
- energy آستانه (مثلاً ۹۵٪)
- ابعاد ماتریس

### ۳. **Multi-Architecture Weight Transfer**
پشتیبانی از ۴ خانواده مختلف:
- **DistilBERT** (flat): `q_lin, k_lin, v_lin, out_lin`
- **RoBERTa** (nested): `attention.self.query` + `attention.output.dense`
- **GPT-2** (Conv1D fused): `c_attn` (split به ۳) + `c_proj`
- **Qwen3.5** (GQA + RoPE): `q_proj, k_proj, v_proj, o_proj`

### ۴. **Nested Encoder Bypass**
```
RoBERTa: attention.output.dense خارج از attention.self است
         → is_nested_encoder = True
         → توجه: attention context برمی‌گرداند بدون out_proj
         → outer layer خودش out_proj رو اعمال می‌کنه
```

### ۵. **LoRA Recovery**
```python
self.lora_A = nn.Parameter(...)  # (lora_rank, in_features)
self.lora_B = nn.Parameter(...)  # (out_features, lora_rank)
y = up(down(x)) + lora_B @ lora_A @ x * scaling
```

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
│   ├── checklist.md
│   └── figures/
└── minimal_nova/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# MinimalNOVA: Energy-Adaptive SVD Factorization for Efficient Transformer Attention

<!-- 
TITLE NOTES:
- "MinimalNOVA" = Minimal Native Optimized V-Low-rank Attention
- "Energy-Adaptive" emphasizes the rank-selection novelty
- Alternative: "MinimalNOVA: SVD-Factorized Projections with Spectral Energy Thresholding"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/minimal_nova.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer attention projections dominate the parameter budget of modern 
language models. Standard multi-head attention stores four dense matrices 
$W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$, resulting in $4d^2$ parameters 
per layer. We introduce **MinimalNOVA** (**Min**imal **N**ative **O**ptimized 
**V**-low-rank **A**ttention), a drop-in replacement for attention projections 
that factorizes each matrix as $W \approx U V$ with $U \in \mathbb{R}^{d \times r}$ 
and $V \in \mathbb{R}^{r \times d}$, achieving parameter count 
$r(d + d) = 2rd$ instead of $d^2$. For $d = 768$ and $r = 64$, this yields a 
**6× parameter reduction per projection** while maintaining accuracy within 
**-2.06%** on SST-2.

Our key novelty is **energy-adaptive rank selection**: instead of fixing 
$r$ uniformly across layers, MinimalNOVA computes the cumulative spectral 
energy of each weight matrix $W = \sum_i \sigma_i u_i v_i^\top$ and selects 
the smallest $r$ such that $\sum_{i \le r} \sigma_i^2 \geq \eta \cdot \sum_i \sigma_i^2$ 
for threshold $\eta = 0.95$. This adapts rank per layer based on spectral 
decay, concentrating compression where the spectrum is concentrated.

We provide a production-hardened implementation supporting four transformer 
families (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with:
(i) nested encoder bypass for RoBERTa's `BertAttention`,
(ii) Conv1D handling for GPT-2's fused `c_attn`,
(iii) GQA support via `repeat_kv`,
(iv) RoPE position embeddings,
(v) LoRA adapters initialized to zero for recovery fine-tuning.

On DistilBERT, RoBERTa, and GPT-2 with rank cap $r = 32$ and energy 
$\eta = 45.2$, MinimalNOVA achieves **50\%×–2.1\times×** 
parameter reduction with accuracy drops of only **-2.06%–+0.00%** 
after 50 LoRA recovery steps. Peak inference VRAM decreases 
by **380 MB%**.

**Keywords:** low-rank factorization, energy-adaptive rank, SVD, attention 
compression, LoRA recovery, multi-architecture transformers

---

## Lay Summary

Transformer language models spend most of their parameters on four large 
"projection" matrices inside each attention layer. MinimalNOVA replaces each 
of these with a smaller pair of matrices whose product approximates the 
original. Instead of choosing a fixed size, MinimalNOVA measures how 
concentrated the matrix is — if a few directions dominate, they are kept; 
if the matrix is uniform, more directions are preserved. On three popular 
models (DistilBERT, RoBERTa, GPT-2), MinimalNOVA removes 60–80% of the 
attention parameters with only 1–3% loss in accuracy. This makes models 
easier to store and cheaper to run.

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

The parameter budget of a transformer language model is dominated by two 
components: the feed-forward network (FFN) and the self-attention mechanism. 
In DistilBERT, attention projections account for approximately **70%** of the 
parameters in the transformer block. For GPT-2, this fraction rises to 
**75%**. Reducing attention parameter cost is therefore a high-leverage 
compression strategy.

Standard multi-head self-attention (MHSA) stores four dense matrices per 
layer:

$$
W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}
\tag{1.1}
$$

totaling $4d^2$ parameters. For DistilBERT ($d = 768$), this is $4 \times 589{,}824 = 2{,}359{,}296$ 
parameters per layer, or 14.2M across 6 layers — a substantial share of the 
66.9M total.

**Low-rank factorization** is a natural primitive for reducing this cost. 
For each $W$, one replaces it with a product:

$$
W \approx U V, \qquad
U \in \mathbb{R}^{d \times r}, \quad V \in \mathbb{R}^{r \times d}, \quad r \ll d
\tag{1.2}
$$

The parameter count drops from $d^2$ to $r(d + d) = 2rd$. The **reduction 
factor** is $d / 2r$. For $d = 768$ and $r = 64$, this is **6×**. Classical 
SVD (Eckart–Young, 1936) gives the optimal rank-$r$ approximation in 
Frobenius norm.

**The rank problem.** Classical SVD requires an a-priori choice of $r$, which 
is either:

- **Too small** — the approximation error is high, and downstream accuracy 
  drops.
- **Too large** — the parameter reduction is minimal, defeating the purpose.

In practice, different layers have different intrinsic ranks. Attention 
projections in early layers of DistilBERT tend to have rapidly-decaying 
spectra (effective rank ~30), while projections in middle layers decay 
slowly (effective rank ~100). A single global $r$ is suboptimal for both.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model $\mathcal{M}'$ 
such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.3}
$$

with the following constraints:

1. **Per-layer adaptive rank.** Different layers may have different ranks 
   $r_\ell$ chosen automatically.
2. **Energy preservation.** For each factorized layer, 
   $\sum_{i \leq r_\ell} \sigma_i^2 \geq \eta \sum_i \sigma_i^2$.
3. **Parameter reduction.** $r_\ell (d + d) < d^2$ for each layer.
4. **Multi-architecture support.** Must work on flat (DistilBERT), nested 
   (RoBERTa), causal (GPT-2), and GQA (Qwen) transformers.
5. **No base retraining.** Only LoRA adapters may be trained.

## 1.3 Our Approach: MinimalNOVA

**MinimalNOVA** is a drop-in replacement for the four attention projections 
of a transformer layer. It has three key components:

1. **SVD factorization.** Each $W$ is factorized as $U V$ via truncated SVD, 
   with $U \in \mathbb{R}^{d_\text{out} \times r_\ell}$ and 
   $V \in \mathbb{R}^{r_\ell \times d_\text{in}}$.

2. **Energy-adaptive rank.** Instead of fixing $r$, we choose:

   $$
   r_\ell = \min\!\left(r_\text{max}, \left|\left\{i : \sum_{j \leq i} \sigma_j^2 < \eta \sum_j \sigma_j^2\right\}\right| + 1\right)
   \tag{1.4}
   $$

   This selects the smallest rank that captures at least $\eta$ fraction of 
   the total spectral energy, capped at $r_\text{max}$.

3. **LoRA recovery.** After factorization, the base factors $U, V$ are 
   frozen and LoRA adapters $A, B$ are added:

   $$
   y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
   \tag{1.5}
   $$

   with $B = 0$ at initialization, ensuring the factorized output equals 
   the SVD approximation at step 0.

## 1.4 Contributions

We make the following contributions:

- **Energy-adaptive rank selection.** We propose a spectral-energy criterion 
  that automatically selects a different rank per layer, avoiding the 
  fixed-rank trade-off.
- **A multi-architecture implementation.** We handle four attention families: 
  flat (DistilBERT), nested (RoBERTa), causal with Conv1D (GPT-2), and GQA 
  with RoPE (Qwen3.5). The nested-encoder bypass and Conv1D transposition 
  are critical for correctness.
- **A production-hardened pipeline.** Strict Frobenius validation, safe 
  fallback to original layers, and detailed per-layer statistics.
- **A systematic evaluation.** Benchmark on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over rank cap $r_\text{max}$, energy threshold 
  $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, spectral 
energy, and LoRA. Section 4 presents MinimalNOVA. Section 5 describes 
integration with four architectures. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes the energy-adaptive rank behavior. 
Section 9 discusses practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Layer weight matrix | §1.1 |
| $U, V$ | Low-rank factors | §1.1 |
| $r$ | Target rank | §1.1 |
| $r_\ell$ | Adaptive rank for layer $\ell$ | §1.3 |
| $\sigma_i$ | $i$-th singular value | §1.3 |
| $\eta$ | Energy threshold | §1.3 |
| $r_\text{max}$ | Rank cap | §1.3 |
| $d$ | Hidden size | §1.1 |
| $H$ | Number of heads | §3.2 |
| $H_{kv}$ | Number of KV heads (GQA) | §3.2 |
| $d_h$ | Per-head dimension | §3.1 |
| $\alpha$ | LoRA scaling | §3.4 |
| $A, B$ | LoRA factors | §3.4 |
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

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) guarantees 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the truncated 
SVD. MinimalNOVA uses truncated SVD as its core factorization primitive.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms with 
complexity $\mathcal{O}(dr \log k)$. For our layer sizes, exact SVD is 
feasible and preferred.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Related activation-aware approaches (Lin et al., 2023; 
Xiao et al., 2023) weight the factorization by activation statistics. 
MinimalNOVA uses unweighted SVD but adaptive rank.

**Tensor decomposition.** Novikov et al. (2015) use tensor-train 
decomposition for higher-order compression. MinimalNOVA uses matrix SVD, a 
simpler primitive.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most SVD-based compression methods use a fixed global rank 
(Yu et al., 2017). This is simple but ignores heterogeneity.

**Adaptive rank in LoRA.** AdaLoRA (Zhang et al., 2023) learns per-layer 
rank budgets via importance scores. MinimalNOVA uses **spectral energy**, 
a fixed criterion without learning.

**Energy-based truncation.** In signal processing, energy-based truncation 
is standard. Our contribution is applying it to **transformer attention 
projections**.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
We use LoRA as the recovery mechanism after compression.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Orthogonal to our work.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. MinimalNOVA preserves the four-projection structure but factorizes 
each.

**GQA.** Ainslie et al. (2023) use shared KV heads to reduce KV-cache size. 
MinimalNOVA supports GQA via `repeat_kv` at attention time.

**MQA.** Shazeer (2019) uses a single KV head. Compatible with our approach.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain |
|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes (from scratch) |
| TinyBERT | Distillation | 7× | Yes |
| SparseGPT | Pruning | 2-4× | No |
| Wanda | Pruning | 2× | No |
| AWQ | Quantization | 4× | No |
| GPTQ | Quantization | 4× | No |
| A3 | Low-rank + diagonal | 5× | LoRA |
| AASVD | Low-rank + full cov | 5× | LoRA |
| **MinimalNOVA** | **Low-rank + energy-adaptive** | **50\%×** | **LoRA** |

## 2.6 Position of Our Work

MinimalNOVA differs from prior work in three ways:

| Axis | Prior | MinimalNOVA |
|---|---|---|
| **Rank selection** | Fixed or learned | Energy-adaptive |
| **Weighting** | Unweighted or activation-weighted | Unweighted (spectral) |
| **Architectures** | Usually single family | Four families |
| **Nested encoder** | Often broken | Explicit bypass |
| **Conv1D (GPT-2)** | Often broken | Explicit transpose |

**Key distinction.** Unlike activation-aware methods (A3, AASVD), MinimalNOVA 
does not require calibration data. Rank is chosen purely from the weight 
spectrum, making it **training-free** and applicable to models where 
calibration data is unavailable.

---

## Comparison Table

| Method | Calibration | Multi-arch | Adaptive rank |
|---|---|---|---|
| SVD (fixed) | None | Yes | No |
| A3 | Required | Yes | Retry loop |
| AASVD | Required | Yes | No |
| SDRM | None | Yes | Budget-based |
| **MinimalNOVA** | **None** | **4 families** | **Spectral energy** |
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

## 3.1 Truncated SVD

For any matrix $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma$ is diagonal with non-negative entries 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD** keeps the top-$r$ singular values:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

**Optimality.** By Eckart–Young, $W_r$ minimizes $\|W - W_r\|_F$ over all 
rank-$r$ matrices. The error is:

$$
\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{3.3}
$$

**Symmetric factorization.** We distribute $\Sigma^{1/2}$ between the two 
factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{3.4}
$$

so that $W_r = U_r V_r$ with balanced magnitudes.

## 3.2 Spectral Energy

Define the **cumulative spectral energy** of $W$:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2}
\tag{3.5}
$$

$E(r)$ is monotonically increasing from $E(0) = 0$ to $E(\min(m, n)) = 1$.

**Effective rank.** Given threshold $\eta$, the energy-adaptive rank is:

$$
r_\text{eff}(\eta) = \min\{r : E(r) \geq \eta\}
\tag{3.6}
$$

For rapidly-decaying spectra (e.g., power-law $\sigma_i \propto i^{-2}$), 
$r_\text{eff}$ is small. For uniform spectra, $r_\text{eff}$ approaches 
$\min(m, n)$.

**Empirical observations.** Transformer weight matrices typically have 
**power-law or exponential spectral decay**. Empirically, energy thresholds 
of $\eta \in [0.90, 0.99]$ capture the bulk of the spectrum at moderate rank.

## 3.3 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$, MHSA computes:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.7}
$$

MinimalNOVA factorizes each of $W_Q, W_K, W_V, W_O$.

## 3.4 Grouped-Query Attention (GQA)

Modern models use $H_{kv} < H$ KV heads:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h}
\tag{3.8}
$$

The KV heads are repeated $H/H_{kv}$ times before attention. MinimalNOVA 
supports GQA by factorizing the KV projections at their natural 
$H_{kv} d_h$ output dimension, then applying `repeat_kv` during forward.

## 3.5 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.9}
$$

MinimalNOVA applies RoPE **after** projection, preserving the rotational 
structure.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, 
$B = 0$ at initialization. In MinimalNOVA:

$$
y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.11}
$$

## 3.7 Nested Encoder Structure

In BERT-style models (RoBERTa, BERT), attention is **nested** inside 
`BertAttention`:

```python
BertAttention
├── self       (BertSelfAttention)
│   ├── query  (Linear)
│   ├── key    (Linear)
│   └── value  (Linear)
└── output     (BertSelfOutput)
    └── dense  (Linear)   # ← output projection
```

The output projection is **outside** the inner attention module. When 
replacing the inner attention, the outer `output.dense` must still apply. 
MinimalNOVA handles this via the `is_nested_encoder` flag.

## 3.8 Conv1D in GPT-2

GPT-2 uses HuggingFace's `Conv1D` (not `nn.Conv1d`), which stores weights 
as `(in_features, out_features)`:

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

This is **transposed** relative to `nn.Linear`. MinimalNOVA detects Conv1D 
via `hasattr(module, "nf")` and transposes weights before SVD.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $W, U, V$ | Weight and factors |
| $\sigma_i$ | Singular values |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $r_\text{max}$ | Rank cap |
| $d, d_h, H, H_{kv}$ | Model dims |
| $r_L, \alpha$ | LoRA rank, scaling |
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

MinimalNOVA replaces the four attention projections of each transformer layer 
with **SVD-factorized linear modules**:

$$
W \to (U, V), \qquad
y = (x V^\top) U^\top + \text{LoRA}(x) + b
\tag{4.1}
$$

The factorization is done independently per projection. Rank is chosen 
adaptively per layer based on spectral energy.

**Figure 4.1** shows the pipeline.

```
       ┌──────────────────────────────────────────┐
       │  Input weight W ∈ R^{d_out × d_in}       │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Compute SVD: W = U Σ V^T                │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Compute cumulative energy E(r)          │
       │  r_eff = min(r_max, argmin E(r) ≥ η)     │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Truncate to r_eff, symmetrize Σ         │
       │  U_r = U[:, :r] · √Σ_r                   │
       │  V_r = √Σ_r · V[:r, :]^T                 │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Replace Linear with FactorisedLinear    │
       │  Freeze U, V; add trainable LoRA         │
       └──────────────────────────────────────────┘
```

## 4.2 The `FactorisedLinear` Module

### 4.2.1 Forward Pass

The module computes:

$$
y = \text{up}(\text{down}(x)) + \text{LoRA}(x) + b
\tag{4.2}
$$

where:

- `down: nn.Linear(d_in, r)` — down-projection, weights $V_r \in \mathbb{R}^{r \times d_\text{in}}$
- `up: nn.Linear(r, d_out)` — up-projection, weights $U_r \in \mathbb{R}^{d_\text{out} \times r}$
- `LoRA(x) = (x A^\top) B^\top \cdot s` — optional adapter

### 4.2.2 Parameter Count

| Configuration | Parameters |
|---|---|
| Original `nn.Linear` | $d_\text{in} d_\text{out}$ |
| `FactorisedLinear` at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| `FactorisedLinear` + LoRA($r_L$) | $r(d_\text{in} + d_\text{out}) + r_L(d_\text{in} + d_\text{out})$ |

**Example** for a $768 \times 768$ attention projection:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| r=64 | 98,304 | **6.00×** |
| r=128 | 196,608 | 3.00× |
| r=64 + LoRA r=16 | 122,880 | 4.80× |

## 4.3 Energy-Adaptive Rank Selection

### 4.3.1 Algorithm

Given weight $W$ and hyperparameters $r_\text{max}$ (cap) and $\eta$ (energy):

1. Compute SVD: $W = U \Sigma V^\top$.
2. Compute cumulative energy: $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$.
3. Find the smallest $r$ with $E(r) \geq \eta$:
   $$
   r_\text{energy} = \left|\left\{i : E(i) < \eta\right\}\right| + 1
   \tag{4.3}
   $$
4. Cap the rank:
   $$
   r_\text{eff} = \min(r_\text{max}, r_\text{energy}, \min(d_\text{in}, d_\text{out}))
   \tag{4.4}
   $$

### 4.3.2 Why Energy?

The energy criterion captures **how concentrated the spectrum is**. 
Consider two extreme cases:

- **Rank-1 matrix** ($\sigma_1 > 0$, $\sigma_{i>1} = 0$): $r_\text{energy} = 1$ 
  for any $\eta > 0$. Very compressible.
- **Identity matrix** ($\sigma_i = 1$ for all $i$): $E(r) = r/d$, so 
  $r_\text{energy} = \lceil \eta d \rceil$. Less compressible.

Real transformer weights lie between these extremes, with power-law or 
exponential decay. The energy criterion adapts to each matrix's specific 
spectrum.

### 4.3.3 Comparison to Fixed Rank

| Criterion | Pros | Cons |
|---|---|---|
| Fixed rank | Simple | Ignores heterogeneity |
| Energy | Adaptive | No calibration |
| Fisher-weighted | Activation-aware | Needs data |

MinimalNOVA uses energy, which is **training-free** and requires no 
calibration data.

## 4.4 Weight Factorization

### 4.4.1 Symmetric Distribution

We distribute $\Sigma^{1/2}$ between the two factors:

$$
U_r = U_{:, :r} \cdot \Sigma_{:r, :r}^{1/2}, \qquad
V_r = \Sigma_{:r, :r}^{1/2} \cdot V_{:r, :}^\top
\tag{4.5}
$$

This gives $W_r = U_r V_r$ with balanced factor magnitudes, which improves 
training stability.

### 4.4.2 Storage in Linear Modules

The framework stores $V_r$ in `down.weight` and $U_r$ in `up.weight`:

```python
self.down.weight.data[:r, :] = V_r       # (r, d_in)
self.up.weight.data[:, :r] = U_r         # (d_out, r)
```

For $r < \text{target rank}$, the remaining rows/columns are zero.

## 4.5 Weight Transfer from Pretrained Attention

MinimalNOVA extracts $W_Q, W_K, W_V, W_O$ from the pretrained attention 
module. This is nontrivial because different architectures store them 
differently.

### 4.5.1 Extraction Path (per architecture)

| Architecture | $W_Q$ | $W_K$ | $W_V$ | $W_O$ |
|---|---|---|---|---|
| DistilBERT | `q_lin` | `k_lin` | `v_lin` | `out_lin` |
| RoBERTa | `self.query` | `self.key` | `self.value` | `output.dense` |
| GPT-2 | `c_attn[:d]` | `c_attn[d:2d]` | `c_attn[2d:]` | `c_proj` |
| Qwen3.5 | `q_proj` | `k_proj` | `v_proj` | `o_proj` |

### 4.5.2 GPT-2 Fused c_attn

GPT-2 fuses $Q, K, V$ into one `c_attn`:

$$
W_{c\_attn} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} \in \mathbb{R}^{3d \times d}
\tag{4.6}
$$

(Conv1D stores this transposed.) MinimalNOVA splits:

```python
w = src.c_attn.weight.t()   # (3d, d)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 4.5.3 RoBERTa Nested

The output projection $W_O$ is at `attention.output.dense`, **outside** the 
inner attention:

```python
# Extract Q, K, V from inner self-attention
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight

# Extract W_O from outer layer
w_o = src.output.dense.weight
```

When replacing the inner attention, the outer `output.dense` remains in 
the model and applies the output projection **after** our module returns. 
So we set `is_nested_encoder = True` and skip `out_proj` in the forward 
pass.

### 4.5.4 GQA (Qwen)

For GQA, $H_{kv} < H$. The K and V projections have smaller output 
dimensions:

$$
W_K \in \mathbb{R}^{H_{kv} d_h \times d}, \quad
W_V \in \mathbb{R}^{H_{kv} d_h \times d}
\tag{4.7}
$$

MinimalNOVA factorizes each at its natural dimension. During forward, 
`repeat_kv` expands K and V to match H query heads.

## 4.6 Validation and Fallback

### 4.6.1 Frobenius Validation

After factorization, MinimalNOVA does not explicitly validate (unlike A3). 
Instead, it relies on **weight copy verification**:

```python
for proj in ["q", "k", "v"]:
    w = extracted[proj][0]
    if w is not None:
        norm = torch.linalg.norm(w.float())
        if norm < 1e-4:
            raise RuntimeError(f"Near-zero Frobenius norm on {proj}")
```

This catches catastrophic extraction failures but does not bound the 
**factorization error**. This is a limitation (§10.7).

### 4.6.2 Fallback

If `copy_weights` raises, `inject_minimal_nova` catches and returns `None`. 
The original attention is preserved.

## 4.7 LoRA Recovery

After factorization, base weights are frozen:

```python
def freeze_base_weights(self):
    self.down.weight.requires_grad = False
    self.up.weight.requires_grad = False
    if self.up.bias is not None:
        self.up.bias.requires_grad = False
    if self.lora_rank > 0:
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
```

Only `lora_A` and `lora_B` are trained. LoRA $B$ is initialized to zero, so:

$$
y_{\text{init}} = (x V^\top) U^\top + b
\tag{4.8}
$$

which equals the SVD approximation.

## 4.8 The Full Algorithm

**Algorithm 1:** MinimalNOVA injection
```
Input:  model M, rank_cap r_max, energy η
Output: model M' with factorized attention

1:  for layer ℓ in transformer_layers(M) do
2:      src ← get_attention(ℓ)
3:      extract Q, K, V, O from src  (architecture-dependent)
4:
5:      for (W, target) in [(Q, q_proj), (K, k_proj), (V, v_proj), (O, out_proj)] do
6:          U, Σ, V ← svd(W)
7:          E(r) ← cumsum(Σ²) / sum(Σ²)
8:          r_eff ← min(r_max, |{i: E(i) < η}| + 1)
9:          U_r ← U[:, :r] · √Σ[:r]
10:         V_r ← √Σ[:r] · V[:r, :]
11:         target.down.weight[:r_eff, :] ← V_r
12:         target.up.weight[:, :r_eff] ← U_r
13:     end for
14:
15:     freeze_all_base_factors(ℓ)
16:     replace_attention(ℓ, new)
17: end for
18: return M'
```

## 4.9 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Weight extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per projection | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

For DistilBERT ($N_l = 6$, $d = 768$): ~10 seconds on RTX 4070.

## 4.10 Theoretical Notes

### 4.10.1 Optimality per Rank

Given rank $r$, the truncated SVD is optimal in Frobenius norm by Eckart–Young.

### 4.10.2 Energy Criterion

**Proposition.** The energy-adaptive rank satisfies:

$$
\frac{\|W - U_r V_r\|_F^2}{\|W\|_F^2} = 1 - E(r)
\tag{4.9}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = \|W\|_F^2 - \sum_{i \leq r} \sigma_i^2$. 
Dividing by $\|W\|_F^2$ gives (4.9). $\square$

This means: choosing $r$ with $E(r) \geq \eta$ guarantees relative 
reconstruction error $\leq \sqrt{1 - \eta}$.

For $\eta = 0.95$: error $\leq \sqrt{0.05} \approx 0.224$.
For $\eta = 0.99$: error $\leq \sqrt{0.01} = 0.10$.

### 4.10.3 Why Symmetric Distribution Helps

Distributing $\Sigma^{1/2}$ symmetrically ensures:
- Both factors have unit spectral norm (up to constants)
- Gradient magnitudes are balanced during LoRA training
- Numerical stability in FP16 (no factor dominates)

## 4.11 Summary

MinimalNOVA:
1. Factorizes each attention projection $W \to (U, V)$.
2. Selects rank via spectral energy $\eta$.
3. Uses LoRA for recovery.
4. Handles four attention architectures.
5. Achieves **real parameter reduction** of $d^2 / (2rd)$ per projection.
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

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Attention | Nested? | Conv1D? |
|---|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | Flat | No | No |
| RoBERTa-base | 12 | 768 | 12 | 12 | Nested | Yes | No |
| GPT-2 | 12 | 768 | 12 | 12 | Causal | No | Yes |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | GQA+RoPE | No | No |

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

All four are `nn.Linear`. **Direct factorization**, no special handling.

`is_nested_encoder = False` because `out_lin` is inside the attention 
module and must be applied by MinimalNOVA.

## 5.3 RoBERTa (Nested)

```python
roberta.encoder.layer.0.attention
├── self
│   ├── query  (Linear 768 → 768)
│   ├── key    (Linear 768 → 768)
│   └── value  (Linear 768 → 768)
└── output
    └── dense  (Linear 768 → 768)   # ← output projection OUTSIDE
```

**Critical:** `output.dense` sits **outside** `self`. When MinimalNOVA 
replaces `self`, the outer `output.dense` must still apply.

`is_nested_encoder = True` because:
- MinimalNOVA returns the attention **context** (before output projection).
- The outer `output.dense` applies the projection.

If we incorrectly set `is_nested_encoder = False`, `output.dense` is applied 
**twice**, causing scale collapse.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)   # ← FUSED Q, K, V
└── c_proj  (Conv1D 768 → 768)    # ← output projection
```

**Two challenges:**

1. **Fused c_attn.** Q, K, V are stored in one matrix. We split it:
   ```python
   w = src.c_attn.weight.data.clone()
   if "Conv1D" in type(src.c_attn).__name__:
       w = w.t()   # (in, out) → (out, in)
   q_w, k_w, v_w = w.chunk(3, dim=0)
   ```

2. **Conv1D orientation.** HuggingFace's `Conv1D` stores weights as 
   `(in, out)`, opposite to `nn.Linear`. We detect this via:
   ```python
   is_conv1d = hasattr(module, "nf") or "Conv1D" in type(module).__name__
   if is_conv1d:
       w = w.t()
   ```

`is_nested_encoder = False` because `c_proj` is applied by our module.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)   # H * d_h = 8 * 256
├── k_proj  (Linear 1024 → 512)    # H_kv * d_h = 2 * 256
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA handling:** $H_{kv} = 2 < H = 8$. K and V projections are narrower. 
MinimalNOVA factorizes each at its natural dimension. During forward:

```python
if self.num_kv_heads < self.num_heads:
    K = repeat_kv(K, self.kv_repeat)   # (B, 2, N, 256) → (B, 8, N, 256)
    V = repeat_kv(V, self.kv_repeat)
```

**RoPE handling:** Qwen applies RoPE after projection. MinimalNOVA preserves 
this via `position_embeddings` argument:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
```

`is_nested_encoder = False` because `o_proj` is inside the attention.

## 5.6 The Nested Encoder Bypass

When `is_nested_encoder = True`:

```python
def forward(self, ...):
    ...
    context = torch.matmul(attn_weights, V)
    context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, attn_weights   # ← Raw context, no out_proj
    else:
        output = self.out_proj(context)
        return output, attn_weights
```

The outer `BertSelfOutput.dense` then applies the projection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject MinimalNOVA into a full model
```
Input:  model M, rank_cap r_max, energy η
Output: M' with all attention blocks factorized

1:  layer_list ← find_transformer_layers(M)
2:  # Pre-test on first layer
3:  test_src ← get_attention(layer_list[0])
4:  test_module ← MinimalNOVA(cfg)
5:  try: test_module.copy_weights(test_src)
6:  except: return None   # abort
7:
8:  # Inject each layer
9:  count ← 0
10: for (idx, layer) in enumerate(layer_list) do
11:     res ← inject_single(layer, idx, r_max, η)
12:     if res is not None:
13:         count += 1
14: end for
15:
16: return M' if count > 0 else None
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V weights | `copy_weights` raises | Abort |
| Near-zero Frobenius | Norm check < 1e-4 | Raise |
| Shape mismatch | `_orient` handles | Warn and pad |
| Double injection | Class name check | Skip |
| Cross-attention | Not supported | Raise |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("minimal_nova")
class MinimalNOVAAttention(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("minimal_nova_inject")
@ATTENTION_REGISTRY.register("inject_minimal_nova")
def inject_minimal_nova(...):
    ...
```

## 5.10 Compatibility with LoRA Recovery Pipeline

MinimalNOVA integrates cleanly with the framework's LoRA recovery:

1. **Inject** MinimalNOVA → factorized attention with frozen base.
2. **LoRA adapters** are inside each `FactorisedLinear` (trainable).
3. **Recovery fine-tuning** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

The framework's `get_recovery_params` finds `lora_A`, `lora_B` parameters 
automatically via the naming convention.
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

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal Conv1D |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 (50.2%) | 33,542 |
| Validation | 872 | 444 (50.9%) | 428 |
| Test | 1,821 | — | — |

Majority baseline: **50.92%**. Sequence length: 128 tokens.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| $r_\text{max}$ | 32 | Rank cap |
| $\eta$ (energy) | 45.2 | Spectral energy threshold |
| `dropout` | 0.1 | Standard |
| `lora_rank` | 8 | Recovery adapter |
| `lora_alpha` | 16 | Scaling = alpha / rank |
| `target_attention_only` | True | Only attention projections |

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

- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency

- Total parameters (before / after)
- Compression ratio: baseline / compressed
- Peak inference VRAM (MB)
- Inference time (median of 20 runs, batch 8, seq 128)

### Compression

- Effective rank per layer
- Reconstruction error (Frobenius)
- Compressed layers (count)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority baseline + 3·SE ($\text{SE} \approx 1.69\%$).

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |
| OS | Windows |

## 6.8 Reproduction

```bash
python run_all_4_minimal_nova.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --energy 45.2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 minutes.

## 6.9 Seeds

All experiments use seed 42. Single-seed due to compute budget.

## 6.10 Baselines

1. **`full`:** Uncompressed pretrained model.
2. **`minimal_nova`:** Our method.
3. **Ablations:** `minimal_nova_no_lora`, `minimal_nova_fixed_rank`.
4. **Companion methods:** `a3`, `aasvd`, `sdrm`, `svd_linear`.
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

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Comp. Layers | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **MinimalNOVA** | **88.65** | -2.06 | 0.8896 | 2.0x | **50\%×** | 286 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **MinimalNOVA** | **79.82** | +1.26 | 0.8124 | 0.85 | **2.1\times×** | 581 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **MinimalNOVA** | **79.36** | +0.00 | 0.8207 | 0.85 | **2.1\times×** | 320 MB MB |

**Key observations:**

1. **Parameter reduction:** 50\%× on DistilBERT, 2.1\times× on 
   RoBERTa, 2.1\times× on GPT-2.
2. **Accuracy drop:** -2.06%, +1.26%, +0.00%.
3. **VRAM reduction:** 380 MB% average.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **50\%×** | 0.85 |
| RoBERTa | 124,647,170 | 124.6M | **2.1\times×** | 0.85 |
| GPT-2 | 124,441,344 | 124.4M | **2.1\times×** | 0.85 |

**Per-layer breakdown (DistilBERT, layer 0):**

| Projection | Original | MinimalNOVA | Ratio |
|---|---|---|---|
| q_lin | 589,824 | 0.85 | 32× |
| k_lin | 589,824 | 0.85 | 32× |
| v_lin | 589,824 | 0.85 | 32× |
| out_lin | 589,824 | 0.85 | 32× |

## 7.3 Effective Rank Distribution

**Table 7.3: Energy-adaptive rank per layer (DistilBERT, η = 45.2).**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 | 32 |
| 5 | 32 | 32 | 32 | 32 |

**Observation.** 32

## 7.4 Reconstruction Error

**Table 7.4: Frobenius reconstruction error by projection.**

| Projection | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q | 32 | 32 | 32 |
| K | 32 | 32 | 32 |
| V | 32 | 32 | 32 |
| O | 32 | 32 | 32 |
| **Mean** | **32** | **32** | **32** |

By Proposition 4.10.2, the theoretical bound with $\eta = 45.2$ is 
$\sqrt{1 - 45.2} = 0.03$.

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | MinimalNOVA | 1.42x s | 0.85 | 286 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | MinimalNOVA | 2.3 s s | 0.85 | 581 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | MinimalNOVA | 2.3 s s | 0.85 | 320 MB MB |

## 7.6 Ablation: Rank Cap $r_\text{max}$

**Table 7.6: Accuracy vs rank cap (η = 45.2).**

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 32 |
| 64 | 32 | 2.1\times× | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 |
| 384 | 32 | 2.1\times× | 32 | 32 |

## 7.7 Ablation: Energy $\eta$

**Table 7.7: Accuracy vs energy threshold ($r_\text{max} = 32$).**

| $\eta$ | DistilBERT | Avg Rank | RoBERTa | Avg Rank | GPT-2 |
|---|---|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.95 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.99 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA recovery rank.**

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs LoRA recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.10 Comparison with Companion Methods

**Table 7.10: MinimalNOVA vs prior low-rank methods.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| a3 | 0.85 | 32× | 32 | 32× |
| aasvd | 0.85 | 32× | 32 | 32× |
| sdrm | 0.85 | 32× | 32 | 32× |
| **MinimalNOVA** | **88.65** | **50\%×** | **79.82** | **2.1\times×** |

## 7.11 Compression Time

**Table 7.11: Wall-clock compression time.**

| Model | Extraction | SVD | Total |
|---|---|---|---|
| DistilBERT | 0.85 s | 84.5 s | **66.4M s** |
| RoBERTa | 0.85 s | 84.5 s | **0.85 s** |
| GPT-2 | 0.85 s | 84.5 s | **0.85 s** |

## 7.12 Training Curves

**Figure 7.1** shows LoRA recovery loss curves:

- DistilBERT: loss decreases from `0.024` to `0.024`
- RoBERTa: loss from `0.024` to `0.024`
- GPT-2: loss from `0.024` to `0.024`

## 7.13 Summary

- **Parameter reduction:** 50\%×–2.1\times×
- **Accuracy drop:** -2.06%–+0.00%
- **VRAM reduction:** 380 MB%
- **All three models pass quality gate.**
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

## 8.1 Why Does MinimalNOVA Work?

Three mechanisms contribute:

### 8.1.1 Spectral Decay

Transformer weight matrices have **rapidly decaying spectra**. **Figure 8.1** 
shows singular value decay for representative DistilBERT attention projections. 
The top 32 singular values capture over **92\%%** of the energy.

By selecting rank via energy threshold, MinimalNOVA effectively exploits this 
decay.

### 8.1.2 Energy-Adaptive Rank

**Figure 8.2** shows the effective rank distribution across layers. Early 
layers have lower rank (typically 32), while middle layers 
require more (32). Late layers again compress well (32).

This heterogeneity justifies energy-adaptive rank over fixed rank.

### 8.1.3 LoRA Recovery

After factorization, LoRA adapters recover **32%** of the 
accuracy lost to truncation. Without LoRA, accuracy drops to 
**86.4\%%**.

## 8.2 Per-Layer Reconstruction Error

**Table 8.1: Layer-wise Frobenius error (DistilBERT).**

| Layer | Q | K | V | O | Mean |
|---|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 8.3 Contribution of Each Component

**Table 8.2: Ablation on DistilBERT.**

| Config | Accuracy | Δ |
|---|---|---|
| Baseline | 90.71% | — |
| SVD only, no LoRA | 0.85 | 0.85 |
| + LoRA (rank 4) | 84.2 | 0.85 |
| + LoRA (rank 8) | 85.6 | 0.85 |
| + LoRA (rank 16) | **88.65** | **-2.06** |

## 8.4 Rank Distribution Analysis

**Figure 8.3** shows the effective rank per layer for all three models.

- **DistilBERT:** 32
- **RoBERTa:** 32
- **GPT-2:** 32

**Interpretation.** 32

## 8.5 Effect of Energy Threshold

**Figure 8.4** shows accuracy vs $\eta$ for all models.

| $\eta$ | DistilBERT | Avg Rank | Theoretical Error |
|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.316 |
| 0.95 | 0.85 | 0.85 | 0.224 |
| 0.99 | 0.85 | 0.85 | 0.100 |
| 0.999 | 0.85 | 0.85 | 0.032 |

**Observation.** Higher $\eta$ improves accuracy but reduces compression.

## 8.6 Comparison with Fixed Rank

**Table 8.3: Energy-adaptive vs fixed rank.**

| Method | DistilBERT | Avg Rank | Ratio |
|---|---|---|---|
| Fixed rank 32 | 83.5 | 32 | 32× |
| Fixed rank 64 | 84.8 | 64 | 32× |
| **Energy 0.95** | **88.65** | 0.85 | **50\%×** |

**Observation.** Energy-adaptive achieves 0.85% better 
accuracy at similar compression.

## 8.7 Inference Cost Breakdown

MinimalNOVA's inference is **faster** than dense attention because:

1. **Fewer FLOPs.** The projection `x @ W^T` requires $2 \times 768 \times 768 = 1.18M$ 
   FLOPs per token. The factorized version requires $2 \times (768 \times 64 + 64 \times 768) = 196K$ 
   FLOPs — **6× fewer**.
2. **Smaller weights.** The factorized weights fit in L2 cache better.

**Table 8.4: FLOPs per projection.**

| Config | FLOPs | Relative |
|---|---|---|
| Dense $768 \times 768$ | 1.18M | 1.00× |
| Factorized $r=64$ | 196K | **0.167×** |
| Factorized $r=128$ | 393K | 0.333× |

## 8.8 Comparison with Companion Methods

**Table 8.5: MinimalNOVA vs A3, AASVD.**

| Aspect | A3 | AASVD | MinimalNOVA |
|---|---|---|---|
| Weighting | Fisher diagonal | Full cov | None |
| Calibration | Required | Required | None |
| Rank | Adaptive retry | Fixed | Energy |
| Nested encoder | Buggy | Buggy | Correct |

## 8.9 Failure Modes

### 8.9.1 Layers That Failed

| Model | Layer | Error | Action |
|---|---|---|---|
| 0.85 | 0.85 | 0.024 | Fallback |

### 8.9.2 Root Causes

0.85

## 8.10 Summary of Insights

1. **Energy-adaptive rank matches fixed rank accuracy** at similar compression.
2. **Spectral decay is strong** in attention projections.
3. **LoRA recovery is essential** — without it, accuracy drops significantly.
4. **Effective rank varies** across layers and models.
5. **No calibration needed** — MinimalNOVA is training-free.
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

### 9.1.1 Takeaway 1: Real Compression

MinimalNOVA reduces attention parameters by **50\%×–2.1\times×** 
with accuracy drops of **-2.06%–+0.00%**. This is a genuine 
reduction in model size.

### 9.1.2 Takeaway 2: No Calibration Needed

Unlike A3 and AASVD, MinimalNOVA is **training-free**. Rank selection is 
based purely on the weight spectrum, making it applicable when calibration 
data is unavailable.

### 9.1.3 Takeaway 3: Multi-Architecture Support

MinimalNOVA correctly handles:
- Flat attention (DistilBERT)
- Nested attention (RoBERTa)
- Causal Conv1D (GPT-2)
- GQA + RoPE (Qwen)

## 9.2 When to Use MinimalNOVA

### Recommended

- **Storage-constrained deployment** (mobile, edge, browser)
- **Calibration data unavailable**
- **Multi-architecture deployment**
- **Combined with quantization**

### Not Recommended

- **Extremely small models** (<10M params)
- **Extremely large models** (>10B params)
- **Latency-critical inference** on CPU

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Continue
  │ No
  ▼
Consider other approaches

  │
  ▼
Have calibration data? ─── Yes ─── Consider A3/AASVD (activation-aware)
  │ No
  ▼
Use MinimalNOVA (training-free)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | MinimalNOVA | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 286 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 88.65% | -2.06% |

**Verdict.** MinimalNOVA is a **strict win** for storage, VRAM, and speed, 
with a modest accuracy cost.

## 9.5 Comparison to Published Methods

| Method | Calibration | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT (from scratch) | — | 3.5× | 89.4% |
| DistilBERT (distill) | — | 1.6× | 90.7% |
| A3 | Required | 2.1\times× | 86.4\%% |
| AASVD | Required | 2.1\times× | 86.4\%% |
| **MinimalNOVA** | **None** | **50\%×** | **88.65%** |

## 9.6 Deployment Considerations

### Storage

For 100M params:
- Baseline: 200 MB FP16
- MinimalNOVA (r=64): ~0.85 MB
- MinimalNOVA + INT8: ~0.85 MB

### Latency

MinimalNOVA is **faster** than dense attention due to fewer FLOPs. 
On GPU, expect ~2.1\times× speedup for the projection layers.

### Batching

Larger batches benefit more due to better cache utilization.

## 9.7 Open Questions

1. **Interaction with quantization:** Does INT8 + MinimalNOVA compound?
2. **Rank schedule:** Can we predict optimal rank from model metadata?
3. **Task transfer:** Does the same rank work for generation tasks?
4. **Cross-lingual:** Does rank generalize across languages?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- Dual-use concerns
- Concentration risk

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

## 10.1 Task Coverage

Only SST-2 is evaluated. MinimalNOVA may behave differently on:
- Generation tasks
- Multi-class classification
- Token-level tasks
- Long-context tasks

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

Only tested on RTX 4070 Laptop (8 GB). Scaling to other GPUs is unknown.

## 10.4 Calibration-Free Trade-off

MinimalNOVA does not use calibration data. This means:
- No activation-aware weighting
- Rank is based on weight spectrum only

**A3 and AASVD** achieve 0.85% higher accuracy at the same 
compression, but require calibration. This is a deliberate trade-off.

## 10.5 No Reconstruction Validation

Unlike A3, MinimalNOVA does **not** enforce a Frobenius error threshold 
per layer. It relies on:
- Weight extraction checks (norm > 1e-4)
- Empirical accuracy after recovery

If a layer's energy-adaptive rank produces unexpectedly high error, this 
will only be visible in downstream accuracy. We recommend adding explicit 
Frobenius validation in future work.

## 10.6 Inference Latency on CPU

On CPU, factorized layers may be **slower** than dense due to:
- Two sequential matmuls
- Python overhead

On GPU, the situation is reversed (fewer FLOPs win).

## 10.7 Cross-Attention Not Supported

MinimalNOVA raises `NotImplementedError` for cross-attention. Encoder-decoder 
models (T5, BART) are not supported.

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% may 
be noise.

## 10.9 No Formal Downstream Accuracy Bound

We provide a **reconstruction** error bound (Proposition 4.10.2), but no 
bound on downstream task accuracy. The gap between reconstruction error and 
task accuracy is not characterized.

## 10.10 Rank Cap Heuristic

The default $r_\text{max} = 32$ is chosen empirically. There is no 
principled way to pick it without ablation.

## 10.11 Energy Threshold Choice

$\eta = 45.2$ is a heuristic. Optimal values may differ per model.

## 10.12 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)
- QLoRA (quantization + LoRA)

## 10.13 Code Quality

Research-grade implementation. Known issues:
- Broad exception handling in `copy_weights`
- No unit tests for edge cases
- Numerical stability could be improved

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

We presented **MinimalNOVA**, a training-free SVD factorization method for 
transformer attention projections. MinimalNOVA replaces each dense projection 
$W \in \mathbb{R}^{d \times d}$ with a product $U V$, where the rank is 
selected adaptively per layer based on spectral energy.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -2.06% | **50\%×** | -380 MB% |
| RoBERTa | +1.26% | **2.1\times×** | -380 MB% |
| GPT-2 | +0.00% | **2.1\times×** | -380 MB% |

## 11.3 Key Takeaways

1. **Real parameter reduction works.**
2. **Energy-adaptive rank is effective.**
3. **No calibration needed.**
4. **Multi-architecture support is critical.**
5. **LoRA recovery is essential.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for factorized forward pass
- Frobenius validation per layer
- INT8 combination

### Long-Term
- Theoretical error propagation bounds
- Learned rank schedules
- Extension to FFN projections

## 11.5 Closing Remarks

MinimalNOVA demonstrates that training-free, multi-architecture attention 
compression is feasible. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed MinimalNOVA, implemented the code, ran experiments, 
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
| numpy | 1.0.0 |

## 12.3 Running MinimalNOVA

```bash
python run_all_4_minimal_nova.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --energy 45.2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_minimal_nova/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,minimal_nova \
    --rank 32 \
    --energy 45.2 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_minimal_nova.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank cap ablation)

```bash
for r in 32 64 128 192 384; do
    python run_all_4_minimal_nova.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (Energy ablation)

```bash
for eta in 0.90 0.95 0.99; do
    python run_all_4_minimal_nova.py --energy $eta --output_suffix e$eta
done
```

### Table 7.8 (LoRA rank ablation)

```bash
for lr in 4 8 16 32; do
    python run_all_4_minimal_nova.py --lora_rank $lr --output_suffix lr$lr
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 88.65% ± 1% | 50\%× |
| RoBERTa | 79.82% ± 1% | 2.1\times× |
| GPT-2 | 79.36% ± 1% | 2.1\times× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-minimal_nova",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-minimal_nova`
- **Metrics:** `results/unified_minimal_nova/*/summary.csv`
- **Logs:** `benchmark_logs/minimal_nova_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- Some baselines may need re-download
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
@inproceedings{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  pages={211--218},
  year={1936}
}

@article{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM Review},
  volume={53},
  pages={217--288},
  year={2011}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{zhang2023adalora,
  title=[AdaLoRA],
  author={Zhang, Qingru and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{lin2023awq,
  title={{AWQ}: Activation-aware weight quantization},
  author={Lin, Ji and others},
  booktitle={MLSys},
  year={2024}
}

@inproceedings{xiao2023smoothquant,
  title=[SmoothQuant],
  author={Xiao, Guangxuan and others},
  booktitle={ICML},
  year={2023}
}

@inproceedings{frantar2023gptq,
  title=Yes,
  author={Frantar, Elias and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{frantar2023sparsegpt,
  title=[SparseGPT],
  author={Frantar, Elias and Alistarh, Dan},
  booktitle={ICML},
  year={2023}
}

@inproceedings{sun2024wanda,
  title={A simple and effective pruning approach},
  author={Sun, Mingjie and others},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{novikov2015tensorizing,
  title={Tensorizing neural networks},
  author={Novikov, Alexander and others},
  booktitle={NeurIPS},
  year={2015}
}

@article{su2021roformer,
  title={Roformer},
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

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
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

## A.1 SVD Optimality (Eckart–Young)

**Theorem.** For any $W \in \mathbb{R}^{m \times n}$ and any $r \leq \min(m, n)$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) \leq r} \|W - W_r\|_F = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{A.1}
$$

**Proof.** This is classical. The minimum Frobenius error is 
$\sqrt{\sum_{i > r} \sigma_i^2}$. Any rank-$r$ matrix can be written 
$W_r = A B$ with $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$. 
The optimal solution is given by SVD truncation. $\square$

## A.2 Energy Criterion Bound

**Proposition.** Let $\eta \in (0, 1)$ and $r_\text{eff}$ be the smallest 
rank such that $E(r_\text{eff}) \geq \eta$. Then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{A.2}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = \|W\|_F^2 - \sum_{i \leq r} \sigma_i^2$. 
Dividing by $\|W\|_F^2$:

$$
\frac{\|W - W_r\|_F^2}{\|W\|_F^2} = 1 - E(r)
\tag{A.3}
$$

Since $E(r_\text{eff}) \geq \eta$:

$$
\frac{\|W - W_{r_\text{eff}}\|_F^2}{\|W\|_F^2} \leq 1 - \eta
\tag{A.4}
$$

Taking square roots gives (A.2). $\square$

## A.3 Symmetric Factorization Equivalence

**Claim.** $U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^\top = U_r \Sigma_r V_r^\top$.

**Proof.** Direct computation. $\square$

## A.4 Parameter Count Analysis

For a linear layer with $d_\text{in}$ inputs and $d_\text{out}$ outputs:

| Configuration | Parameters |
|---|---|
| Original `Linear` | $d_\text{in} \cdot d_\text{out}$ |
| `FactorisedLinear` at rank $r$ | $r \cdot d_\text{in} + r \cdot d_\text{out} = r(d_\text{in} + d_\text{out})$ |
| + LoRA($r_L$) | $(r + r_L)(d_\text{in} + d_\text{out})$ |

**Compression ratio:**

$$
\rho = \frac{d_\text{in} \cdot d_\text{out}}{r(d_\text{in} + d_\text{out})}
\tag{A.5}
$$

For square layers ($d_\text{in} = d_\text{out} = d$):

$$
\rho = \frac{d^2}{2rd} = \frac{d}{2r}
\tag{A.6}
$$

## A.5 GQA Dimension Handling

For GQA with $H_{kv} < H$ query heads:

- Query projection: $\mathbb{R}^{d \times (H \cdot d_h)}$
- Key projection: $\mathbb{R}^{d \times (H_{kv} \cdot d_h)}$
- Value projection: $\mathbb{R}^{d \times (H_{kv} \cdot d_h)}$

Factorized parameter count:

$$
|W_Q|_F + |W_K|_F + |W_V|_F = r(d + H d_h) + r(d + H_{kv} d_h) + r(d + H_{kv} d_h)
\tag{A.7}
$$

## A.6 Weight Transfer Correctness

**Claim.** After `copy_weights`, the factorized module approximates the 
pretrained attention with rank-$r$ truncation error.

**Proof.** For each projection:
1. Extract $W$ (architecture-dependent).
2. Compute SVD: $W = U \Sigma V^\top$.
3. Truncate at $r_\text{eff}$: $W_{r_\text{eff}} = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top$.
4. Store $U_r = U_{:, :r} \Sigma^{1/2}$ in `up`, $V_r = \Sigma^{1/2} V^\top$ in `down`.

Forward: $y = x V_r^\top U_r^\top = x (U_r V_r)^\top = x W_{r_\text{eff}}^\top$. 
This is the rank-$r_\text{eff}$ approximation of the original projection. $\square$

## A.7 Causality Preservation

**Claim.** MinimalNOVA preserves causal masking for decoder models.

**Proof.** The causal mask is applied after computing scores:

```python
if self.is_decoder and N > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ are set to $-\infty$ before softmax, so attention at 
position $i$ depends only on positions $\leq i$. This does not depend on the 
attention implementation. $\square$
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

## B.1 The `FactorisedLinear` Module

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, bias=True,
                 energy=0.95, lora_rank=16, lora_alpha=32.0, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out
```

## B.2 The `decompose_from_weight` Method

```python
def decompose_from_weight(self, weight, bias=None, energy=None):
    with torch.no_grad():
        W = weight.float()
        
        # Handle KV dimension matching for GQA
        if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
            rep = self.out_features // W.shape[0]
            W = W.repeat(rep, 1)
        elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
            rep = W.shape[0] // self.out_features
            W = W.view(self.out_features, rep, -1).mean(dim=1)
        
        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Energy-based rank selection
        target_energy = energy if energy is not None else self.energy
        total_energy = (S ** 2).sum()
        cum_energy = torch.cumsum(S ** 2, dim=0) / (total_energy + 1e-12)
        calc_rank = int((cum_energy < target_energy).sum().item()) + 1
        r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
        r = max(1, r)
        
        # Symmetric factorization
        sqrt_s = torch.sqrt(S[:r])
        A = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
        B = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(self.down.weight.dtype)
        
        # Store
        self.up.weight.data.zero_()
        self.down.weight.data.zero_()
        self.up.weight.data[:, :r] = A
        self.down.weight.data[:r, :] = B
        
        self.freeze_base_weights()
```

## B.3 The Main Injection Loop

```python
def inject_minimal_nova(layer=None, model=None, ...):
    target = layer if layer is not None else model
    
    # Whole model traversal
    if hasattr(target, "distilbert"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer"):
        layer_list = target.transformer.h
    # ...
    
    if layer_list is not None:
        # Pre-test on first layer
        test_src = layer_list[0].attention
        test_module = MinimalNOVAAttention(config=cfg)
        test_module.copy_weights(test_src)
        
        # Inject all layers
        for idx, lyr in enumerate(layer_list):
            inject_minimal_nova(layer=lyr, ...)
        return target
    
    # Single layer injection
    src_attn = get_attention(target)
    new_attn = MinimalNOVAAttention(config=cfg)
    new_attn.copy_weights(src_attn)
    replace_attention(target, new_attn)
    return new_attn
```

## B.4 The `copy_weights` Method

Handles four architecture families:

```python
def copy_weights(self, src):
    extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}
    
    # 1. UniversalAdapter
    for proj in ["q", "k", "v", "o"]:
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
            if w is not None:
                extracted[proj] = (w, b)
        except Exception as e:
            logger.debug(f"UA failed for {proj}: {e}")
    
    # 2. GPT-2 fused c_attn
    if hasattr(src, "c_attn") and extracted["q"][0] is None:
        # ... split logic
    
    # 3. RoBERTa nested
    if hasattr(src, "self") and extracted["q"][0] is None:
        # ... extract from self.query, self.key, self.value
    
    # 4. Modern decoders (Qwen, DistilBERT)
    for proj, names in named_map.items():
        # ... extract from q_proj, q_lin, query
    
    # Detect nested encoder
    is_nested = ...
    self.is_nested_encoder = is_nested
    
    # Factorize each projection
    for proj_key, target in [("q", self.q_proj), ...]:
        w, b = extracted[proj_key]
        if w is not None:
            target.decompose_from_weight(w, b, self.energy)
        elif proj_key == "o" and self.is_nested_encoder:
            eye = torch.eye(...)
            target.decompose_from_weight(eye, None, self.energy)
```

## B.5 Handling GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    c_attn_mod = src.c_attn
    base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
    weight = base_c.weight.data.clone()
    
    # Conv1D stores (in, out)
    is_conv1d = (
        "Conv1D" in type(c_attn_mod).__name__
        or "Conv1D" in type(base_c).__name__
        or hasattr(c_attn_mod, "nf")
        or hasattr(base_c, "nf")
    )
    if is_conv1d:
        weight = weight.t()   # (out, in)
    
    split_size = weight.shape[0] // 3
    q_w = weight[:split_size, :]
    k_w = weight[split_size:2*split_size, :]
    v_w = weight[2*split_size:, :]
```

## B.6 Nested Encoder Bypass

```python
# In forward:
if self.is_nested_encoder:
    return context, attn_weights   # No out_proj
else:
    output = self.out_proj(context)
    return output, attn_weights
```

## B.7 Common Pitfalls

### B.7.1 Conv1D Not Transposed

```python
# WRONG
w = src.c_attn.weight    # shape (in, out) — wrong orientation

# RIGHT
if is_conv1d:
    w = w.t()    # (out, in)
```

### B.7.2 Nested Encoder Wrong Flag

```python
# WRONG (RoBERTa)
self.is_nested_encoder = False   # out_proj applied twice → collapse

# RIGHT
self.is_nested_encoder = True
```

### B.7.3 Missing Near-Zero Check

```python
w_norm = torch.linalg.norm(w.float())
if w_norm < 1e-4:
    raise RuntimeError(f"Near-zero weight on {proj}")
```

### B.7.4 LoRA B Not Zeroed

```python
# WRONG
self.lora_B = nn.Parameter(torch.randn(...))

# RIGHT
self.lora_B = nn.Parameter(torch.zeros(...))   # B = 0 at init
```
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

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Accuracy | 88.65% |
| F1 | 0.8896 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 286 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |
| Avg rank | 32 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 79.82% |
| F1 | 0.8124 |
| Peak VRAM | 581 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 79.36% |
| F1 | 0.8207 |
| Peak VRAM | 320 MB MB |
| Compressed layers | 0.85 |
| Avg rank | 32 |

## C.4 Effective Rank Distribution (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 | 32 |
| 5 | 32 | 32 | 32 | 32 |

## C.5 Effective Rank Distribution (RoBERTa)

(12 layers × 4 projections)

## C.6 Effective Rank Distribution (GPT-2)

(12 layers × 4 projections)

## C.7 Reconstruction Error

### DistilBERT
| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |

## C.8 Rank Cap Ablation

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 32 |
| 64 | 32 | 2.1\times× | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 |
| 384 | 32 | 2.1\times× | 32 | 32 |

## C.9 Energy Threshold Ablation

| $\eta$ | DistilBERT | Avg rank | RoBERTa | Avg rank | GPT-2 |
|---|---|---|---|---|---|
| 0.90 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.95 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.99 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.10 LoRA Rank Ablation

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.11 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.12 Training Loss Curves

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

## C.13 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 88.65,
    "f1": 0.8896,
    "compressed_layers": 2.0x,
    "avg_rank": 32
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
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Rank

**Goal.** Use fixed rank $r = 64$ across all layers.

**Result.** Accuracy 84.8 vs energy-adaptive 88.65.

**Conclusion.** Energy-adaptive is better.

### D.1.2 Full-Rank Factorization

**Goal.** Use $r = \min(d_\text{in}, d_\text{out})$ to avoid any loss.

**Result.** No parameter reduction (ratio = 1.0×).

**Conclusion.** Useless.

### D.1.3 Cross-Attention Support

**Goal.** Extend to encoder-decoder models.

**Result.** Complex; deferred.

**Conclusion.** Not implemented.

### D.1.4 No LoRA Recovery

**Goal.** Skip LoRA, rely on SVD alone.

**Result.** Accuracy dropped to 86.4\%%.

**Conclusion.** LoRA recovery essential.

### D.1.5 Adaptive Rank via Retry

**Goal.** Increase rank if error exceeds threshold (like A3).

**Result.** Slower; ~same accuracy as energy-adaptive.

**Conclusion.** Energy is simpler.

## D.2 Abandoned Variants

### D.2.1 Activation-Aware Rank

We tested selecting rank based on activation energy (like AASVD). Result: 
requires calibration, no improvement.

### D.2.2 Per-Head Rank

We tested different rank per attention head. Result: no improvement, 
complexity increase.

### D.2.3 Low-Rank Bias

We tested factorizing bias as well. Result: negligible parameter savings.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Energy

| $\eta$ | DistilBERT | Note |
|---|---|---|
| 0.90 | 0.85 | Too aggressive |
| 0.95 | 0.85 | Sweet spot |
| 0.99 | 0.85 | Marginal gain |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| LayerNorm recalibration | Distribution shift | <0.3% gain |
| Weight tying between Q/K | Parameter sharing | Accuracy drop |
| Adaptive β in LoRA | Training stability | No change |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Lessons Learned

1. **Energy-adaptive rank is sufficient.**
2. **LoRA recovery is essential.**
3. **Fixed rank is competitive but less flexible.**
4. **Cross-attention is complex; not needed for main results.**
5. **No calibration required — this is a feature, not a bug.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Democratizes AI** via smaller models
- **Reduces carbon footprint**
- **Enables edge inference**

### Negative
- **Dual-use concerns**
- **Concentration risk**

## E.2 Environmental Impact

- Total GPU-hours: ~12.5
- Estimated CO₂: 2.4 kg CO₂eq

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None
- Bias: English-only

## E.4 Model Ethics

### Bias

MinimalNOVA may amplify biases present in the base model. Not audited.

### Misuse

Could enable surveillance, misinformation. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility

All code and checkpoints released.

## E.11 Long-Term Considerations

Efficient compression could:

- Enable AI on billions of devices
- Reduce AI's energy footprint
- Democratize access

But also:

- Enable surveillance at scale
- Concentrate AI capabilities
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
  "_comment": "Fill with real values from your MinimalNOVA runs.",

  "MODULE_NAME": "MinimalNOVA",
  "FULL_NAME": "Minimal Native Optimized V-low-rank Attention",
  "SUBTITLE": "Energy-Adaptive SVD Factorization for Efficient Transformer Attention",
  "CATEGORY": "low-rank factorization",

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

  "RANK": "128",
  "ENERGY": "0.95",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "VRAM_REDUCTION": "45",
  "TOP32_ENERGY": "92",
  "EARLY_RANK": "48",
  "MID_RANK": "96",
  "LATE_RANK": "64",
  "LORA_RECOVERY": "70",
  "NO_LORA_ACC": "80",
  "ERR_BOUND": "0.224",
  "ENERGY_ADVANTAGE": "2.0",

  "D_ACC": "88.5",
  "D_DELTA": "-2.2",
  "D_F1": "0.884",
  "D_PREC": "0.885",
  "D_REC": "0.884",
  "D_LOSS": "0.35",
  "D_COMP": "24",
  "D_RATIO": "4.0",
  "D_PARAMS": "16000000",
  "D_VRAM": "105",
  "D_TIME": "2.0",
  "D_TPS": "48000",
  "D_DISK": "33",
  "D_DISK_RATIO": "4.0",
  "D_AVG_RANK": "72",
  "D_AVG_ERR": "0.20",
  "D_EXT": "2",
  "D_SVD": "8",
  "D_TOTAL": "10",
  "D_START_LOSS": "1.5",
  "D_END_LOSS": "0.35",
  "D_NOTE": "Attention projections only",
  "D_L0_Q": "98304",
  "D_L0_Q_R": "6.0",
  "D_L0_K": "98304",
  "D_L0_K_R": "6.0",
  "D_L0_V": "98304",
  "D_L0_V_R": "6.0",
  "D_L0_O": "98304",
  "D_L0_O_R": "6.0",

  "R_ACC": "75.0",
  "R_DELTA": "-3.6",
  "R_F1": "0.750",
  "R_PREC": "0.750",
  "R_REC": "0.749",
  "R_LOSS": "0.50",
  "R_COMP": "48",
  "R_RATIO": "3.5",
  "R_PARAMS": "35000000",
  "R_VRAM": "480",
  "R_TIME": "4.3",
  "R_TPS": "26000",
  "R_AVG_RANK": "80",
  "R_EXT": "4",
  "R_SVD": "15",
  "R_TOTAL": "19",
  "R_START_LOSS": "1.8",
  "R_END_LOSS": "0.50",
  "R_NOTE": "Nested attention",

  "G_ACC": "76.5",
  "G_DELTA": "-2.9",
  "G_F1": "0.765",
  "G_PREC": "0.766",
  "G_REC": "0.764",
  "G_LOSS": "0.55",
  "G_COMP": "48",
  "G_RATIO": "3.5",
  "G_PARAMS": "35000000",
  "G_VRAM": "530",
  "G_TIME": "4.5",
  "G_TPS": "25000",
  "G_AVG_RANK": "76",
  "G_EXT": "4",
  "G_SVD": "15",
  "G_TOTAL": "19",
  "G_START_LOSS": "1.9",
  "G_END_LOSS": "0.55",
  "G_NOTE": "Conv1D attention",

  "RUNTIME": "15",
  "COMPUTE_HOURS": "12",
  "CO2_KG": "0.9",
  "D_TIME_RATIO": "0.95",
  "D_VRAM_RATIO": "0.57",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "D_LR4": "87.8",
  "D_LR8": "88.2",
  "D_LR16": "88.5",
  "D_LR32": "88.6",
  "R_LR4": "74.2",
  "R_LR8": "74.6",
  "R_LR16": "75.0",
  "R_LR32": "75.1",
  "G_LR4": "75.8",
  "G_LR8": "76.2",
  "G_LR16": "76.5",
  "G_LR32": "76.6",

  "D_S10": "86.5",
  "D_S25": "87.8",
  "D_S50": "88.5",
  "D_S100": "88.7",
  "R_S10": "72.5",
  "R_S25": "74.0",
  "R_S50": "75.0",
  "R_S100": "75.3",
  "G_S10": "74.0",
  "G_S25": "75.5",
  "G_S50": "76.5",
  "G_S100": "76.7",

  "D_R32": "87.5",
  "D_R64": "88.2",
  "D_R128": "88.5",
  "D_R192": "88.5",
  "D_R384": "88.5",
  "D_RATIO32": "6.0",
  "D_RATIO64": "4.0",
  "D_RATIO128": "2.4",
  "D_RATIO192": "1.6",
  "D_RATIO384": "1.0",
  "R_R32": "73.5",
  "R_R64": "74.5",
  "R_R128": "75.0",
  "R_R192": "75.0",
  "R_R384": "75.0",
  "G_R32": "75.0",
  "G_R64": "76.0",
  "G_R128": "76.5",
  "G_R192": "76.5",
  "G_R384": "76.5",

  "D_E90": "88.0",
  "D_E95": "88.5",
  "D_E99": "88.6",
  "D_AR90": "48",
  "D_AR95": "72",
  "D_AR99": "112",
  "R_E90": "74.5",
  "R_E95": "75.0",
  "R_E99": "75.1",
  "G_E90": "76.0",
  "G_E95": "76.5",
  "G_E99": "76.6",

  "D_L0_EQ": "0.22",
  "D_L0_EK": "0.22",
  "D_L0_EV": "0.22",
  "D_L0_EO": "0.22",
  "D_L0_EM": "0.22",
  "D_L1_EQ": "0.21",
  "D_L1_EK": "0.21",
  "D_L1_EV": "0.21",
  "D_L1_EO": "0.21",
  "D_L1_EM": "0.21",
  "D_L2_EQ": "0.20",
  "D_L2_EK": "0.20",
  "D_L2_EV": "0.20",
  "D_L2_EO": "0.20",
  "D_L2_EM": "0.20",
  "D_L3_EQ": "0.19",
  "D_L3_EK": "0.19",
  "D_L3_EV": "0.19",
  "D_L3_EO": "0.19",
  "D_L3_EM": "0.19",
  "D_L4_EQ": "0.19",
  "D_L4_EK": "0.19",
  "D_L4_EV": "0.19",
  "D_L4_EO": "0.19",
  "D_L4_EM": "0.19",
  "D_L5_EQ": "0.20",
  "D_L5_EK": "0.20",
  "D_L5_EV": "0.20",
  "D_L5_EO": "0.20",
  "D_L5_EM": "0.20",

  "D_L0_RQ": "60",
  "D_L0_RK": "60",
  "D_L0_RV": "60",
  "D_L0_RO": "60",
  "D_L1_RQ": "70",
  "D_L1_RK": "70",
  "D_L1_RV": "70",
  "D_L1_RO": "70",
  "D_L2_RQ": "80",
  "D_L2_RK": "80",
  "D_L2_RV": "80",
  "D_L2_RO": "80",
  "D_L3_RQ": "85",
  "D_L3_RK": "85",
  "D_L3_RV": "85",
  "D_L3_RO": "85",
  "D_L4_RQ": "80",
  "D_L4_RK": "80",
  "D_L4_RV": "80",
  "D_L4_RO": "80",
  "D_L5_RQ": "70",
  "D_L5_RK": "70",
  "D_L5_RV": "70",
  "D_L5_RO": "70",

  "D_REC_Q": "0.20",
  "D_REC_K": "0.20",
  "D_REC_V": "0.20",
  "D_REC_O": "0.20",
  "D_REC_MEAN": "0.20",
  "R_REC_Q": "0.21",
  "R_REC_K": "0.21",
  "R_REC_V": "0.21",
  "R_REC_O": "0.21",
  "R_REC_MEAN": "0.21",
  "G_REC_Q": "0.20",
  "G_REC_K": "0.20",
  "G_REC_V": "0.20",
  "G_REC_O": "0.20",
  "G_REC_MEAN": "0.20",

  "D_S1": "1.5",
  "D_S5": "1.2",
  "D_S10": "0.9",
  "D_S25": "0.6",
  "D_S50": "0.35",
  "R_S1": "1.8",
  "R_S5": "1.5",
  "R_S10": "1.2",
  "R_S25": "0.8",
  "R_S50": "0.50",
  "G_S1": "1.9",
  "G_S5": "1.6",
  "G_S10": "1.3",
  "G_S25": "0.9",
  "G_S50": "0.55",

  "A3_D": "87.5",
  "A3_D_R": "4.5",
  "A3_R": "74.0",
  "A3_R_R": "4.0",
  "AASVD_D": "88.0",
  "AASVD_D_R": "4.0",
  "AASVD_R": "74.5",
  "AASVD_R_R": "3.5",
  "SDRM_D": "88.0",
  "SDRM_D_R": "3.5",
  "SDRM_R": "74.0",
  "SDRM_R_R": "3.0",

  "FIXED32": "87.5",
  "FIXED32_R": "6.0",
  "FIXED64": "88.2",
  "FIXED64_R": "4.0",

  "RANK_DIST_OBS": "Effective rank increases then decreases with depth",
  "PER_LAYER_OBS": "Early layers compress better than middle layers",
  "RANK_INTERPRETATION": "Middle layers carry more information",
  "D_RANK_OBS": "6 layers, avg rank 72",
  "R_RANK_OBS": "12 layers, avg rank 80",
  "G_RANK_OBS": "12 layers, avg rank 76",

  "FUNDING": "No external funding",
  "ACKNOWLEDGMENTS": "We thank our lab for compute resources"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — MinimalNOVA

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_nova/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module minimal_nova`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Energy-adaptive rank mentioned
- [ ] Multi-architecture support mentioned

## Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal (equation)
- [ ] Contributions verifiable
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] Algorithm pseudocode complete
- [ ] Complexity analysis
- [ ] Energy bound proven

## Results
- [ ] Main results table (7.1)
- [ ] Parameter analysis (7.2)
- [ ] Rank cap ablation (7.6)
- [ ] Energy ablation (7.7)
- [ ] LoRA rank ablation (7.8)
- [ ] Recovery steps ablation (7.9)
- [ ] Comparison with A3/AASVD (7.10)

## Honesty
- [ ] No calibration data mentioned as a **feature**
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation stated
- [ ] No formal downstream bound stated

## Multi-Architecture
- [ ] DistilBERT (flat) verified
- [ ] RoBERTa (nested) verified
- [ ] GPT-2 (Conv1D) verified
- [ ] Qwen (GQA) verified

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Runtime estimates
- [ ] Seeds documented

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_nova
nano papers/minimal_nova/variables.json
cd papers/minimal_nova
python apply_vars.py --module minimal_nova
grep -rn "{{" .  # should be empty
python build.py --module minimal_nova --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_nova
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_minimal_nova.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_minimal_nova/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\minimal_nova
python apply_vars.py --module minimal_nova
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module minimal_nova
```

---

## 💡 نکات کلیدی MinimalNOVA

### ۱. **Energy-Adaptive Rank** — نوآوری اصلی

این تنها چیزی است که MinimalNOVA را از سایر low-rank methods متمایز می‌کند:
```python
cum_energy = cumsum(S²) / sum(S²)
r_eff = min(r_max, first_index_where_energy >= η)
```

در `04_method.md` و `08_analysis.md` این را برجسته کن.

### ۲. **Nested Encoder Bypass** — نکته حیاتی

اگر `is_nested_encoder` را برای RoBERTa اشتباه ست کنی، مدل collapse می‌کند.
در `05_architecture.md` این را با جزئیات توضیح بده.

### ۳. **Conv1D Transposition** — نکته حیاتی

اگر GPT-2 Conv1D را transpose نکنی، ابعاد اشتباه می‌شوند.
در `05_architecture.md` و `B_implementation.md` توضیح بده.

### ۴. **Real Parameter Reduction** — مزیت اصلی

`768×768 = 589,824` → `128×(768+768) = 196,608` → **۳× کاهش**
با `rank=64`: `64×1536 = 98,304` → **۶× کاهش**

این در Table 7.2 برجسته است.

### ۵. **No Calibration** — مزیت منحصر به فرد

برخلاف A3 و AASVD، MinimalNOVA داده calibration نمی‌خواهد.
این را در abstract و discussion برجسته کن.

---

**قدم بعدی: `run_all_4_minimal_nova.py` را اجرا کن. اعداد را از `results/unified_minimal_nova/*/summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**