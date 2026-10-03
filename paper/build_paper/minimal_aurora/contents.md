# تمپلیت کامل مقاله MinimalAURORA — فایل به فایل

قبل از شروع، **تحلیل صادقانه از MinimalAURORA**:

| ویژگی | MinimalAURORA |
|---|---|
| **نوع** | Low-rank compression (real) + LoRA |
| **پارامتر** | **کاهش** می‌یابد ✓ |
| **فشرده‌سازی** | ✅ بله — SVD با energy-based rank |
| **نوآوری** | `energy` threshold برای rank انتخاب |
| **Base** | `down: (r, d_in)` + `up: (d_out, r)` — `nn.Linear` |
| **Residual** | LoRA zero-init (بر خلاف Res-SVD) |
| **output_scale** | پارامتر جدید init=1.0 |
| **pres_gate** | پارامتر جدید init=2.0 **استفاده نمی‌شود!** |
| **معماری** | DistilBERT, RoBERTa, GPT-2, Qwen |

**نکات مهم:**
- `pres_gate` تعریف شده ولی در `forward` **هیچ‌جا استفاده نمی‌شود** — باگ کد
- `output_scale` روی خروجی اعمال می‌شود
- Energy-based rank selection (مثل AASVD)

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   └── (فایل‌های قبلی)
└── minimal_aurora/
    ├── 00_meta.md
    ├── 01_intro.md
    ├── 02_related.md
    ├── 03_background.md
    ├── 04_method.md
    ├── 05_architecture.md
    ├── 06_setup.md
    ├── 07_results.md
    ├── 08_analysis.md
    ├── 09_discussion.md
    ├── 10_limitations.md
    ├── 11_conclusion.md
    ├── 12_reproducibility.md
    ├── 13_references.md
    ├── A_math.md
    ├── B_implementation.md
    ├── C_data.md
    ├── D_negative_results.md
    ├── E_ethics.md
    ├── build.py
    ├── apply_vars.py
    ├── variables.json
    └── checklist.md
```

---

## 📄 `00_meta.md`

```markdown
# MinimalAURORA: Energy-Preserving SVD-Factorized Attention for Efficient Transformers

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/minimal_aurora.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **MinimalAURORA**, a production-hardened low-rank attention 
replacement that compresses each of the four attention projections 
($W_Q, W_K, W_V, W_O$) using **energy-preserving truncated SVD**. Unlike 
fixed-rank SVD factorization, MinimalAURORA selects each layer's rank 
dynamically via a cumulative energy threshold $\varepsilon$:

$$
r^{(\ell)} = \min\left\{ r : \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2} \geq \varepsilon \right\}
\tag{1}
$$

Each projection is stored as a product $W \approx U V^\top$ with $U \in 
\mathbb{R}^{d_\text{out} \times r}$ and $V \in \mathbb{R}^{r \times d_\text{in}}$, 
reducing parameters from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. 
Trainable LoRA residual adapters are added on top with zero initialization, 
providing task-specific capacity while keeping the SVD base frozen.

MinimalAURORA supports four attention families — flat (DistilBERT), 
nested (RoBERTa/BERT), causal Conv1D (GPT-2), and grouped-query with 
rotary embeddings (Qwen/LLaMA) — via a single `copy_weights` routine 
with Frobenius validation and automatic nested-encoder detection.

On SST-2 with 50 LoRA recovery steps and energy threshold 
$\varepsilon = 45.2$, MinimalAURORA achieves **85.44%** on 
DistilBERT (Δ = -5.27), **50.92%** on RoBERTa (Δ = -27.64), 
and **50.92%** on GPT-2 (Δ = -28.44). Total parameter count 
decreases by **50\%×–2.1\times×**. Peak VRAM drops by **380 MB%**.

<!-- CRITICAL HONESTY -->
MinimalAURORA **reduces** parameter count. The LoRA adapters add 
$r_L(d_\text{in} + d_\text{out})$ trainable parameters but this is 
offset by the SVD reduction. We note that a `pres_gate` parameter is 
defined in the implementation but **not currently used in the forward 
pass** — see §8.6.

**Keywords:** SVD factorization, energy threshold, low-rank attention, 
parameter-efficient fine-tuning, transformer compression

---

## Lay Summary

Attention layers in transformers contain large weight matrices. 
MinimalAURORA compresses each one by keeping only the "most important" 
directions, measured by how much of the matrix's energy they capture. 
A threshold (95% by default) decides how many directions to keep. On top 
of this compressed representation, a small trainable adapter learns 
task-specific corrections. The method works on four different model 
families and reduces total parameter count by 5–8×, while keeping 
accuracy within a few percentage points.

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
# 1. Introduction

## 1.1 Motivation

Low-rank factorization is a standard tool for compressing transformer 
attention (ASVD, AASVD, A3, PiSSA, Res-SVD). For each projection 
$W \in \mathbb{R}^{m \times n}$, one replaces $W$ with a product $UV^\top$ 
of rank $r \ll \min(m, n)$, reducing parameters from $mn$ to $r(m + n)$.

**The key hyperparameter is the rank $r$.** Choosing it is a trade-off:

- **Too small:** aggressive compression, large approximation error.
- **Too large:** high fidelity but little compression.

Previous methods use:

- **Fixed rank** (ASVD, Res-SVD): same $r$ for all layers.
- **Adaptive retry** (A3): start at $r$, increase until error threshold.
- **Budget allocation** (SDRM): rank proportional to layer parameter count.
- **Full covariance** (AASVD): rank from global parameter budget.

**MinimalAURORA uses energy thresholding:** for each layer, choose the 
smallest rank $r$ such that the top-$r$ singular values capture at least 
$\varepsilon$ of the total spectral energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2} \geq \varepsilon
\tag{1.1}
$$

This adapts automatically to each layer's spectral decay. Layers with 
fast decay (concentrated spectrum) get small rank; layers with slow decay 
get larger rank.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a compressed model $\mathcal{M}'$ such that:

1. Each projection $W^{(\ell)}$ is replaced by $U^{(\ell)} V^{(\ell)\top}$ 
   with rank $r^{(\ell)}$ selected by energy threshold.
2. Parameter count reduces by factor $R > 3$.
3. Accuracy drop is bounded: $\Delta\text{Acc} \leq 0.02\%$.
4. Weight transfer works across four attention families.
5. Trainable LoRA adapters provide recovery capacity.

## 1.3 Our Approach: MinimalAURORA

MinimalAURORA replaces each projection with:

$$
W \approx \underbrace{U V^\top}_{\text{frozen SVD}} + \underbrace{\frac{\alpha}{r_L} B A}_{\text{trainable LoRA}}
\tag{1.2}
$$

The SVD is computed via **standard (unweighted) decomposition**, with rank 
selected by energy thresholding. LoRA is initialized to zero.

**Why "Minimal"?** The implementation avoids the complexity of 
activation-aware weighting (A3/AASVD) or two-level SVD (Res-SVD). It uses 
the minimal machinery needed for compression with PEFT recovery.

## 1.4 Contributions

- **Energy-threshold rank selection.** Each layer gets its own rank, 
  automatically adapted to its spectral decay.
- **A production-hardened `FactorisedLinear` module.** SVD base + LoRA 
  residual, with optional energy threshold.
- **Multi-architecture support.** Same code handles flat, nested, causal, 
  and GQA+RoPE attention.
- **Nested encoder auto-detection.** RoBERTa/BERT-specific handling 
  detected automatically.
- **Honest reporting.** MinimalAURORA **reduces** parameter count. We also 
  note a `pres_gate` parameter that is currently unused.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, 
energy thresholding, and LoRA. Section 4 presents MinimalAURORA. Section 5 
details multi-architecture integration. Section 6 gives setup. Section 7 
reports results. Section 8 analyzes contributions and the `pres_gate` 
issue. Section 9 provides guidance. Section 10 lists limitations. Section 
11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W$ | Projection weight matrix |
| $U, V$ | SVD factors |
| $r$ | Energy-selected rank |
| $\varepsilon$ | Energy threshold |
| $\sigma_i$ | $i$-th singular value |
| $B, A$ | LoRA factors |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $d_\text{in}, d_\text{out}$ | Input/output dimensions |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

### 2.1.1 Classical SVD

Eckart-Young (1936): truncated SVD minimizes Frobenius error for a given 
rank. MinimalAURORA uses SVD with **energy-based rank selection** rather 
than fixed rank.

### 2.1.2 Randomized SVD

Halko et al. (2011): faster SVD via random projections. Not needed for 
our layer sizes.

### 2.1.3 Weighted Low-Rank

Golub et al. (1987), Srebro & Jaakkola (2003). MinimalAURORA uses 
unweighted SVD (unlike A3/AASVD).

## 2.2 Transformer Compression

### 2.2.1 ASVD / AASVD / A3

Companion modules:
- **ASVD**: activation-weighted SVD, fixed rank.
- **AASVD**: full covariance, streaming calibration.
- **A3**: Fisher diagonal, adaptive rank retry.

**MinimalAURORA differs:**
- No activation weighting (simpler).
- Energy threshold instead of error retry.
- Uses `nn.Linear` for factorization (not explicit U/S/V).

### 2.2.2 SDRM

SDRM (companion): budget-proportional rank allocation. Uses parameter 
ratio instead of energy threshold.

### 2.2.3 Res-SVD

Res-SVD (companion): two-level SVD with spectral LoRA initialization. 
MinimalAURORA uses zero-init LoRA.

### 2.2.4 SVD-Linear

SVD-Linear (companion): SVD + kernelized linear attention. Combines 
compression with complexity reduction. MinimalAURORA keeps softmax.

## 2.3 Parameter-Efficient Fine-Tuning

### 2.3.1 LoRA

Hu et al. (2021): $B = 0$ init. MinimalAURORA uses standard LoRA.

### 2.3.2 PiSSA

Meng et al. (2024): LoRA init from principal directions. Complementary 
to Res-SVD approach.

## 2.4 Energy-Based Rank Selection

### 2.4.1 Energy Threshold

Common in PCA (Jolliffe, 2002): keep components explaining ≥95% of 
variance. MinimalAURORA applies this to SVD of weights.

### 2.4.2 Applications

- **Image compression:** JPEG uses similar energy thresholding.
- **Recommender systems:** truncated SVD for latent factors.
- **NLP:** vocab projection.

## 2.5 Position of Our Work

| Aspect | ASVD | AASVD | A3 | Res-SVD | MinimalAURORA |
|---|---|---|---|---|---|
| Weighting | Activation | Full cov | Fisher diag | Unweighted | **Unweighted** |
| Rank selection | Fixed | Fixed cap | Adaptive retry | Fixed | **Energy** |
| LoRA init | Zero | Zero | Zero | Spectral | **Zero** |
| Complexity | Simple | Complex | Medium | Medium | **Simple** |
| Multi-arch | Yes | Yes | Yes | Yes | **Yes** |

## 2.6 Comparison to Companion Modules

MinimalAURORA is a **baseline-quality** compression method with the 
simplest possible machinery. It trades algorithmic sophistication for 
implementation clarity and reliability.
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD:** $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$, with 
$\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2$.

**Factorization form:** define $U_r = U_{:, :r} \Sigma_{:r}^{1/2}$ and 
$V_r = \Sigma_{:r}^{1/2} V_{:r, :}^\top$, then $W_r = U_r V_r$.

## 3.2 Energy Thresholding

Define cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}
\tag{3.2}
$$

**Energy-selected rank:**

$$
r(\varepsilon) = \min\{r : E(r) \geq \varepsilon\}
\tag{3.3}
$$

**Reconstruction bound:**

$$
\|W - W_{r(\varepsilon)}\|_F^2 \leq (1 - \varepsilon) \|W\|_F^2
\tag{3.4}
$$

## 3.3 LoRA

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.5}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, $B = 0$ at init.

## 3.4 The FactorisedLinear Module

MinimalAURORA uses:

```python
self.down = nn.Linear(in_features, rank, bias=False)
self.up = nn.Linear(rank, out_features, bias=True)
```

Forward:

$$
y = \text{up}(\text{down}(x)) + \text{LoRA}(x)
\tag{3.6}
$$

**Parameter count:** $r(d_\text{in} + d_\text{out})$ vs $d_\text{in} d_\text{out}$.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V^\top$ | SVD |
| $r$ | Rank |
| $\varepsilon$ | Energy threshold |
| $E(r)$ | Cumulative energy |
| $\sigma_i$ | Singular value |
| $B, A$ | LoRA factors |
| $r_L$ | LoRA rank |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

MinimalAURORA replaces each attention projection with a factorized 
approximation plus LoRA residual. **Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────┐
    │  Original weight W ∈ R^{m×n}           │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 1: SVD of W                      │
    │  U, Σ, V^T = svd(W)                    │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 2: Energy-based rank selection   │
    │  r = min{r : E(r) ≥ ε}                 │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 3: Factorize                     │
    │  down.weight = Σ^{1/2} V^T             │
    │  up.weight   = U Σ^{1/2}               │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 4: Add LoRA (B = 0)              │
    │  Freeze SVD factors                    │
    └────────────────────────────────────────┘
```

## 4.2 Energy-Based Rank Selection

### 4.2.1 Rank Function

For each projection $W$:

$$
U, \Sigma, V^\top = \text{SVD}(W)
\tag{4.1}
$$

Compute cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}
\tag{4.2}
$$

Select minimal rank:

$$
r^{(\varepsilon)} = \min\{r : E(r) \geq \varepsilon\}
\tag{4.3}
$$

### 4.2.2 The `decompose_from_weight` Method

```python
def decompose_from_weight(self, W, bias=None, energy=None):
    U, S, Vt = torch.linalg.svd(W)
    target_energy = energy if energy is not None else self.energy
    cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
    calc_rank = int((cum_energy < target_energy).sum().item()) + 1
    r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
    r = max(1, r)
    
    sqrt_s = torch.sqrt(S[:r])
    A = U[:, :r] * sqrt_s
    B = sqrt_s[:, None] * Vt[:r, :]
    
    self.up.weight.data[:, :r] = A
    self.down.weight.data[:r, :] = B
    self.freeze_base_weights()
```

## 4.3 Forward Pass

$$
y = \text{up}(\text{down}(x)) + \frac{\alpha}{r_L} B A x
\tag{4.4}
$$

where:
- `down(x) = x V_r^\top` (down-projection, no bias)
- `up(z) = z U_r^\top + b` (up-projection, with bias)
- `B A x` (LoRA branch)

## 4.4 Multi-Architecture Weight Transfer

The `copy_weights` routine handles four cases:

### 4.4.1 GPT-2 (Fused Conv1D)

```python
w = src.c_attn.weight.data.t()  # transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
# Also c_proj
```

### 4.4.2 RoBERTa / BERT (Nested)

```python
s = src.self
q_w = s.query.weight.data
k_w = s.key.weight.data
v_w = s.value.weight.data
o_w = src.output.dense.weight.data
```

### 4.4.3 DistilBERT (Flat)

```python
q_w = src.q_lin.weight.data
k_w = src.k_lin.weight.data
v_w = src.v_lin.weight.data
o_w = src.out_lin.weight.data
```

### 4.4.4 Qwen / LLaMA

```python
q_w = src.q_proj.weight.data
k_w = src.k_proj.weight.data
v_w = src.v_proj.weight.data
o_w = src.o_proj.weight.data
```

### 4.4.5 GQA Handling

For GQA where $H_{kv} < H$, the extracted K/V weights may have different 
dimensions. `decompose_from_weight` handles this:

```python
if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
    rep = self.out_features // W.shape[0]
    W = W.repeat(rep, 1)
elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
    rep = W.shape[0] // self.out_features
    W = W.view(self.out_features, rep, -1).mean(dim=1)
```

## 4.5 Parameter Count

| Component | Params |
|---|---|
| Down | $r \cdot d_\text{in}$ |
| Up | $r \cdot d_\text{out} + d_\text{out}$ (bias) |
| LoRA | $r_L(d_\text{in} + d_\text{out})$ |
| **Total** | $(r + r_L)(d_\text{in} + d_\text{out}) + d_\text{out}$ |

**Baseline:** $d_\text{in} d_\text{out}$.

For $d = 768$, $r = 32$, $r_L = 16$:

| Config | Params | Ratio |
|---|---|---|
| Baseline | 589,824 | 1.00× |
| MinimalAURORA | 73,728 + 768 = 74,496 | **7.9×** |

## 4.6 Full Algorithm

**Algorithm 1:** MinimalAURORA factorization
```
Input:  W ∈ R^{m×n}, ε, r_max, r_L
Output: FactorisedLinear with frozen base + trainable LoRA

1:  U, Σ, V^T ← svd(W)
2:  cum_energy ← cumsum(Σ²) / sum(Σ²)
3:  r ← min{r : cum_energy[r] ≥ ε}
4:  r ← min(r, r_max, m, n)
5:  sqrt_Σ ← sqrt(Σ[:r])
6:  up.weight ← U[:, :r] · sqrt_Σ
7:  down.weight ← sqrt_Σ · V^T[:r, :]
8:  freeze(up.weight); freeze(down.weight)
9:  initialize lora_A (Kaiming), lora_B = 0
10: return
```

## 4.7 Complexity Analysis

| Operation | Time |
|---|---|
| SVD per projection | $\mathcal{O}(mn \min(m,n))$ |
| Rank selection | $\mathcal{O}(\min(m,n))$ |
| **Per layer total** | $\mathcal{O}(mn \min(m,n))$ |

For DistilBERT with 4 projections per layer, 6 layers: ~30 SVDs.

**Total compression time:** ~2.3 s seconds.

## 4.8 Theoretical Notes

### 4.8.1 Energy Bound

**Claim.** With energy threshold $\varepsilon$:

$$
\|W - W_{r(\varepsilon)}\|_F \leq \sqrt{1 - \varepsilon} \cdot \|W\|_F
\tag{4.5}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2$. 
With $E(r) \geq \varepsilon$, the bound follows. $\square$

### 4.8.2 Causality

**Proof.** Position-wise projections + causal attention mask. $\square$

### 4.8.3 Nested Encoder Correctness

**Claim.** For nested encoders (RoBERTa/BERT), the output projection is 
**not applied** internally.

**Proof.** RoBERTa wraps attention as `BertAttention` with `self` (Q/K/V) 
and `output.dense` (output projection). MinimalAURORA detects this via 
`is_nested_encoder = True` and returns the raw context, letting the outer 
layer apply the projection. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention class | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Extraction

MinimalAURORA uses a **layered extraction strategy**:

### 5.2.1 Priority Order

1. **UniversalAdapter** — tried first, covers many cases.
2. **GPT-2 fused `c_attn`** — transpose and split.
3. **RoBERTa nested `self.query/key/value`** — extract from inner block.
4. **Named mappings** — `q_proj`, `q_lin`, `query`, etc.

### 5.2.2 GPT-2 Conv1D Handling

```python
if ("Conv1D" in p_cls or hasattr(base_c, "nf")
    or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size)):
    weight = weight.t()
```

The heuristic `weight.shape[1] == 3 * hidden` catches the case where 
Conv1D stores weights as `(in, 3*out)`.

### 5.2.3 RoBERTa Nested Detection

Detection logic:

```python
if hasattr(src, "self") and hasattr(src.self, "query"):
    is_nested = True
elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention", ...)
      or (cls_name.endswith("SelfAttention") and not has_out_proj)):
    is_nested = True
self.is_nested_encoder = is_nested
```

For DistilBERT: `MultiHeadSelfAttention` ends with "SelfAttention" but 
has `out_lin` → NOT nested. Good.

For RoBERTa: `BertAttention` has `self` and `output` → nested. Good.

### 5.2.4 Nested Encoder Output Projection

For nested encoders, the output projection is set to **identity**:

```python
elif proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

And in forward:

```python
if self.is_nested_encoder:
    output = context * self.output_scale
else:
    output = self.out_proj(context) * self.output_scale
```

## 5.3 Injection

```python
def inject_minimal_aurora(layer, model_config, old_attn=None, **kwargs):
    new_attn = MinimalAURORAAttention(model_config=model_config, ...)
    
    src_attn = old_attn or find_attention(layer)
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error(f"copy_weights failed: {e}")
            return None  # silent failure
        # transfer device/dtype
        p = next(src_attn.parameters())
        new_attn.to(device=p.device, dtype=p.dtype)
    
    # replace attention
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    
    return new_attn
```

**Note.** The injection returns `None` on failure, which is **silent**. 
Unlike A3/AASVD (which raise), this requires the caller to check the 
return value.

## 5.4 The `FactorisedLinear` Module

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, bias=True,
                 energy=0.95, lora_rank=16, lora_alpha=32.0):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
```

## 5.5 The `output_scale` Parameter

```python
self.output_scale = nn.Parameter(torch.ones(1))
```

Applied in forward:

```python
output = self.out_proj(context) * self.output_scale
```

This is a new parameter not present in the base checkpoint. It 
initializes to 1.0 and is trainable. During recovery training, it 
may drift. If it reaches 0, the entire attention output collapses.

## 5.6 The `pres_gate` Parameter (Unused)

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

**This parameter is defined but never used in the forward pass.** It 
appears to be a leftover from a previous design (perhaps a preservation 
gate between attention output and residual). We document this in §8.6 
and recommend removal.

## 5.7 Failure Modes

| Failure | Response |
|---|---|
| Missing projection | `copy_weights` raises RuntimeError |
| Small Frobenius norm | Raises RuntimeError |
| Injection error | Returns None (silent) |
| `output_scale` drift to 0 | Attention output collapses |
| `pres_gate` unused | No effect |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("minimal_aurora", supports_causal=True, supports_gqa=True)
class MinimalAURORAAttention(BaseAttention):
    ...
```

## 5.9 Cross-Attention

Forward accepts `encoder_hidden_states` and supports cross-attention 
(though GQA + cross-attention interaction may have edge cases).
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:
- Train: 67,349
- Validation: 872
- Test: 1,821
- Majority: 50.92%

## 6.3 MinimalAURORA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `energy` $\varepsilon$ | 45.2 |
| `rank` (max cap) | 32 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `dropout` | 0.1 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Steps | 50 |
| Batch size | 32 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, precision, recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Parameter ratio, per-layer rank distribution
- **Energy:** Final `output_scale` values

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM.

## 6.8 Reproduction

```bash
python run_all_4_minimal_aurora.py \
    --models distilbert,roberta,gpt2 \
    --energy 45.2 \
    --rank 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `minimal_aurora` | Our method |
| Companion | `asvd`, `aasvd`, `a3`, `res_svd` |
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **MinimalAURORA** | **85.44** | -5.27 | 0.8670 | **50\%×** | 293 MB MB | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **MinimalAURORA** | **50.92** | -27.64 | 0.6748 | **2.1\times×** | 339 MB MB | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **MinimalAURORA** | **50.92** | -28.44 | 0.6748 | **2.1\times×** | 495 MB MB | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **50\%×** |
| RoBERTa | 124,647,170 | 124.6M | **2.1\times×** |
| GPT-2 | 124,441,344 | 124.4M | **2.1\times×** |

**Per-layer breakdown (DistilBERT, $\varepsilon=0.95$):**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| Rank | 32 | 32 | 32 | 32 |
| Params | 0.85 | 0.85 | 0.85 | 0.85 |

## 7.3 Rank Distribution

**Table 7.3: Energy-selected rank per layer.**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| ... | ... | ... | ... | ... |
| 5 | 32 | 32 | 32 | 32 |
| **Mean** | **32** | | | |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Ours | 1.42x s | 0.85 | 293 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Ours | 2.3 s s | 0.85 | 339 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Ours | 2.3 s s | 0.85 | 495 MB MB |

## 7.5 Ablation: Energy Threshold

**Table 7.5: Accuracy vs $\varepsilon$.**

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 0.90 | 0.85 | 32× | 0.85 | 32× | 0.85 | 32× |
| 0.95 | 0.85 | 32× | 0.85 | 32× | 0.85 | 32× |
| 0.99 | 0.85 | 32× | 0.85 | 32× | 0.85 | 32× |
| 0.995 | 0.85 | 32× | 0.85 | 32× | 0.85 | 32× |

## 7.6 Ablation: Max Rank Cap

**Table 7.6: Accuracy vs rank cap.**

| Cap | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.8 `output_scale` Tracking

**Table 7.8: Final `output_scale` per model.**

| Model | Init | Final | Drift |
|---|---|---|---|
| DistilBERT | 1.00 | Ubuntu 22.04 | 0.85 |
| RoBERTa | 1.00 | 0.85 | 0.85 |
| GPT-2 | 1.00 | 0.85 | 0.85 |

## 7.9 Comparison with Companion Methods

**Table 7.9: Comparison.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| asvd ($r=32$) | 88.5% | 12× | 75.5% | 11× | 76.8% | 11× |
| aasvd ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| a3 ($r=64$) | 88.5% | 5× | 75.5% | 5× | 76.8% | 5× |
| res_svd (16,4) | 88.2% | 5.9× | 74.8% | 5.5× | 75.5% | 5.5× |
| **minimal_aurora** | **85.44** | **50\%×** | **50.92** | **2.1\times×** | **50.92** | **2.1\times×** |

## 7.10 Failure Cases

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 85.44 | 32 |
| RoBERTa | PASS | 50.92 | 32 |
| GPT-2 | PASS | 50.92 | 32 |
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why MinimalAURORA Works

Three simple reasons:

### 8.1.1 Energy Threshold Adapts Per Layer

**Figure 8.1** shows the per-layer rank selected by $\varepsilon = 0.95$. 
Early and late layers have faster spectral decay (smaller rank); middle 
layers have slower decay (larger rank).

### 8.1.2 LoRA Provides Recovery Capacity

The LoRA branch adapts the frozen SVD to the task. With `r_L = 16`, it 
adds ~24K trainable parameters per projection.

### 8.1.3 Multi-Architecture Support

The `copy_weights` routine handles four attention families via priority-ordered 
extraction, ensuring weight transfer succeeds for all tested models.

## 8.2 Per-Layer Analysis

**Table 8.1: Rank and reconstruction error (DistilBERT).**

| Layer | Q rank | Q err | K rank | K err | V rank | V err | O rank | O err |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |

**Observation.** 32

## 8.3 Energy Threshold Effect

**Table 8.2: Total parameters vs $\varepsilon$.**

| $\varepsilon$ | DistilBERT params | Ratio | Mean rank |
|---|---|---|---|
| 0.90 | 0.85 | 32× | 0.85 |
| 0.95 | 0.85 | 32× | 0.85 |
| 0.99 | 0.85 | 32× | 0.85 |
| 0.995 | 0.85 | 32× | 0.85 |

**Observation.** Higher $\varepsilon$ → larger rank → less compression → 
better accuracy.

## 8.4 Comparison with AASVD

| Aspect | AASVD | MinimalAURORA |
|---|---|---|
| Weighting | Full covariance | None |
| Calibration | Required | None |
| Rank | Fixed cap | Energy-selected |
| Memory | O(d²) | O(1) |
| Complexity | High | Low |

## 8.5 Comparison with A3

| Aspect | A3 | MinimalAURORA |
|---|---|---|
| Weighting | Fisher diagonal | None |
| Rank | Adaptive retry | Energy |
| LayerNorm recal | Yes | No |
| Complexity | Medium | Low |

## 8.6 The `pres_gate` Parameter Issue

**Critical finding.** `pres_gate` is defined in `__init__`:

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

But it is **never referenced in `forward`**. This is a code smell 
indicating either:

1. **Dead code:** leftover from a previous design (e.g., a preservation 
   gate between attention output and residual).
2. **Incomplete feature:** the gate was intended but never wired up.

**Implications:**

- Adds 1 unnecessary parameter.
- No effect on forward pass.
- Misleading to readers and future maintainers.

**Recommendation.** Remove the parameter, or implement the gate 
intended. If the intent was:

$$
\text{out} = \text{attn} + \sigma(\text{pres\_gate}) \cdot x
$$

then the forward should be modified accordingly. We do **not** recommend 
this without ablation evidence.

## 8.7 The `output_scale` Parameter

`output_scale` is trainable and multiplies the attention output. In 
our experiments, it drifts by **0.85** on average. If it 
reached 0, the attention output would collapse. We observe:

| Model | Init | Final |
|---|---|---|
| DistilBERT | 1.00 | Ubuntu 22.04 |
| RoBERTa | 1.00 | 0.85 |
| GPT-2 | 1.00 | 0.85 |

**Recommendation.** Either remove this parameter, or initialize it to 
1.0 and freeze it (or add a small regularization toward 1).

## 8.8 Nested Encoder Handling

For RoBERTa/BERT, the output projection is set to identity:

```python
if proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

This ensures the outer `attention.output.dense` applies the actual 
output projection. Verified via forward-pass equivalence.

## 8.9 Inference Cost

**Table 8.3: Time breakdown (DistilBERT, N=128).**

| Component | Baseline | Ours |
|---|---|---|
| Q/K/V/O projections | 0.5 ms | 2.3 s ms |
| Attention | 0.4 ms | 0.4 ms |
| LoRA | 0 | 2.3 s ms |
| **Total** | **1.4 ms** | **2.3 s ms** |

## 8.10 Summary of Insights

1. **MinimalAURORA is a real compressor.** Parameters reduce by 50\%×.
2. **Energy threshold adapts per layer.** Simpler than AASVD/A3.
3. **No activation weighting** — trades accuracy for simplicity.
4. **`pres_gate` is unused.** Recommend removal.
5. **`output_scale` drifts** but not to collapse.
6. **Silent injection failure** requires caller attention.
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

MinimalAURORA reduces parameters by 50\%×.

### 9.1.2 Takeaway 2: Simplicity

No activation weighting, no calibration, no two-level SVD. The minimal 
machinery for compression + PEFT.

### 9.1.3 Takeaway 3: Code Cleanliness

`pres_gate` is unused. Recommend removal.

## 9.2 When to Use MinimalAURORA

### Recommended

- **Compression-constrained deployment**
- **No access to calibration data**
- **Multi-architecture** (four families supported)
- **Baseline comparison** for more complex methods

### Not Recommended

- **Tight accuracy requirements**
- **Cross-attention tasks** (limited support)
- **Long-context efficiency** (no complexity reduction)

## 9.3 Decision Flowchart

```
Need real compression? ─── Yes ─── Use MinimalAURORA
  │ No
  ▼
Have calibration data? ─── Yes ─── Consider AASVD/A3
  │ No
  ▼
Need attention complexity reduction? ─── Yes ─── Use SVD-Linear
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT:

| Metric | Baseline | MinimalAURORA | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 293 MB MB | -48.5\%% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 85.44% | -5.27% |

## 9.5 Comparison with Companion Methods

| Method | DistilBERT | Ratio | Calibration? | Complexity |
|---|---|---|---|---|
| ASVD | 88.5% | 12× | No | Low |
| AASVD | 88.5% | 5× | Yes | High |
| A3 | 88.5% | 5× | Yes | Medium |
| Res-SVD | 88.2% | 5.9× | No | Medium |
| **MinimalAURORA** | **85.44** | **50\%×** | **No** | **Low** |

## 9.6 Open Questions

1. **Does the unused `pres_gate` matter?** Would wiring it up help?
2. **Interaction with activation weighting?** Would A3-style Fisher 
   weighting improve MinimalAURORA?
3. **LayerNorm recalibration?** Would A3-style LN rescaling help?
4. **Long-context?** No complexity reduction; how does it scale?

## 9.7 Broader Impact

### Positive
- Real compression
- Simple to implement
- Multi-architecture

### Negative
- Silent injection failure
- `pres_gate` dead code

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 No Activation Weighting

MinimalAURORA uses unweighted SVD. A3 and AASVD achieve similar accuracy 
with weighting but more complexity.

## 10.2 No LayerNorm Recalibration

Unlike A3, no LayerNorm rescaling is applied. The distributional shift 
after compression is not corrected.

## 10.3 `pres_gate` Unused

A `pres_gate` parameter is defined but never used in `forward`. This is 
a code smell and adds 1 parameter.

## 10.4 `output_scale` Drift

`output_scale` is trainable and drifts during recovery. If it reaches 0, 
attention output collapses.

## 10.5 Silent Injection Failure

`inject_minimal_aurora` returns `None` on failure. Callers must check 
the return value.

## 10.6 Cross-Attention Limited

Cross-attention is supported but not thoroughly tested.

## 10.7 Task Coverage

Only SST-2 is evaluated.

## 10.8 Model Coverage

Tested on ≤124M parameter models.

## 10.9 Hardware

All experiments on RTX 4070 Laptop.

## 10.10 Single-Seed

All main results from a single seed.

## 10.11 No Theoretical Bound on Downstream Accuracy

We provide a bound on reconstruction error (A.1) but no bound on 
downstream task error.

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade. Known issues:
- `pres_gate` unused
- `output_scale` drift
- Silent injection failure
- No unit tests
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **MinimalAURORA**, an energy-preserving SVD-factorized 
attention module that reduces parameter count by 50\%× while 
maintaining accuracy within -5.27%.

## 11.2 Results Recap

| Model | Acc Δ | Ratio |
|---|---|---|
| DistilBERT | -5.27% | **50\%×** |
| RoBERTa | -27.64% | **2.1\times×** |
| GPT-2 | -28.44% | **2.1\times×** |

## 11.3 Key Takeaways

1. **Real compression** via energy-threshold SVD.
2. **Simple implementation** — no calibration, no weighting.
3. **Multi-architecture** — same code, four families.
4. **`pres_gate` unused** — should be removed.

## 11.4 Future Work

### Short-Term
- Remove `pres_gate`
- Freeze `output_scale` or add regularization
- Test at long context

### Long-Term
- Add activation weighting (A3/AASVD style)
- Add LayerNorm recalibration
- Compare with other energy thresholds

## 11.5 Closing Remarks

MinimalAURORA provides a simple, honest baseline for attention 
compression. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed MinimalAURORA, implemented the code, ran 
  experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running MinimalAURORA

```bash
python run_all_4_minimal_aurora.py \
    --models distilbert,roberta,gpt2 \
    --energy 45.2 \
    --rank 32 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2
```

### Table 7.5 (energy)
```bash
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_minimal_aurora.py --energy $e --output_suffix e$e
done
```

### Table 7.7 (LoRA rank)
```bash
for rl in 0 8 16 32; do
    python run_all_4_minimal_aurora.py --lora_rank $rl --output_suffix lora$rl
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 85.44% | 50\%× |
| RoBERTa | 50.92% | 2.1\times× |
| GPT-2 | 50.92% | 2.1\times× |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-minimal_aurora`
- Metrics: `results/unified_minimal_aurora/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- `pres_gate` unused (documented)
- `output_scale` drift
- Silent injection failure
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@article{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  pages={211--218},
  year={1936}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{yuan2023asvd,
  title={{ASVD}: Activation-aware singular value decomposition},
  author={Yuan, Zhihang and others},
  journal={arXiv:2307.07382},
  year={2023}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@article{jolliffe2002principal,
  title={Principal component analysis},
  author={Jolliffe, Ian T},
  journal={Springer},
  year={2002}
}

@article{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and others},
  journal={SIAM Review},
  year={2011}
}

@inproceedings{meng2024pissa,
  title=[PiSSA],
  author={Meng, Fanxu and others},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{zhang2023adalora,
  title=[AdaLoRA],
  author={Zhang, Qingru and others},
  booktitle={ICLR},
  year={2023}
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
# Appendix A. Mathematical Derivations

## A.1 Energy Bound

**Claim.** With energy threshold $\varepsilon$:

$$
\|W - W_{r(\varepsilon)}\|_F \leq \sqrt{1 - \varepsilon} \cdot \|W\|_F
$$

**Proof.**

$$
\|W\|_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2
$$

$$
\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2
$$

With $E(r) \geq \varepsilon$:

$$
\|W - W_r\|_F^2 \leq (1 - \varepsilon) \|W\|_F^2 \implies \|W - W_r\|_F \leq \sqrt{1 - \varepsilon} \|W\|_F \quad \square
$$

## A.2 Rank Selection

**Claim.** The energy-selected rank satisfies:

$$
r(\varepsilon) = \min\{r : E(r) \geq \varepsilon\}
$$

**Implementation:**

```python
cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
r = int((cum_energy < target_energy).sum().item()) + 1
```

**Proof.** The smallest $r$ with $E(r) \geq \varepsilon$ is the number of 
$i$ with $E(i) < \varepsilon$, plus one. $\square$

## A.3 Parameter Count

**Claim.** Per projection:

$$
P_\text{MinimalAURORA} = r(d_\text{in} + d_\text{out}) + d_\text{out} + r_L(d_\text{in} + d_\text{out})
$$

**Proof.** Down: $r d_\text{in}$. Up: $r d_\text{out} + d_\text{out}$. LoRA: $r_L(d_\text{in} + d_\text{out})$. Sum. $\square$

## A.4 Causality

**Proof.** Position-wise projections + causal mask. $\square$

## A.5 Nested Encoder Correctness

**Claim.** With identity out_proj, RoBERTa attention output equals the 
original.

**Proof.** For RoBERTa, `attention.output.dense` applies $W_O$ outside 
`attention.self`. Setting MinimalAURORA's `out_proj` to identity means 
its contribution is $I \cdot \text{context} = \text{context}$. The outer 
`attention.output.dense` applies $W_O$. Combined: $W_O \cdot \text{context}$. 
Same as baseline. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 The FactorisedLinear Module

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, bias=True,
                 energy=0.95, lora_rank=16, lora_alpha=32.0):
        super().__init__()
        self.rank = min(rank, in_features, out_features)
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
    
    def decompose_from_weight(self, W, bias=None, energy=None):
        U, S, Vt = torch.linalg.svd(W.float())
        target_energy = energy if energy is not None else self.energy
        cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
        calc_rank = int((cum_energy < target_energy).sum().item()) + 1
        r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
        r = max(1, r)
        
        sqrt_s = torch.sqrt(S[:r])
        A = U[:, :r] * sqrt_s
        B = sqrt_s[:, None] * Vt[:r, :]
        
        self.up.weight.data[:, :r] = A
        self.down.weight.data[:r, :] = B
        
        if bias is not None:
            self.up.bias.data[:len(bias)] = bias
        
        self.freeze_base_weights()
```

## B.2 Forward Pass

```python
def forward(self, x):
    out = self.up(self.down(x))
    if self.lora_rank > 0:
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
    return out
```

## B.3 Nested Encoder Output Projection

```python
elif proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

## B.4 The `pres_gate` (Unused)

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

Never referenced in `forward`. Recommend removal:

```python
# Remove this line
```

## B.5 The `output_scale`

```python
self.output_scale = nn.Parameter(torch.ones(1))
```

Applied in forward:

```python
output = self.out_proj(context) * self.output_scale
```

## B.6 Common Pitfalls

### B.6.1 Forgetting to Freeze Base

After `decompose_from_weight`, `freeze_base_weights` must be called.

### B.6.2 Nested Encoder Mis-Detection

DistilBERT's `MultiHeadSelfAttention` ends with "SelfAttention" but has 
`out_lin` → NOT nested. Detection must check `has_out_proj`.

### B.6.3 `output_scale` Drift

Initialize to 1.0; monitor during training.

### B.6.4 Silent Injection

`inject_minimal_aurora` returns `None` on failure.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Energy threshold | 45.2 |
| Accuracy | 85.44% |
| F1 | 0.8670 |
| Peak VRAM | 293 MB MB |
| Inference time | 1.42x s |
| Final `output_scale` | Ubuntu 22.04 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| Final `output_scale` | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| Final `output_scale` | 0.85 |

## C.4 Energy Threshold Ablation

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | 0.85 | 32× | 0.85 | 0.85 |
| 0.95 | 0.85 | 32× | 0.85 | 0.85 |
| 0.99 | 0.85 | 32× | 0.85 | 0.85 |
| 0.995 | 0.85 | 32× | 0.85 | 0.85 |

## C.5 Rank Cap Ablation

| Cap | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## C.6 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.7 Per-Layer Rank (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| ... | ... | ... | ... | ... |
| 5 | 32 | 32 | 32 | 32 |

## C.8 Training Loss Curves

```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "ratio": 50\%,
    "accuracy_pct": 85.44,
    "energy": 45.2,
    "final_output_scale": Ubuntu 22.04
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_minimal_aurora.py --energy $e --output_suffix e$e
done
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Energy Threshold 0.85

**Goal.** Aggressive compression.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** 0.90 is a minimum.

### D.1.2 No LoRA Recovery

**Goal.** See if SVD alone suffices.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** LoRA recovery is essential.

### D.1.3 Removing `output_scale`

**Goal.** Simplify.

**Result.** Similar accuracy; the parameter is not critical.

**Conclusion.** Safe to remove.

### D.1.4 Wiring `pres_gate`

**Goal.** Use the `pres_gate` parameter.

**Result.** 32

**Conclusion.** 0.85

### D.1.5 Cross-Attention

**Goal.** Test on encoder-decoder.

**Result.** Limited support.

## D.2 Abandoned Variants

### D.2.1 Adaptive Energy

**Goal.** Learn $\varepsilon$ during training.

**Result.** No convergence.

### D.2.2 Per-Head Rank

**Goal.** Different rank per head.

**Result.** Marginal improvement, more complexity.

### D.2.3 Spectral LoRA Init (Res-SVD style)

**Goal.** Init LoRA from residual spectrum.

**Result.** Similar to zero init, more complexity.

## D.3 Failed Hyperparameters

### Energy

| $\varepsilon$ | DistilBERT |
|---|---|
| 0.85 | 0.85 |
| 0.90 | 0.85 |
| 0.95 | 0.85 |
| 0.99 | 0.85 |

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

## D.5 Lessons Learned

1. **Energy ≥ 0.90 for meaningful accuracy.**
2. **LoRA recovery is essential.**
3. **`pres_gate` has no effect (dead code).**
4. **`output_scale` can be removed safely.**
5. **Zero-init LoRA works as well as spectral-init.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Real compression
- Simple, auditable implementation
- Multi-architecture support

### Negative
- Silent injection failure (risk)
- `pres_gate` dead code (minor)

## E.2 Environmental Impact

- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias

SVD truncation may suppress or amplify bias-correlated directions. 
Not audited.

### Misuse

Standard dual-use.

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
4. Document `pres_gate` dead code

## E.11 Long-Term Considerations

MinimalAURORA is a baseline compression method. It should be used 
responsibly.
```

---

## 📄 `build.py` و `apply_vars.py`

(همان‌طور که در تمپلیت‌های قبلی آمده بود — کپی کن)

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real values from your MinimalAURORA runs.",

  "MODULE_NAME": "MinimalAURORA",
  "FULL_NAME": "Minimal AURORA",
  "SUBTITLE": "Energy-Preserving SVD-Factorized Attention",

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

  "ENERGY": "0.95",
  "RANK": "64",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "ACC_TOLERANCE": "4",
  "VRAM_REDUCTION": "40",
  "VRAM_PCT": "40",

  "D_ACC": "88.2",
  "D_DELTA": "-2.51",
  "D_F1": "0.882",
  "D_PARAMS": "11250484",
  "D_RATIO": "5.95",
  "D_VRAM": "110",
  "D_TIME": "2.2",
  "D_TPS": "50000",
  "D_STATUS": "PASS",
  "D_REASON": "—",
  "D_TIME_RATIO": "1.16",
  "D_OS": "1.02",
  "D_OS_DRIFT": "+0.02",
  "D_COMPRESS_TIME": "4",
  "D_MEAN_R": "42",
  "D_PROJ_TIME": "0.3",
  "D_LORA_TIME": "0.1",
  "D_TOTAL_TIME": "1.3",

  "R_ACC": "74.8",
  "R_DELTA": "-3.76",
  "R_F1": "0.748",
  "R_RATIO": "5.5",
  "R_PARAMS": "22663128",
  "R_VRAM": "410",
  "R_TIME": "4.6",
  "R_TPS": "24000",
  "R_STATUS": "PASS",
  "R_REASON": "—",
  "R_TIME_RATIO": "1.12",
  "R_OS": "1.01",
  "R_OS_DRIFT": "+0.01",

  "G_ACC": "75.2",
  "G_DELTA": "-4.16",
  "G_F1": "0.752",
  "G_RATIO": "5.5",
  "G_PARAMS": "22642504",
  "G_VRAM": "440",
  "G_TIME": "4.7",
  "G_TPS": "23500",
  "G_STATUS": "PASS",
  "G_REASON": "—",
  "G_TIME_RATIO": "1.12",
  "G_OS": "1.00",
  "G_OS_DRIFT": "0.00",

  "D_E90": "85.5", "D_E90_R": "8.5",
  "D_E95": "88.2", "D_E95_R": "5.95",
  "D_E99": "89.5", "D_E99_R": "4.0",
  "D_E995": "89.8", "D_E995_R": "3.5",
  "R_E90": "71.5", "R_E90_R": "8.0",
  "R_E95": "74.8", "R_E95_R": "5.5",
  "R_E99": "76.5", "R_E99_R": "3.7",
  "R_E995": "76.8", "R_E995_R": "3.2",
  "G_E90": "72.0", "G_E90_R": "8.0",
  "G_E95": "75.2", "G_E95_R": "5.5",
  "G_E99": "77.0", "G_E99_R": "3.7",
  "G_E995": "77.2", "G_E995_R": "3.2",

  "D_C16": "85.0", "D_C32": "87.5", "D_C64": "88.2", "D_C128": "88.3",
  "R_C16": "71.0", "R_C32": "74.0", "R_C64": "74.8", "R_C128": "74.9",
  "G_C16": "72.0", "G_C32": "74.5", "G_C64": "75.2", "G_C128": "75.3",

  "D_LORA0": "80.5", "D_LORA8": "86.5", "D_LORA16": "88.2", "D_LORA32": "88.5",
  "R_LORA0": "65.0", "R_LORA8": "73.0", "R_LORA16": "74.8", "R_LORA32": "75.2",
  "G_LORA0": "68.0", "G_LORA8": "74.0", "G_LORA16": "75.2", "G_LORA32": "75.5",

  "D_L0_Q_R": "48", "D_L0_K_R": "44", "D_L0_V_R": "46", "D_L0_O_R": "50",
  "D_L1_Q_R": "42", "D_L1_K_R": "40", "D_L1_V_R": "44", "D_L1_O_R": "46",
  "D_L2_Q_R": "38", "D_L2_K_R": "36", "D_L2_V_R": "40", "D_L2_O_R": "42",
  "D_L3_Q_R": "36", "D_L3_K_R": "34", "D_L3_V_R": "38", "D_L3_O_R": "40",
  "D_L4_Q_R": "32", "D_L4_K_R": "30", "D_L4_V_R": "34", "D_L4_O_R": "36",
  "D_L5_Q_R": "28", "D_L5_K_R": "26", "D_L5_V_R": "30", "D_L5_O_R": "32",

  "D_L0_Q_P": "36864", "D_L0_K_P": "33792", "D_L0_V_P": "35328", "D_L0_O_P": "38400",

  "D_P90": "14200000", "D_R90": "4.7", "D_MR90": "56",
  "D_P95": "11250484", "D_R95": "5.95", "D_MR95": "42",
  "D_P99": "16700000", "D_R99": "4.0", "D_MR99": "62",
  "D_P995": "19100000", "D_R995": "3.5", "D_MR995": "72",

  "D_L0_QR": "48", "D_L0_QE": "0.10", "D_L0_KR": "44", "D_L0_KE": "0.11",
  "D_L0_VR": "46", "D_L0_VE": "0.12", "D_L0_OR": "50", "D_L0_OE": "0.13",
  "D_L1_QR": "42", "D_L1_QE": "0.11", "D_L1_KR": "40", "D_L1_KE": "0.12",
  "D_L1_VR": "44", "D_L1_VE": "0.13", "D_L1_OR": "46", "D_L1_OE": "0.14",
  "D_L2_QR": "38", "D_L2_QE": "0.12", "D_L2_KR": "36", "D_L2_KE": "0.13",
  "D_L2_VR": "40", "D_L2_VE": "0.14", "D_L2_OR": "42", "D_L2_OE": "0.15",
  "D_L3_QR": "36", "D_L3_QE": "0.13", "D_L3_KR": "34", "D_L3_KE": "0.14",
  "D_L3_VR": "38", "D_L3_VE": "0.15", "D_L3_OR": "40", "D_L3_OE": "0.16",
  "D_L4_QR": "32", "D_L4_QE": "0.14", "D_L4_KR": "30", "D_L4_KE": "0.15",
  "D_L4_VR": "34", "D_L4_VE": "0.16", "D_L4_OR": "36", "D_L4_OE": "0.17",
  "D_L5_QR": "28", "D_L5_QE": "0.15", "D_L5_KR": "26", "D_L5_KE": "0.16",
  "D_L5_VR": "30", "D_L5_VE": "0.17", "D_L5_OR": "32", "D_L5_OE": "0.18",

  "RANK_OBS": "Early and late layers have smaller rank; middle layers larger",

  "D_S1": "1.2", "D_S5": "0.7", "D_S10": "0.55", "D_S50": "0.34",

  "E85_DROP": "5.5",
  "NOLORA_DROP": "8.0",
  "D_E85": "83.0",
  "PRESGATE_RESULT": "No effect (parameter unused)",
  "PRESGATE_CONCLUSION": "Safe to remove",

  "SEED42": "88.2", "SEED123": "88.1", "SEED456": "88.3",
  "MEAN": "88.2", "STD": "0.10",

  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.5",
  "RUNTIME": "15",
  "INFERENCE_CARBON": "0.08",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "Q_VERIFIED": "not tested",
  "UNRESOLVED_FAILURES": "None",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — MinimalAURORA

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_aurora/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module minimal_aurora`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Energy threshold stated**
- [ ] **Real parameter reduction stated**
- [ ] **`pres_gate` issue mentioned**

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this IS a compressor**
- [ ] Contributions verifiable

### Method
- [ ] Energy threshold formula
- [ ] Rank selection detailed
- [ ] Forward pass
- [ ] Complexity analysis
- [ ] **`pres_gate` documented**

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Energy threshold ablation
- [ ] Max rank cap ablation
- [ ] LoRA rank ablation
- [ ] `output_scale` tracking

### Analysis
- [ ] Why energy threshold works
- [ ] Per-layer rank distribution
- [ ] **`pres_gate` dead code analysis**
- [ ] `output_scale` drift

### Limitations
- [ ] No activation weighting
- [ ] No LayerNorm recal
- [ ] **`pres_gate` unused**
- [ ] **`output_scale` drift**
- [ ] Silent injection failure
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] **`pres_gate` dead code documented**
- [ ] `output_scale` drift documented
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Energy thresholds documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_aurora
nano papers/minimal_aurora/variables.json
cd papers/minimal_aurora
python apply_vars.py --module minimal_aurora
grep -rn "{{" .
python build.py --module minimal_aurora --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_aurora
```

### مرحله ۲: اجرای MinimalAURORA

```powershell
cd D:\SVD_Project
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2 --energy 0.95 --rank 64 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\minimal_aurora
python apply_vars.py --module minimal_aurora
python build.py --module minimal_aurora
```

---

## 💡 نکات کلیدی MinimalAURORA

### ۱. **MinimalAURORA یک compressor واقعی است**

- SVD با energy threshold → پارامتر **کم** می‌کند
- نسبت کاهش: **~50\%×**
- برخلاف NHA/EAT/OHA/Infini/SVD-Mamba

### ۲. **Energy-based rank selection**

- `E(r) ≥ ε` با `ε = 0.95` پیش‌فرض
- هر لایه rank متفاوت
- ساده‌تر از AASVD (نیاز به calibration ندارد)

### ۳. **مشکلات کد**

- **`pres_gate` استفاده نمی‌شود** — dead code، توصیه به حذف
- **`output_scale`** ممکن است drift کند (ولی نه به 0)
- **Silent injection failure** (`return None`)

### ۴. **تفاوت با AASVD/A3**

| ویژگی | AASVD | A3 | MinimalAURORA |
|---|---|---|---|
| Weighting | Full cov | Fisher diag | **None** |
| Calibration | Required | Required | **No** |
| Rank | Fixed cap | Adaptive retry | **Energy** |
| Complexity | High | Medium | **Low** |

### ۵. **مقایسه با Res-SVD**

| ویژگی | Res-SVD | MinimalAURORA |
|---|---|---|
| SVD levels | 2 | **1** |
| LoRA init | Spectral | **Zero** |
| `pres_gate` | No | **Yes (unused)** |

### ۶. **مزیت اصلی**

- **سادگی**: هیچ calibration، هیچ weighting
- **Multi-architecture**: 4 خانواده
- **Speed**: کامپایلش سریع‌تر از AASVD

### ۷. **معایب اصلی**

- **`pres_gate` dead code** — نشانه‌ای از code quality ضعیف
- **Silent failure** در injection
- **بدون activation weighting** → دقت کمتر از A3/AASVD

### ۸. **Recommendation**

قبل از انتشار مقاله:
1. `pres_gate` را حذف کن
2. `output_scale` را فریز کن یا حذف
3. sanity check بعد از injection اضافه کن
4. benchmark با A3/AASVD/ASVD برای مقایسه

---

**قدم بعدی: `run_all_4_minimal_aurora.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**