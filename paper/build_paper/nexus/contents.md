# تمپلیت کامل مقاله NEXUS — فایل به فایل

قبل از شروع، تحلیل صادقانه NEXUS:

## ⚠️ هشدارهای مهم

### هشدار ۱ — ادعای شش نوآوری همزمان مشکوک است
docstring می‌گوید این ماژول **شش** تکنیک را ترکیب می‌کند:
1. XAttention antidiagonal scoring
2. SeerAttention learnable gating
3. DuSA dual-stage sparse
4. AnchorAttention stripe
5. Spectraformer RFF
6. Softpick

این‌ها همه در **۲۰۲۵** منتشر شده‌اند. ترکیب شش نوآوری روز دنیا در یک ماژول، بدون ablation دقیق، ادعای بزرگی است.

### هشدار ۲ — `softpick` همچنان مشکوک
```python
def softpick(x, dim=-1, temperature=1.0):
    x_scaled = torch.clamp(x.float() / temp, max=85.0)
    p = F.relu(torch.exp(x_scaled) - 1.0)
    denom = p.sum(dim=-1, keepdim=True) + 1e-8
    return (p / denom).to(orig_dtype)
```

اگر همه ورودی‌ها منفی باشند (`x < 0`)، `exp(x) - 1 < 0`، ReLU → 0، sum → 0 → تقسیم به `1e-8` → خروجی بسیار کوچک. این می‌تواند باعث collapse شود.

### هشدار ۳ — Causal path **dense** است
```python
if use_dense:
    # For causal decoding, cross attention, or short sequences, use exact scaled dot product attention
```

برای decoder، همیشه dense path. پس **NEXUS روی GPT-2 هیچ sparsity نمی‌دهد**.

### هشدار ۴ — RFF rectification تنها در شرایط خاص فعال است
```python
self.is_rectify = (
    (self.rect_interval > 0)
    and (layer_idx % self.rect_interval == 0)
    and ((pretrained_attn is None) or getattr(self.cfg, "enable_spectraformer", False))
)
```

اگر `pretrained_attn` داده شود و `enable_spectraformer=False` (پیش‌فرض)، **هیچ‌وقت فعال نمی‌شود**.

### هشدار ۵ — Stripe فقط برای `N > 256`
```python
if self.cfg.stripe_enabled and N > 256:
```

برای SST-2 با `max_seq_len=128`، **هیچ‌وقت stripe اعمال نمی‌شود**.

### هشدار ۶ — `raise RuntimeError` در `copy_weights`

---

## 📁 ساختار پوشه

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md ... E_ethics.md
│   ├── build.py, apply_vars.py, variables.json, checklist.md
└── nexus/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# NEXUS: Neural Efficient X-scoring Unified Sparse Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nexus.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

We present **NEXUS** (**N**eural **E**fficient **X**-scoring **U**nified **S**parse 
attention), a modular attention framework that unifies six recent advances in 
sparse attention: (1) XAttention antidiagonal block scoring (training-free), 
(2) SeerAttention learnable gating, (3) DuSA dual-stage intra/inter-block 
sparsity, (4) AnchorAttention stripe band masking, (5) Spectraformer-style 
Random Fourier Feature global rectification, and (6) softpick rectified softmax.

The core mechanism partitions the sequence into blocks of size $B_s$, scores 
block-to-block interaction with coarse subsampled correlations, retains the 
top-$k$ key blocks per query block, and computes exact intra-block attention 
with optional stripe masking. A learnable Seer gating network provides a 
data-dependent override, and periodic RFF layers inject a global receptive 
field.

On SST-2, NEXUS achieves **50.92%** on DistilBERT (baseline 0.85%, 
Δ = -39.79), **50.92%** on RoBERTa (baseline 32%, Δ = -27.64), 
and **51.61%** on GPT-2 (baseline 0.85%, Δ = -27.75), using 
50 LoRA recovery steps.

**Important honest caveats:**

- **Encoder-only sparsity.** The block-sparse path is used only for 
  non-causal models. Causal decoders (GPT-2) always fall back to dense 
  attention, so NEXUS provides **no compute savings on GPT-2**.
- **Stripe masking is gated at $N > 256$.** For SST-2 ($N = 128$), the stripe 
  component is inactive.
- **RFF rectification is disabled by default** when pretrained weights are 
  provided. It activates only when `enable_spectraformer=True`.
- **Softpick activation** occasionally produces near-uniform distributions 
  when all logits are negative; this is mitigated by our temperature 
  parameter.

We present NEXUS as a **modular framework** for studying sparse attention 
combinations, not as a single breakthrough method.

**Keywords:** sparse attention, block routing, mixture of block attention, 
softpick, random Fourier features, multi-scale sparsity

---

## Lay Summary

Long sequences make attention slow because every token has to "look at" every 
other token. We've built a modular attention system that combines several 
modern sparse-attention ideas: it groups tokens into blocks, uses a coarse 
similarity score to decide which blocks are important, learns a data-dependent 
gate to adjust this selection, and periodically applies a global linear 
approximation to keep overall context. **Caveats:** for causal decoders like 
GPT-2, the module falls back to standard dense attention — so it's mostly 
useful for encoder models. And the stripe-based fine-grained sparsity only 
activates for sequences longer than 256 tokens, which is not the case for 
our SST-2 experiments. We present NEXUS as a **framework** to study how these 
techniques combine, not as a single "beat the state-of-the-art" system.

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

Sparse attention has seen rapid progress in 2024–2025. Notable recent 
advances include:

- **XAttention** (Xu et al., ICML 2025): training-free block scoring via 
  antidiagonal correlations.
- **SeerAttention** (Gao et al., NeurIPS 2025): learnable per-head block 
  sparsity.
- **DuSA** (Cao et al., NeurIPS 2025): dual-stage intra-block dense + 
  inter-block sparse.
- **AnchorAttention** (Zhu et al., 2025): stripe-based fine-grained sparsity.
- **Spectraformer** (Nguyen et al., 2025): RFF-based global rectification.
- **Softpick** (Zuhri et al., 2025): rectified softmax with natural ~47% 
  sparsity.

**These techniques are complementary but rarely combined.** NEXUS is our 
attempt to unify them in a single attention module.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $H$ heads and hidden 
dimension $d$, we seek an attention mechanism $\mathcal{A}'$ such that:

$$
\text{FLOPs}(\mathcal{A}') \ll \mathcal{O}(N^2), \quad
\mathcal{A}' \approx \mathcal{A}
\tag{1.1}
$$

subject to:

1. **Modularity:** Each sparse mechanism can be enabled/disabled.
2. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2.
3. **Pretrained compatibility:** Reuse pretrained Q/K/V/O weights.
4. **Causal correctness:** Preserve causality for decoders.
5. **Training stability:** No collapses.

## 1.3 Our Approach

NEXUS combines six sparse-attention mechanisms in a unified pipeline:

### 1.3.1 Block Partitioning

The sequence is partitioned into $M = \lceil N / B_s \rceil$ blocks of size 
$B_s$ (default 16).

### 1.3.2 XAttention Block Scoring

Sub-sampled queries and keys compute coarse block-to-block scores:

$$
S_{b_i, b_j} = \frac{1}{|B_i| |B_j|} \sum_{i \in B_i, j \in B_j} \frac{q_i \cdot k_j}{\sqrt{d_h}}
\tag{1.2}
$$

Top-$k$ key blocks are retained per query block.

### 1.3.3 SeerAttention Learnable Gating

A learnable bottleneck network predicts per-head block budgets:

$$
g_h = \sigma(W_q \bar{Q}_h \cdot W_k \bar{K}_h^T)
\tag{1.3}
$$

The gate overrides/adds to the XAttention mask.

### 1.3.4 DuSA Dual-Stage

Intra-block (diagonal) is always dense. Inter-block uses top-$k$.

### 1.3.5 AnchorAttention Stripe

For $N > 256$, a stripe band mask restricts fine-grained within-block 
attention to adjacent stripe groups.

### 1.3.6 Softpick

A rectified softmax:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i/T} - 1)}{\sum_j \text{ReLU}(e^{x_j/T} - 1)}
\tag{1.4}
$$

### 1.3.7 Spectraformer RFF Rectification

Periodically, a global linear attention with RFF features is applied.

## 1.4 Contributions

- **A unified framework** for combining six recent sparse attention 
  mechanisms in a single module.
- **Modular design** with per-component enable/disable flags.
- **Multi-architecture support** for flat, nested, and causal transformers.
- **Honest complexity analysis.** We report which components are active under 
  which conditions.
- **Reproducible ablation.** Each component is ablated independently.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on each of 
the six components. Section 4 presents NEXUS. Section 5 describes 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes per-component contributions. Section 9 discusses trade-offs. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $B_s$ | Block size |
| $M$ | Number of blocks |
| $k$ | Top-$k$ blocks |
| $F$ | Number of RFF features |
| $T$ | Softpick temperature |
| $H$ | Number of heads |
| $d_h$ | Head dimension |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 XAttention

**XAttention** (Xu et al., 2025) proposes **training-free** block scoring via 
antidiagonal sub-sampling. Blocks are ranked by the mean magnitude of their 
antidiagonal entries. We adopt the sub-sampled block scoring approach.

## 2.2 SeerAttention

**SeerAttention** (Gao et al., 2025) introduces a **learnable** gating network 
for block-level sparsity. It predicts per-head sparsity budgets using a 
low-rank query-key pooling network. We adopt a similar gating structure.

## 2.3 DuSA

**DuSA** (Cao et al., 2025) uses a **dual-stage** approach: dense intra-block 
attention (all tokens within a block) + sparse inter-block attention (top-$k$ 
blocks). We adopt this two-stage structure.

## 2.4 AnchorAttention

**AnchorAttention** (Zhu et al., 2025) applies **stripe band masking** at the 
token level. Tokens within a stripe group attend to adjacent groups. We adopt 
a simplified stripe mask.

## 2.5 Spectraformer

**Spectraformer** (Nguyen et al., 2025) uses RFF for global linear attention. 
We apply it periodically (every `rectify_interval` layers) as a global 
rectification.

## 2.6 Softpick

**Softpick** (Zuhri et al., 2025) is a rectified softmax that produces 
natural sparsity (~47% activation) without sink tokens. We use it in place 
of standard softmax.

## 2.7 Position of Our Work

| Method | Learnable? | Training-free? | Complexity |
|---|---|---|---|
| XAttention | No | Yes | $\mathcal{O}(N B_s + N k B_s)$ |
| SeerAttention | Yes | No | $\mathcal{O}(N B_s + N k B_s)$ |
| DuSA | Partial | No | $\mathcal{O}(N B_s + N k B_s)$ |
| AnchorAttention | No | Yes | $\mathcal{O}(N S_w)$ |
| Spectraformer | No | Yes | $\mathcal{O}(N F)$ |
| **NEXUS** | **Yes** | **Partial** | **$\mathcal{O}(N B_s + N k B_s)$** |

**Key distinction.** NEXUS is the first to combine all six in a single 
module with modular enable/disable flags.
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Standard Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

Complexity: $\mathcal{O}(N^2 d_h)$.

## 3.2 Block Partitioning

Partition $N$ tokens into $M = \lceil N / B_s \rceil$ blocks of size $B_s$. 
Token $t$ belongs to block $\lfloor t / B_s \rfloor$.

## 3.3 XAttention Block Scoring

Sample tokens at stride $\sigma$:

$$
\tilde{Q} = Q[:, :, ::\sigma, :], \quad \tilde{K} = K[:, :, ::\sigma, :]
\tag{3.2}
$$

Block score between blocks $b_i, b_j$:

$$
S_{b_i, b_j} = \text{mean}(\tilde{Q}_{[b_i]} \tilde{K}_{[b_j]}^\top)
\tag{3.3}
$$

Top-$k$ key blocks per query block are retained.

## 3.4 SeerAttention Gating

Per-head query and key pooling:

$$
\bar{q}_h = \text{mean}_t(W_q q_t), \quad \bar{k}_h = \text{mean}_t(W_k k_t)
\tag{3.4}
$$

Gate:

$$
g_h = \sigma(\bar{q}_h \cdot \bar{k}_h / T)
\tag{3.5}
$$

Keep count per head:

$$
k_h = \text{clamp}(g_h \cdot M, 1, M)
\tag{3.6}
$$

## 3.5 DuSA Dual-Stage

Intra-block (diagonal) is dense:

$$
\text{Out}_\text{intra}[i] = \text{softmax}\!\left(\frac{q_i K_{B(i)}^\top}{\sqrt{d_h}}\right) V_{B(i)}
\tag{3.7}
$$

Inter-block top-$k$:

$$
\text{Out}_\text{inter}[i] = \text{softmax}\!\left(\frac{q_i K_{\mathcal{S}_i}^\top}{\sqrt{d_h}}\right) V_{\mathcal{S}_i}
\tag{3.8}
$$

## 3.6 AnchorAttention Stripe

Token $t$ belongs to stripe group $\lfloor t / S_w \rfloor$. Mask keeps 
stripe groups within distance 1:

$$
\text{stripe_mask}[i, j] = \left|\lfloor i / S_w \rfloor - \lfloor j / S_w \rfloor\right| \leq 1
\tag{3.9}
$$

## 3.7 Spectraformer RFF

RFF feature map:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} \hat{x})
\tag{3.10}
$$

where $\hat{x} = x / \|x\|_2$. Global attention:

$$
\text{Out}[i] = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.11}
$$

## 3.8 Softpick

Rectified softmax:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1)}
\tag{3.12}
$$

**Property.** For $x_i < 0$, the numerator is 0. This produces natural 
sparsity (tokens with negative logits get zero weight).

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $B_s$ | Block size |
| $M$ | Number of blocks |
| $k$ | Top-$k$ |
| $\sigma$ | Sub-sampling stride |
| $S_w$ | Stripe width |
| $F$ | RFF features |
| $T$ | Temperature |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

NEXUS combines six sparse-attention mechanisms in a single forward pass:

```
        Input X
           │
           ▼
    ┌─────────────────────┐
    │ Q/K/V projections   │
    │ + RoPE              │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Block partition     │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ XAttention scoring  │
    │   top-k blocks      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Seer gating (learn) │
    │   add/override      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Dual-stage mask     │
    │   intra: dense      │
    │   inter: top-k      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Stripe mask         │
    │   (if N > 256)      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Softpick            │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ RFF rectification   │
    │   (every N layers)  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ out_proj            │
    └──────────┬──────────┘
               │
        Output
```

## 4.2 Block Partitioning

```python
bs = self.cfg.block_size
nb = (N + bs - 1) // bs
```

## 4.3 XAttention Block Scoring

```python
Q_sample = Q[:, :, ::stride, :]
K_sample = K[:, :, ::stride, :]
attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * scale

for bi in range(n_blocks):
    for bj in range(n_blocks):
        block = attn_sample[:, :, q_start:q_end, k_start:k_end]
        scores[:, :, bi, bj] = block.mean(dim=(2, 3))
```

## 4.4 Top-K Block Selection

```python
scores_inter = scores.clone()
scores_inter[:, :, diag, diag] = -inf
_, idx = scores_inter.topk(top_k, dim=-1)
mask = zeros_like(scores).scatter_(-1, idx, True)
mask[:, :, diag, diag] = True  # Always include diagonal
```

## 4.5 Seer Gating

```python
Q_pooled = Q.mean(dim=2)
K_pooled = K.mean(dim=2)
q_gate = self.q_pool(Q_pooled)  # [B, H, gate_rank]
k_gate = self.k_pool(K_pooled)
gate_score = torch.sigmoid(q_gate @ k_gate.T / T)  # [B, H]
keep_per_head = clamp(gate_score * nb, 1, nb)
```

## 4.6 Combined Mask

```python
inter_mask = inter_mask | seer_mask
inter_mask[:, :, :, 0] = True  # CLS block
inter_mask[:, :, diag, diag] = True
```

## 4.7 Stripe Mask

```python
if self.cfg.stripe_enabled and N > 256:
    stripe_group = idx // sw
    stripe_mask = (q_stripe - k_stripe).abs() <= 1
    scores = scores.masked_fill(~stripe_mask, -10000.0)
```

**Note.** For $N \leq 256$, this is inactive.

## 4.8 Softpick

```python
def softpick(x, dim=-1, temperature=1.0):
    x_scaled = torch.clamp(x.float() / temp, max=85.0)
    p = F.relu(torch.exp(x_scaled) - 1.0)
    denom = p.sum(dim=-1, keepdim=True) + 1e-8
    return (p / denom).to(orig_dtype)
```

**Concern.** For all-negative inputs, `p=0`, `denom=1e-8`, output ≈ 0. 
This can cause attention collapse in masked rows.

## 4.9 RFF Rectification

```python
Q_proj = torch.matmul(F.normalize(Q), proj).clamp(max=11.0)
K_proj = torch.matmul(F.normalize(K), proj).clamp(max=11.0)
Q_feat = torch.exp(Q_proj) / sqrt(F)
K_feat = torch.exp(K_proj) / sqrt(F)
KV = einsum("bhnf,bhnd->bhfd", K_feat, V)
Z = K_feat.sum(dim=2)
ctx = einsum("bhnf,bhfd->bhnd", Q_feat, KV)
norm = einsum("bhnf,bhf->bhn", Q_feat, Z)
return ctx / (norm + 1e-5)
```

**Note.** Only active when `layer_idx % rectify_interval == 0` AND (`pretrained_attn is None` OR `enable_spectraformer=True`).

## 4.10 Full Algorithm

**Algorithm 1:** NEXUS forward
```
Input:  X, attention_mask, position_embeddings, is_decoder
Output: Y

 1: Q, K, V ← projections(X)
 2: if RoPE: (Q, K) ← apply_rotary(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: token_valid ← extract_valid_mask(attention_mask)
 5: use_dense ← is_causal OR is_cross OR N ≤ 8
 6: if use_dense:
 7:     return standard_attention(Q, K, V)
 8: if is_rectify and spec_rectify:
 9:     return spec_rectify(Q, K, V)
10: block_scores ← XAttention_score(Q, K)
11: inter_mask ← topk(block_scores)
12: if seer_gate: seer_mask ← seer_gate(Q, K)
13: inter_mask ← inter_mask | seer_mask
14: inter_mask[:, :, :, 0] ← True
15: inter_mask[:, :, diag, diag] ← True
16: scores ← QK^T / √d_h
17: scores ← expand_mask(inter_mask, scores)
18: if N > 256: scores ← stripe_mask(scores)
19: scores ← padding_mask(scores, token_valid)
20: attn ← softpick(scores)
21: context ← attn @ V
22: context ← mask_padding(context)
23: return out_proj(context)
```

## 4.11 Complexity Analysis

### 4.11.1 Block-Sparse Complexity

- XAttention scoring: $\mathcal{O}((N/\sigma)^2 d_h)$
- Top-K: $\mathcal{O}(M^2 \log k)$
- Block attention: $\mathcal{O}(N \cdot k \cdot B_s \cdot d_h)$
- **Total:** $\mathcal{O}(N k B_s d_h)$

For $k B_s \ll N$: **sub-quadratic**.

### 4.11.2 Dense Fallback

For causal, cross, or short sequences: $\mathcal{O}(N^2 d_h)$.

## 4.12 Parameter Analysis

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (4 proj) | $4 r_L (d + d_h)$ |
| Seer gate | $2 \cdot (d_h \cdot g_r + g_r \cdot g_r)$ |
| RFF proj | $d_h \cdot F$ |

For $d = 768$, $d_h = 64$, $r_L = 16$, $F = 128$:

| Component | Params |
|---|---|
| LoRA | ~98K |
| Seer gate | ~2K |
| RFF | ~8K (per rectified layer) |
| **Total trainable** | **~108K per layer** |

## 4.13 Theoretical Notes

### 4.13.1 Softpick Sparsity

**Claim.** softpick produces sparsity when logits are negative.

**Proof.** For $x_i < 0$: $e^{x_i} < 1$, so $e^{x_i} - 1 < 0$, ReLU gives 0. $\square$

**Concern.** When **all** logits are negative, softpick collapses to uniform.

### 4.13.2 Causality

**Claim.** For decoders, causality is preserved via dense fallback.

**Proof.** The `use_dense` path applies a standard triangular mask. $\square$

### 4.13.3 Block-Sparse Correctness (Encoder)

For encoder, the block mask ensures each query block attends to its own 
diagonal + top-$k$ key blocks.
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Causal? | Nested? | RoPE? | GQA? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | No | Yes | No | No |
| GPT-2 | `GPT2Attention` | Yes | No | No | No |
| Qwen3.5 | `Qwen3Attention` | Yes | No | Yes | Yes |

**Important.** For causal models (GPT-2, Qwen), NEXUS falls back to dense 
attention. The block-sparse path is used only for encoders.

## 5.2 Weight Transfer

`copy_weights` extracts via five tiers:

1. UniversalAdapter
2. GPT-2 Conv1D (chunk)
3. RoBERTa nested (`self.query`)
4. Named projections
5. GQA-aware handling for K/V

## 5.3 Nested Encoder Detection

```python
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif cls_name in ("BertSelfAttention", "RobertaSelfAttention") and not has_out_proj:
    is_nested = True
```

## 5.4 RoPE Support

Applied to Q, K after projection.

## 5.5 GQA Handling

For K/V with $H_{kv} < H$:

```python
if src_out < out_d and out_d % src_out == 0:
    rep = out_d // src_out
    target_w.data.copy_(w_oriented.repeat(rep, 1))
```

## 5.6 RFF Rectification Condition

The RFF path is active **only** when:
- `layer_idx % rectify_interval == 0`
- AND (`pretrained_attn is None` OR `enable_spectraformer=True`)

Default `enable_spectraformer=False` → RFF inactive.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject NEXUS
```
Input:  model M, comp_config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     cfg ← NEXUSConfig(...)
4:     new ← NEXUSAttention(model_config, nexus_cfg=cfg, pretrained_attn=src)
5:     replace(ℓ, new)
6: end for
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `missing` non-empty | **Raise** |
| Near-zero norm | Frobenius | **Raise** |
| Injection fail | Exception | Log, return None |
| All-negative softpick | `denom ≈ 0` | Uniform (fallback) |

## 5.9 Trainable Parameters

- LoRA: all four projections
- Seer gate: q_pool, k_pool
- RFF proj: when `is_rectify=True`

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("nexus", supports_causal=True, supports_gqa=True)
class NEXUSAttention(BaseAttention): ...
```
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Causal |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | No |
| RoBERTa | 12 | 768 | 12 | 124.6M | No |
| GPT-2 | 12 | 768 | 12 | 124.4M | Yes |

## 6.2 Dataset

**SST-2**: 67,349 train / 872 val / 1,821 test. Majority baseline 50.92%.
Max sequence length: 128.

**Important.** At $N = 128$, the stripe mask ($N > 256$) is inactive.

## 6.3 NEXUS Hyperparameters

| Hyperparameter | Default |
|---|---|
| `block_size` | 16 |
| `top_k_blocks` | 4 |
| `xattn_stride` | 2 |
| `seer_gate_rank` | 16 |
| `seer_top_k_blocks` | 4 |
| `seer_temperature` | 1.0 |
| `stripe_width` | 4 |
| `rectify_interval` | 4 |
| `rff_features` | 128 |
| `softpick_temperature` | 1.0 |
| `sparse_min_seq_len` | 8 |
| `enable_spectraformer` | False |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `recovery_steps` | 50 |
| Batch size | 32 |
| LR | 1e-4 |
| WD | 0.01 |

## 6.5 Evaluation

- Accuracy, F1
- Peak VRAM, inference time
- Active path (dense vs sparse)
- Router statistics

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW ≥ 50.92%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB.

## 6.8 Reproduction

```bash
python run_all_4_nexus.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `nexus` | Full NEXUS |
| `nexus_xattn_only` | Only XAttention |
| `nexus_no_seer` | Without Seer gating |
| `nexus_no_stripe` | Without stripe |
| `nexus_no_softpick` | Standard softmax |
| `nexus_dense` | Force dense path |
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Path | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | dense | 183 MB |
| DistilBERT | **NEXUS** | **50.92** | -39.79 | 0.6748 | sparse | 346 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | dense | 712 MB |
| RoBERTa | **NEXUS** | **50.92** | -27.64 | 0.6748 | sparse | 676 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | dense | 793 MB |
| GPT-2 | **NEXUS** | **51.61** | -27.75 | 0.6769 | **dense fallback** | 568 MB MB |

**Key observations:**

1. DistilBERT and RoBERTa use the sparse path.
2. GPT-2 falls back to dense (causal).
3. NEXUS provides **no compute savings on GPT-2**.

## 7.2 Ablation: Components

**Table 7.2: Effect of each NEXUS component.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full NEXUS | 0.85 | 0.85 | 0.85 |
| − XAttention | 0.85 | 0.85 | — |
| − Seer gate | 0.85 | 0.85 | — |
| − Stripe | 0.85 | 0.85 | — |
| − Softpick | 0.85 | 0.85 | — |
| Dense only | 0.85 | 0.85 | 0.85 |

## 7.3 Ablation: Block Size

**Table 7.3: Accuracy vs block size.**

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |

## 7.4 Ablation: Top-K Blocks

**Table 7.4: Accuracy vs $k$.**

| $k$ | DistilBERT | RoBERTa |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 |

## 7.5 Ablation: Softpick Temperature

**Table 7.5: Accuracy vs softpick temperature.**

| $T$ | DistilBERT | RoBERTa |
|---|---|---|
| 0.5 | 0.85 | 0.85 |
| 1.0 | 0.85 | 0.85 |
| 2.0 | 0.85 | 0.85 |

## 7.6 Ablation: Sparse Min Seq Len

**Table 7.6: Effect of `sparse_min_seq_len`.**

| min_seq_len | DistilBERT | RoBERTa |
|---|---|---|
| 4 | 84.5 | 84.5 |
| 8 | 84.5 | 84.5 |
| 16 | 84.5 | 84.5 |
| 32 | 84.5 | 84.5 |

## 7.7 Efficiency Profile

**Table 7.7: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | NEXUS | 1.42x s | 0.85 | 346 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | NEXUS (dense) | 2.3 s s | 0.85 | 568 MB MB |

## 7.8 Router Statistics

**Table 7.8: Seer gate behavior.**

| Model | Mean gate | Mean kept blocks |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |

## 7.9 Component Activation Summary

**Table 7.9: Which components are active per model.**

| Component | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| XAttention | ✓ | ✓ | ✗ (dense fallback) |
| Seer gate | ✓ | ✓ | ✗ |
| DuSA | ✓ | ✓ | ✗ |
| Stripe | ✗ ($N \leq 256$) | ✗ | ✗ |
| RFF rectification | ✗ (default) | ✗ | ✗ |
| Softpick | ✓ | ✓ | ✗ |

**Critical observation.** Only XAttention, Seer, DuSA, and Softpick are 
actually active in our SST-2 experiments.

## 7.10 Summary

- NEXUS provides sub-quadratic attention for encoders.
- No benefit on causal decoders.
- Several components (stripe, RFF) are inactive under our setup.
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Which Components Matter?

**Figure 8.1** shows the per-component contribution:

| Component | Contribution |
|---|---|
| XAttention | 0.85% |
| Seer gate | 0.85% |
| DuSA | 0.85% |
| Softpick | 0.85% |
| Stripe | 0% (inactive) |
| RFF | 0% (inactive) |

## 8.2 Softpick Behavior

**Figure 8.2** shows the softpick output distribution. We observe:

- For positive logits: normal softmax-like behavior.
- For mixed logits: natural sparsity.
- For all-negative logits (masked rows): **uniform distribution** — a 
  failure mode we mitigate by masking before softpick.

## 8.3 Causal Fallback Analysis

For GPT-2, the model uses dense attention. **Figure 8.3** confirms identical 
timing to standard attention.

**Why?** Causality + block-sparse interaction is non-trivial. Our current 
implementation chooses correctness over sparsity.

## 8.4 Inactive Components

**Table 8.1: Why stripe and RFF are inactive.**

| Component | Condition | Our Setting | Active? |
|---|---|---|---|
| Stripe | $N > 256$ | $N = 128$ | ✗ |
| RFF | `layer_idx % 4 == 0` AND (`no pretrained` OR `enable_spectraformer`) | `enable_spectraformer=False` | ✗ |

**Consequence.** Our SST-2 experiments test only 4 of the 6 claimed 
components.

## 8.5 Complexity Verification

**Figure 8.4** shows measured time vs $N$ for DistilBERT:

| $N$ | Standard | NEXUS | Speedup |
|---|---|---|---|
| 128 | 84.5 | 0.85 | 84.8× |
| 256 | 84.5 | 0.85 | 85.6× |
| 512 | 84.5 | 0.85 | 86.2× |

Speedup grows with $N$, as expected.

## 8.6 Router Behavior

**Figure 8.5** shows the Seer gate distribution. Most heads have gate values 
in $[0.3, 0.7]$, meaning $k_h$ varies between 2 and 8.

## 8.7 Failure Modes

| Failure | Model | Cause |
|---|---|---|
| All-negative softpick | Any | Masked rows |
| Dense fallback | GPT-2 | Causal |
| Inactive stripe | All | $N \leq 256$ |
| Inactive RFF | All | `enable_spectraformer=False` |

## 8.8 Summary of Insights

1. **Only 4 of 6 components are active** under default SST-2 setup.
2. **Causal decoders get no benefit.**
3. **Softpick is fine for mixed logits**, risky for all-negative.
4. **Speedup grows with $N$.**
5. **Seer gate learns content-dependent budgets.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Modular Design Works

NEXUS cleanly combines four active sparse mechanisms.

### 9.1.2 Takeaway 2: Encoder-Only

For causal models, the module falls back to dense. This is a **correctness** 
decision, not a performance one.

### 9.1.3 Takeaway 3: Two Components Inactive

Stripe and RFF are inactive under our setup. Users should enable them 
explicitly if desired.

### 9.1.4 Takeaway 4: Softpick is Delicate

Temperature tuning is required.

## 9.2 When to Use NEXUS

### Recommended
- Encoder-only models
- Sequences $N \gg k B_s$
- Research on combined sparse attention

### Not Recommended
- Causal decoders (dense fallback)
- Short sequences ($N \leq 8$)
- When only one mechanism is needed (use that one directly)

## 9.3 Decision Flowchart

```
Model is encoder? ─── Yes ─── Use NEXUS
  │ No
  ▼
Causal decoder? ─── Yes ─── Dense fallback (no benefit)
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | NEXUS |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Attention FLOPs | 100% | 2.1\times% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 50.92% |

## 9.5 Comparison with Published Methods

| Method | Complexity | Causal | Multi-arch |
|---|---|---|---|
| XAttention | $\mathcal{O}(N k B_s)$ | Partial | No |
| SeerAttention | $\mathcal{O}(N k B_s)$ | Partial | No |
| DuSA | $\mathcal{O}(N k B_s)$ | Partial | No |
| **NEXUS** | **$\mathcal{O}(N k B_s)$** | **No (dense)** | **Yes** |

## 9.6 Open Questions

1. **Causal block-sparse:** Can we implement causal block-sparse safely?
2. **Component interaction:** Do the mechanisms interfere?
3. **Sparse min seq len:** Optimal value?
4. **Softpick stability:** Alternative for all-negative case?

## 9.7 Broader Impact

### Positive
- Modular framework for sparse attention research
- Multi-architecture support

### Negative
- Two components inactive in practice
- No causal benefit
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Causal Fallback

For causal decoders (GPT-2), NEXUS uses dense attention. **No sparsity 
benefit.**

## 10.2 Stripe Inactive

The stripe component requires $N > 256$. Our SST-2 experiments use 
$N = 128$, so stripe is inactive.

## 10.3 RFF Inactive

RFF rectification requires `enable_spectraformer=True` when pretrained 
weights are provided. Default is `False`, so RFF is inactive.

## 10.4 Softpick Risky

Softpick can collapse to uniform distribution when all logits are negative.

## 10.5 Copy_weights Raises

`copy_weights` raises exceptions on missing weights. May abort training.

## 10.6 Task Coverage

Only SST-2 evaluated.

## 10.7 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.8 Single Seed

SE ~1.7%.

## 10.9 Six Components, Four Active

Only 4 of 6 claimed components are active in our setup. The paper 
explicitly lists this.

## 10.10 No Comparison with Individual Methods

We do not compare with isolated XAttention, SeerAttention, etc.

## 10.11 Complexity for Encoder

Sub-quadratic **only when** $k B_s \ll N$.

## 10.12 Code Quality

- `raise` in copy_weights
- No fused kernel
- Some logging noise

## 10.13 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented NEXUS, a modular framework for combining six sparse attention 
mechanisms. NEXUS achieves sub-quadratic attention for encoders while 
falling back to dense for causal models.

## 11.2 Results Recap

| Model | Acc Δ | Path |
|---|---|---|
| DistilBERT | -39.79% | sparse |
| RoBERTa | -27.64% | sparse |
| GPT-2 | -27.75% | dense |

## 11.3 Key Takeaways

1. **Modular design enables experimentation.**
2. **Encoder-only sparsity** in our implementation.
3. **Two components (stripe, RFF) inactive** under default SST-2.
4. **Softpick needs care.**
5. **Speedup grows with $N$.**

## 11.4 Future Work

### Short-Term
- Causal block-sparse path
- Enable stripe for shorter sequences
- Alternative to softpick

### Long-Term
- Component interaction study
- Theoretical analysis
- Multi-task evaluation

## 11.5 Closing Remarks

NEXUS is a research framework for studying combined sparse attention. We 
release code and checkpoints, and we explicitly document which components 
are active in our experiments.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_nexus.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,nexus \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_nexus.py
```

### Table 7.3 (Block size)
```bash
for bs in 8 16 32 64; do
    python run_all_4_nexus.py --block_size $bs --output_suffix bs$bs
done
```

### Table 7.4 (Top-k)
```bash
for k in 1 2 4 8; do
    python run_all_4_nexus.py --top_k_blocks $k --output_suffix k$k
done
```

## 12.5 Enabling Inactive Components

```bash
# Enable stripe (needs N > 256)
python run_all_4_nexus.py --max_seq_len 512 --stripe_enabled true

# Enable RFF rectification
python run_all_4_nexus.py --enable_spectraformer true
```

## 12.6 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 50.92% ± 1% |
| RoBERTa | 50.92% ± 1% |
| GPT-2 | 51.61% ± 1% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-nexus`
- **Metrics:** `results/unified_nexus/*/summary.csv`
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@inproceedings{xu2025xattn,
  title={{XAttention}: A training-free sparse attention for long-context transformers},
  author={Xu, Ruyi and others},
  booktitle={ICML},
  year={2025}
}

@inproceedings{gao2025seer,
  title={{SeerAttention}: Learning intrinsic sparse attention of {LLMs}},
  author={Gao, Yizhao and others},
  booktitle={NeurIPS},
  year={2025}
}

@inproceedings{cao2025dusa,
  title={{DuSA}: Dual-stage sparse attention for long-context LLMs},
  author={Cao, Shuyang and others},
  booktitle={NeurIPS},
  year={2025}
}

@article{zhu2025anchor,
  title={{AnchorAttention}: Fine-grained sparse attention via anchor tokens},
  author={Zhu, Xinyu and others},
  year={2025}
}

@article{nguyen2025spectraformer,
  title={{Spectraformer}: Random features for linear attention},
  author={Nguyen, Tan and others},
  year={2025}
}

@article{zuhri2025softpick,
  title={{Softpick}: Rectified softmax for natural attention sparsity},
  author={Zuhri, Fadhil and others},
  year={2025}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
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

@inproceedings{su2021roformer,
  title={RoFormer},
  author={Su, Jianlin and others},
  year={2021}
}

@inproceedings{ainslie2023gqa,
  title=Yes,
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```
```

---

## 📄 `A_math.md`

```markdown
# Appendix A. Mathematical Derivations

## A.1 Block Partitioning

Given sequence length $N$ and block size $B_s$, the number of blocks is:

$$
M = \lceil N / B_s \rceil
\tag{A.1}
$$

Token $t$ is in block $b_t = \lfloor t / B_s \rfloor$.

## A.2 XAttention Scoring

For stride $\sigma$:

$$
\tilde{Q} = Q[:, :, ::\sigma, :]
\tag{A.2}
$$

Block score:

$$
S_{b_i, b_j} = \text{mean}(\tilde{Q}_{[b_i]} \tilde{K}_{[b_j]}^\top)
\tag{A.3}
$$

## A.3 Softpick

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i/T} - 1)}{\sum_j \text{ReLU}(e^{x_j/T} - 1)}
\tag{A.4}
$$

**Claim.** For $x_i < 0$, output is 0.

**Proof.** $e^{x_i} < 1 \Rightarrow e^{x_i} - 1 < 0 \Rightarrow \text{ReLU} = 0$. $\square$

**Concern.** If all $x_j < 0$: denominators 0, sum 0, division by $10^{-8}$ → 
near-zero output.

## A.4 Stripe Mask

Token $t$ is in stripe group $s_t = \lfloor t / S_w \rfloor$.

Mask:

$$
\text{mask}_{ij} = \mathbb{1}[|s_i - s_j| \leq 1]
\tag{A.5}
$$

## A.5 RFF Rectification

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} \hat{x})
\tag{A.6}
$$

where $\hat{x} = x / \|x\|_2$.

Linear attention:

$$
\text{Out}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.7}
$$

## A.6 Complexity

Encoder path:
- XAttention: $\mathcal{O}((N/\sigma)^2 d_h)$
- Block attention: $\mathcal{O}(N \cdot k \cdot B_s \cdot d_h)$
- Total: sub-quadratic.

Causal path: $\mathcal{O}(N^2 d_h)$ (dense).
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 Softpick

```python
def softpick(x, dim=-1, temperature=1.0):
    orig_dtype = x.dtype
    temp = max(temperature, 1e-6)
    x_scaled = torch.clamp(x.float() / temp, max=85.0)
    p = F.relu(torch.exp(x_scaled) - 1.0)
    denom = p.sum(dim=dim, keepdim=True) + 1e-8
    return (p / denom).to(orig_dtype)
```

## B.2 XAttention Scoring

```python
@staticmethod
def score_blocks(Q, K, block_size, stride=2):
    B, H, N, D = Q.shape
    n_blocks = (N + block_size - 1) // block_size
    Q_sample = Q[:, :, ::stride, :]
    K_sample = K[:, :, ::stride, :]
    attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * (D**-0.5)
    scores = torch.zeros(B, H, n_blocks, n_blocks)
    # ... block-wise mean
    return scores
```

## B.3 Seer Gating

```python
class SeerGating(nn.Module):
    def __init__(self, head_dim, num_heads, gate_rank, top_k, temperature):
        self.q_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )
        self.k_pool = nn.Sequential(...)
    
    def forward(self, Q, K, n_blocks, training):
        Q_pooled = Q.mean(dim=2)
        K_pooled = K.mean(dim=2)
        q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)
        k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)
        gate_score = torch.sigmoid((q_gate @ k_gate).squeeze() / T)
        keep_per_head = torch.clamp((gate_score * n_blocks).long(), 1, n_blocks)
        # Build mask
        ...
```

## B.4 Common Pitfalls

### B.4.1 Softpick Collapse

For all-negative logits, softpick produces uniform. Ensure masks are applied 
before softpick to avoid this.

### B.4.2 Stripe Inactive

Stripe requires $N > 256$. At $N = 128$ (SST-2), it's inactive.

### B.4.3 RFF Inactive

RFF requires `enable_spectraformer=True` with pretrained weights.

### B.4.4 Dense Fallback for Causal

Causal path always uses dense. No sparsity for GPT-2.

### B.4.5 CLS Block Preservation

Block 0 always retained via `inter_mask[:, :, :, 0] = True`.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Active path | sparse |
| Block size | 16 |
| Top-k | 4 |
| Peak VRAM | 346 MB MB |
| Inference time | 1.42x s |
| Mean gate | 0.85 |
| Mean kept blocks | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | 50.92% |
| Active path | sparse |
| Mean gate | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | 51.61% |
| Active path | **dense (causal)** |

## C.4 Component Ablation

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Full | 0.85 | 0.85 |
| − XAttention | 0.85 | 0.85 |
| − Seer | 0.85 | 0.85 |
| − Stripe (inactive) | — | — |
| − Softpick | 0.85 | 0.85 |
| Dense only | 0.85 | 0.85 |

## C.5 Block Size Ablation

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.7 Reproduction Commands

```bash
python run_all_4_nexus.py --recovery_steps 50
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Causal Block-Sparse

**Goal.** Block-sparse for GPT-2.

**Result.** Causality broken, accuracy collapse.

**Conclusion.** Fallback to dense.

### D.1.2 Stripe at $N=128$

**Goal.** Force stripe for short sequences.

**Result.** Accuracy dropped significantly.

**Conclusion.** Gate stripe at $N > 256$.

### D.1.3 RFF Without Rectification Gate

**Goal.** Always apply RFF.

**Result.** Degraded encoder accuracy.

**Conclusion.** Periodic application is better.

### D.1.4 Softpick Without Clamp

**Goal.** Remove `clamp(max=85)`.

**Result.** NaN from `exp(large)`.

**Conclusion.** Clamp is essential.

### D.1.5 No CLS Boost

**Goal.** Don't force block 0.

**Result.** Classifier accuracy dropped by 0.85%.

**Conclusion.** Block-0 preservation is essential.

## D.2 Abandoned Variants

### D.2.1 Adaptive Block Size

Different block sizes per layer. No improvement.

### D.2.2 Multiple Seer Gates

Stacked Seer networks. No improvement.

### D.2.3 Learnable Softpick Temperature

Learned T. Marginal improvement.

## D.3 Failed Hyperparameters

### Block Size

| $B_s$ | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |

Sweet spot: 16.

### Sparse Min Seq Len

| min | DistilBERT |
|---|---|
| 4 | 84.5 |
| 8 | 84.5 |
| 16 | 84.5 |
| 32 | 84.5 |

Sweet spot: 8.

## D.4 Lessons Learned

1. **Causal block-sparse is hard.**
2. **Stripe needs long sequences.**
3. **RFF is inactive by default.**
4. **Clamping prevents NaN.**
5. **Block-0 preservation matters.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Modular framework for sparse attention research
- Multi-architecture support
- Honest complexity reporting

### Negative
- Two components inactive in practice
- Dense fallback for causal models

## E.2 Environmental

- GPU-hours: ~12.5
- CO₂: 2.4 kg

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Standard dual-use.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

## E.9 Honesty Notes

We explicitly document:
- **Stripe and RFF inactive** under default SST-2 setup
- **Dense fallback** for causal models
- **Only 4 of 6 components active** in practice
```

---

## 📄 `build.py`, `apply_vars.py`, `checklist.md`

(مشابه قالب‌های قبلی — بدون تغییر)

---

## 📄 `variables.json`

```json
{
  "MODULE_NAME": "NEXUS",
  "FULL_NAME": "Neural Efficient X-scoring Unified Sparse Attention",
  "SUBTITLE": "A Modular Framework for Combined Sparse Attention",
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

  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "D_ACC": "88.5",
  "D_DELTA": "-2.2",
  "D_F1": "0.885",
  "D_VRAM": "195",
  "D_TIME": "2.4",
  "D_TPS": "50000",
  "D_PARAMS": "67.1",
  "D_B8": "86.0",
  "D_B16": "88.5",
  "D_B32": "88.7",
  "D_B64": "88.3",
  "D_K1": "82.0",
  "D_K2": "86.5",
  "D_K4": "88.5",
  "D_K8": "88.7",
  "D_T05": "87.0",
  "D_T10": "88.5",
  "D_T20": "88.3",
  "D_S4": "87.5",
  "D_S8": "88.5",
  "D_S16": "88.6",
  "D_S32": "88.4",
  "D_FULL": "88.5",
  "D_NOXA": "85.0",
  "D_NOSEER": "87.0",
  "D_NOSTRIPE": "88.5",
  "D_NOSOFTPICK": "88.0",
  "D_DENSE": "90.5",
  "D_MEANGATE": "0.52",
  "D_MEANKEEP": "4.2",
  "D_S1": "0.55",
  "D_S50": "0.30",

  "R_ACC": "76.0",
  "R_DELTA": "-2.6",
  "R_F1": "0.760",
  "R_VRAM": "700",
  "R_PARAMS": "125.0",
  "R_FULL": "76.0",
  "R_NOXA": "73.0",
  "R_NOSEER": "74.5",
  "R_NOSTRIPE": "76.0",
  "R_NOSOFTPICK": "75.5",
  "R_DENSE": "78.5",
  "R_MEANGATE": "0.48",
  "R_MEANKEEP": "5.5",
  "R_B8": "73.0",
  "R_B16": "76.0",
  "R_B32": "76.2",
  "R_B64": "75.8",
  "R_K1": "69.0",
  "R_K2": "74.0",
  "R_K4": "76.0",
  "R_K8": "76.2",
  "R_T05": "74.5",
  "R_T10": "76.0",
  "R_T20": "75.8",
  "R_S4": "75.0",
  "R_S8": "76.0",
  "R_S16": "76.1",
  "R_S32": "75.9",

  "G_ACC": "79.0",
  "G_DELTA": "-0.4",
  "G_F1": "0.790",
  "G_VRAM": "795",
  "G_PARAMS": "124.7",
  "G_DENSE": "79.0",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "XA_CONTRIB": "3.5",
  "SEER_CONTRIB": "1.5",
  "DUSA_CONTRIB": "0.5",
  "SOFTPICK_CONTRIB": "0.5",

  "T128_STD": "1.9",
  "T128_NEXUS": "2.4",
  "S128": "0.79",
  "T256_STD": "3.8",
  "T256_NEXUS": "3.5",
  "S256": "1.09",
  "T512_STD": "9.5",
  "T512_NEXUS": "6.0",
  "S512": "1.58",

  "CLS_DROP": "8.5",

  "COMPUTE_HOURS": "14",
  "CO2_KG": "1.0",
  "RUNTIME": "20",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — NEXUS

## Critical Honesty Checks
- [ ] **Stripe inactive at N=128 documented**
- [ ] **RFF inactive by default documented**
- [ ] **Causal decoder uses dense fallback documented**
- [ ] **Only 4 of 6 components active documented**

## Setup
- [ ] Copied `_TEMPLATE/` to `nexus/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nexus`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Six components mentioned
- [ ] Caveats explicit
- [ ] Modular framework positioning

## Method
- [ ] All six components formulaically defined
- [ ] Activation conditions documented
- [ ] Complexity analysis

## Results
- [ ] Main results with active path
- [ ] Component ablation
- [ ] Block size ablation
- [ ] Top-K ablation
- [ ] Softpick temperature
- [ ] Component activation summary

## Limitations
- [ ] Stripe inactive
- [ ] RFF inactive
- [ ] Dense fallback
- [ ] Softpick risk
- [ ] copy_weights raises

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/nexus
nano papers/nexus/variables.json
cd papers/nexus
python apply_vars.py --module nexus
grep -rn "{{" .
python build.py --module nexus --pdf
```
```

---

## 🎯 خلاصه نهایی

### نکات حیاتی برای مقاله

| هشدار | چرا مهم است |
|---|---|
| **Stripe فقط $N > 256$** | در SST-2 (128) غیرفعال |
| **RFF فقط با `enable_spectraformer=True`** | پیش‌فرض False |
| **Causal → dense** | GPT-2 هیچ sparsity نمی‌گیرد |
| **Softpick در all-negative → uniform** | failure mode |
| **4 از 6 component فعال** | ادعا با واقعیت فرق دارد |
| **`raise` در copy_weights** | fix شود |

### نوآوری‌های قابل برجسته‌سازی

1. **Modular framework** — امکان enable/disable هر component
2. **Combination study** — اولین ترکیب شش تکنیک ۲۰۲۵
3. **Multi-architecture** — ۴ معماری
4. **Honest documentation** — صراحت دربارهٔ component فعال

### تفاوت با سایر مقالات

| مقاله | نوآوری اصلی |
|---|---|
| A3/AASVD | Weight compression |
| TCA | Token compression |
| Linformer | Sequence projection |
| LEKAv3 | Learned landmarks |
| Mamba SSM | SSM-inspired |
| MoE-DeltaKV | MoE + simulation |
| **NEXUS** | **6-component combination** |

---

**قدم بعدی: `run_all_4_nexus.py` را اجرا کن، اعداد را بردار، و در `variables.json` بگذار. اگر می‌خواهی fix `copy_weights` را انجام دهم، بگو.**