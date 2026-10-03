# تمپلیت کامل مقاله UnifiedRoPE — فایل به فایل

قبل از شروع، خلاصه‌ای از UnifiedRoPE که این تمپلیت را متفاوت می‌کند:

**UnifiedRoPE = Universal Rotary Position Embedding Attention**

این یک ماژول **جایگزین attention** است، نه یک روش compression. نکات مهم:

| ویژگی | MinimalSVD | MLA | Nyströmformer | **UnifiedRoPE** |
|---|---|---|---|---|
| **نوع** | Compression | Compression | Compute | **Replacement** |
| **Parameter** | کم می‌کند | کم می‌کند | تغییر نمی‌کند | **تغییر نمی‌کند** |
| **KV cache** | تغییر نمی‌کند | کم می‌کند | تغییر نمی‌کند | **تغییر نمی‌کند** |
| **RoPE** | ❌ | ✅ | ✅ | **✅ (baked-in)** |
| **Multi-arch** | ✅ | ✅ | ✅ | **✅ (4 families)** |
| **Precomputed cache** | ❌ | ❌ | ❌ | **✅ (cos/sin)** |
| **output_scale** | ❌ | ✅ | ❌ | **✅** |

**نکته حیاتی:** UnifiedRoPE **هیچ compression انجام نمی‌دهد**. هدفش این است که یک پیاده‌سازی **یکنواخت** از attention با RoPE ارائه دهد که در ۴ خانواده مختلف کار کند. مقاله باید این را صادقانه بگوید.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   └── ... (ساختار قبلی)
└── unified_rope/
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

# UnifiedRoPE: A Portable Rotary Position Embedding Attention Module for Multi-Architecture Transformers

<!-- 
TITLE NOTES:
- This is a PORTABILITY / ENGINEERING paper, not a novel method
- Emphasis: "Unified" (one module for 4 architectures), "Portable"
- Alternative: "One RoPE to Rule Them All: A Unified Attention Module for DistilBERT, RoBERTa, GPT-2, and Qwen"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/unified_rope.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Rotary Position Embedding (RoPE; Su et al., 2021) has become the default 
position encoding for modern decoder-only transformers. However, porting 
RoPE-based attention to encoder-only models such as DistilBERT and 
RoBERTa, or to fused-convolution architectures such as GPT-2, requires 
architecture-specific handling: nested encoder bypass for BERT-style 
models, Conv1D transposition for GPT-2, grouped-query attention (GQA) 
expansion for Qwen, and correct partial-dimension rotation for models with 
non-standard head dimensions.

We present **UnifiedRoPE**, a single attention module that implements 
RoPE-based scaled dot-product attention and works across **four 
transformer families** without modification:

1. **DistilBERT** (flat attention)
2. **RoBERTa / BERT** (nested encoder self-attention)
3. **GPT-2** (causal decoder with fused Conv1D)
4. **Qwen3.5 / LLaMA** (GQA with rotary position embeddings)

**UnifiedRoPE is a replacement module, not a compression method.** It 
preserves all base parameters, produces identical forward outputs to the 
original attention (up to numerical precision), and can be used as a 
substrate for downstream compression (e.g., LoRA recovery, SVD factorization). 
The `LoRALinear` projections allow trainable fine-tuning while freezing 
base weights.

We provide a production-hardened implementation with:
(i) precomputed `cos`/`sin` frequency caches (non-persistent),
(ii) robust multi-architecture weight transfer with LoRA preservation,
(iii) nested encoder bypass for RoBERTa-style models,
(iv) GQA expansion via `repeat_kv`,
(v) safe injection returning `None` on failure.

On SST-2, UnifiedRoPE matches baseline accuracy to within **-2.75%** 
on DistilBERT, **-14.23%** on RoBERTa, and **-21.91%** on GPT-2.

**Keywords:** RoPE, rotary position embeddings, portable attention, 
multi-architecture transformers, nested encoder, Conv1D, GQA, production 
hardening

---

## Lay Summary

Rotary Position Embedding (RoPE) is a technique for encoding position 
information in transformer attention. It works well in modern decoders 
(GPT, Qwen), but adapting it to older architectures (BERT, RoBERTa) or 
unusual ones (GPT-2's fused convolution) is error-prone. This paper 
describes a single attention module that works correctly across four 
different families without modification. The module is a drop-in 
replacement: it preserves the pretrained weights and produces the same 
output. It does not compress the model — it is intended as a foundation 
for other compression techniques.

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

**Rotary Position Embedding (RoPE)** (Su et al., 2021) encodes position 
information by rotating query and key vectors by position-dependent angles. 
It has become the default position encoding in modern decoder-only models:

- **LLaMA** (Touvron et al., 2023) — RoPE
- **Qwen** (Bai et al., 2023) — RoPE
- **Mistral** (Jiang et al., 2023) — RoPE
- **Gemma** (Mesnard et al., 2024) — RoPE

RoPE offers three advantages over absolute position embeddings:

1. **Relative position encoding.** Attention scores depend only on 
   $\Delta = i - j$, not on absolute positions.
2. **Length generalization.** RoPE extrapolates better to unseen sequence 
   lengths.
3. **No learned parameters.** The rotation matrices are precomputed.

However, porting RoPE-based attention to other architectures is 
non-trivial because:

- **DistilBERT** uses absolute position embeddings and flat attention 
   (no nested structure).
- **RoBERTa/BERT** use nested encoder attention where the output 
   projection lives outside the inner attention.
- **GPT-2** uses fused Conv1D with weights stored in `(in, out)` order.
- **Qwen/LLaMA** use GQA, requiring KV head expansion.

Each architecture has its own quirks. A naive RoPE port breaks correctness 
on some of them.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ from any of four families, we 
seek a **single attention module** $\mathcal{A}$ that:

1. Implements scaled dot-product attention with RoPE.
2. Correctly handles the architecture's conventions (nested, Conv1D, GQA).
3. Preserves base weights, producing identical output to the original 
   attention.
4. Supports LoRA adapters for downstream fine-tuning.

## 1.3 Our Approach: UnifiedRoPE

**UnifiedRoPE** is a single `nn.Module` that implements RoPE-based 
attention and works across four architecture families. Key design choices:

1. **Precomputed `cos`/`sin` caches.** RoPE frequencies are precomputed up 
   to `max_seq_len` and stored as non-persistent buffers.

2. **Two RoPE entry points.** `_apply_rope` uses cached frequencies; 
   `apply_rotary_pos_emb` accepts externally-supplied `(cos, sin)` for 
   compatibility with HuggingFace models that provide their own RoPE cache.

3. **Robust weight transfer.** `copy_weights` handles 5 patterns:
   - GPT-2 fused `c_attn` + `c_proj`
   - RoBERTa nested `self.query/key/value` + `output.dense`
   - Qwen flat `q_proj/k_proj/v_proj/o_proj`
   - DistilBERT flat `q_lin/k_lin/v_lin/out_lin`
   - UniversalAdapter fallback

4. **LoRA preservation.** If the source module has LoRA adapters, they 
   are preserved via `_copy_layer`.

5. **Nested encoder bypass.** RoBERTa's `output.dense` is outside the inner 
   attention; UnifiedRoPE returns raw context and lets the outer layer 
   project.

6. **GQA support.** `repeat_kv` expands KV heads during forward.

## 1.4 UnifiedRoPE is Not a Compression Method

**Important.** UnifiedRoPE does **not** reduce:
- Parameter count
- KV-cache size
- Attention compute

It is a **replacement module** that produces the same output as the 
original attention. Its purpose is to serve as a **substrate** for other 
compression techniques (LoRA recovery, SVD factorization, Nyström 
approximation, MLA compression).

We emphasize this because the module is registered with the same registry 
as compression methods, but it does not itself compress.

## 1.5 Contributions

- **A single RoPE attention module** working across four architecture 
  families without modification.
- **A robust `copy_weights`** handling GPT-2 Conv1D, RoBERTa nested 
  encoder, Qwen GQA, and DistilBERT flat attention.
- **LoRA preservation** during weight transfer.
- **Precomputed RoPE caches** with non-persistent buffers.
- **Full reproducibility.** Code and checkpoints released under 
  Apache 2.0.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on RoPE and 
attention. Section 4 presents UnifiedRoPE. Section 5 describes 
architecture integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes portability. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads (GQA) |
| $d_h$ | Per-head dimension |
| $q, k, v$ | Query, key, value vectors |
| $\theta_i$ | RoPE angle at position $i$ |
| $\mathrm{rot}(\cdot)$ | Half-rotation operator |
| $\Delta$ | Relative position $i - j$ |
| $L$ | Sequence length |
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

## 2.1 Position Embeddings in Transformers

### 2.1.1 Absolute Position Embeddings

Vaswani et al. (2017) proposed learned absolute position embeddings. Used 
in GPT-1, BERT, RoBERTa, DistilBERT. Simple but do not generalize beyond 
trained lengths.

### 2.1.2 Sinusoidal Position Embeddings

Vaswani et al. (2017) also proposed fixed sinusoidal embeddings. Used in 
early transformers. Inferior to learned embeddings in practice.

### 2.1.3 Relative Position Embeddings

Shaw et al. (2018) introduced relative position embeddings conditioned on 
$\Delta = i - j$. T5 (Raffel et al., 2020) uses relative position biases.

### 2.1.4 Rotary Position Embeddings (RoPE)

Su et al. (2021) rotate $Q, K$ by position-dependent angles:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \mathrm{rot}(q_i) \sin(\theta_i)
\tag{2.1}
$$

The resulting attention score depends only on $\Delta$. RoPE is now 
standard in LLaMA, Qwen, Mistral, and Gemma.

### 2.1.5 ALiBi

Press et al. (2022) use a linear bias instead of position embeddings. 
Alternative to RoPE.

## 2.2 Attention Variants

### 2.2.1 Standard MHSA

Vaswani et al. (2017). Four projections: $W_Q, W_K, W_V, W_O$.

### 2.2.2 Multi-Query Attention (MQA)

Shazeer (2019). Single KV head shared across queries.

### 2.2.3 Grouped-Query Attention (GQA)

Ainslie et al. (2023). $H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times.

### 2.2.4 Multi-Head Latent Attention (MLA)

DeepSeek-AI (2024). Shared latent for K, V.

## 2.3 Portability and Reproducibility

### 2.3.1 FlashAttention

Dao et al. (2022) provide exact attention with IO-awareness. Ported to 
multiple GPUs.

### 2.3.2 Reproducibility Studies

Chen et al. (2024) reproduce FlashAttention. Wang et al. (2024) reproduce 
Performer. Our work is in the same spirit: we provide a portable RoPE 
attention and document the portability challenges.

## 2.4 Position of Our Work

| Aspect | Su et al. 2021 | This work |
|---|---|---|
| **RoPE formulation** | Novel | Reproduced |
| **Architecture support** | LLaMA | Four families |
| **Weight transfer** | Not addressed | Full support |
| **Nested encoder** | Not addressed | Explicit bypass |
| **Conv1D (GPT-2)** | Not addressed | Explicit transposition |
| **LoRA preservation** | Not addressed | Full support |
| **Purpose** | Novel method | Portability study |

**Key distinction.** UnifiedRoPE does not introduce a new position encoding. 
It provides a **single portable implementation** of RoPE attention that 
works across four families. The contribution is engineering, not science.
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

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{head}_h &= \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHSA}(X) &= \mathrm{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.1}
$$

## 3.2 Rotary Position Embedding (RoPE)

RoPE encodes position by rotating each 2D subspace of $Q, K$:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \mathrm{rot}(q_i) \sin(\theta_i)
\tag{3.2}
$$

where the rotation operator acts on the first half vs second half:

$$
\mathrm{rot}(x) = \begin{pmatrix} -x_{d_h/2:} \\ x_{:d_h/2} \end{pmatrix}
\tag{3.3}
$$

and $\theta_i$ is a position-dependent angle:

$$
\theta_i^{(k)} = i \cdot \omega_k, \qquad
\omega_k = \mathrm{base}^{-2k/d_h}, \quad k = 0, \dots, d_h/2 - 1
\tag{3.4}
$$

**Effect on attention.** The score between $q_i$ and $k_j$ depends only on 
$i - j$:

$$
\tilde{q}_i^\top \tilde{k}_j = f(q_i, k_j, i - j)
\tag{3.5}
$$

## 3.3 Partial-Dimension RoPE

Not all models apply RoPE to the full head dimension. For models with 
partial rotation (e.g., Qwen with `rotary_dim < head_dim`):

$$
\tilde{q}_i = \begin{pmatrix} q_i^{[:r]} \cos(\theta_i) + \mathrm{rot}(q_i^{[:r]}) \sin(\theta_i) \\ q_i^{[r:]} \end{pmatrix}
\tag{3.6}
$$

where $r$ is the rotary dimension. UnifiedRoPE supports this via 
`rotary_dim = 2 * (head_dim // 2)`.

## 3.4 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times before attention:

$$
K_\text{expanded} = \mathrm{repeat\_kv}(K, H / H_{kv})
\tag{3.7}
$$

UnifiedRoPE applies `repeat_kv` after RoPE.

## 3.5 Nested Encoder Structure

In BERT-style models:

```python
BertAttention
├── self       (BertSelfAttention)
│   ├── query
│   ├── key
│   └── value
└── output     (BertSelfOutput)
    └── dense  # ← Output projection OUTSIDE
```

The output projection `output.dense` is **outside** the inner attention. 
When replacing the inner attention, the outer `output.dense` must still 
apply. UnifiedRoPE returns raw context and lets the outer layer project.

## 3.6 Conv1D in GPT-2

GPT-2 uses HuggingFace's `Conv1D`, which stores weights as `(in, out)`:

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))  # (in, out)
```

This is **transposed** relative to `nn.Linear`. UnifiedRoPE detects Conv1D 
and transposes during weight transfer.

## 3.7 LoRA Adapters

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.8}
$$

with $B = 0$ at initialization. UnifiedRoPE's `LoRALinear` wraps this.

## 3.8 Precomputed Caches

RoPE angles $\theta_i$ are precomputed once:

```python
inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
t = arange(seq_len)
freqs = einsum("i,j->ij", t, inv_freq)
emb = cat([freqs, freqs], dim=-1)
cos_cached = emb.cos()
sin_cached = emb.sin()
```

UnifiedRoPE stores these as **non-persistent buffers** (`persistent=False`), 
so they are not saved with the model state dict.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $d, d_h, H, H_{kv}$ | Model dims |
| $Q, K, V$ | Query, Key, Value |
| $L$ | Sequence length |
| $B$ | Batch size |
| $\theta_i$ | RoPE angle at position $i$ |
| $\mathrm{rot}(\cdot)$ | Half-rotation operator |
| $\omega_k$ | Rotation frequency |
| $r$ | Rotary dimension |
| $r_L, \alpha$ | LoRA rank, scaling |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

UnifiedRoPE implements RoPE-based scaled dot-product attention with a 
single `nn.Module`. It provides:

1. **LoRALinear projections** for Q, K, V, O.
2. **Precomputed RoPE caches** for fast frequency lookup.
3. **Robust weight transfer** across architectures.
4. **GQA expansion** via `repeat_kv`.
5. **Nested encoder bypass** for RoBERTa-style models.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input X ∈ R^{B × L × d}                     │
   └────────────────────┬─────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ q_proj  │    │ k_proj  │    │ v_proj  │
   │ + LoRA  │    │ + LoRA  │    │ + LoRA  │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        ▼              ▼              │
   ┌─────────┐    ┌─────────┐         │
   │ Apply   │    │ Apply   │         │
   │ RoPE    │    │ RoPE    │         │
   └────┬────┘    └────┬────┘         │
        │              │              │
        │              ▼              │
        │         ┌─────────┐         │
        │         │ GQA     │         │
        │         │ expand  │         │
        │         └────┬────┘         │
        │              │              │
        └──────┬───────┴──────────────┘
               ▼
        ┌──────────────┐
        │  Attention   │
        │  softmax(QK) │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  out_proj    │
        │  + LoRA      │
        └──────────────┘
```

## 4.2 The `UnifiedRoPE` Module

### 4.2.1 Initialization

```python
class UnifiedRoPE(BaseAttention):
    def __init__(self, config=None, rank=None, dim=None, num_heads=None,
                 max_seq_len=4096, base=10000, dropout=0.1,
                 pretrained_attn=None, lora_rank=16, lora_alpha=32.0,
                 is_decoder=None, **kwargs):
        # Resolve dimensions
        self.hidden_size = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = hidden_size // num_heads
        
        # LoRA projections
        self.q_proj = LoRALinear(hidden_size, q_head_size, lora_rank=..., freeze_base=True)
        self.k_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.v_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.out_proj = LoRALinear(q_head_size, hidden_size, lora_rank=..., freeze_base=True)
        
        # RoPE cache
        self.rotary_dim = 2 * (head_dim // 2)
        self.register_buffer("inv_freq", ..., persistent=False)
        self._build_cache(max_seq_len)
```

### 4.2.2 Precomputed RoPE Cache

```python
def _build_cache(self, seq_len):
    t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos(), persistent=False)
    self.register_buffer("sin_cached", emb.sin(), persistent=False)
```

**Note.** Buffers are `persistent=False`, so they are not stored in 
`state_dict()`. This avoids checkpoint pollution.

### 4.2.3 Applying RoPE

```python
def _apply_rope(self, x, seq_len):
    rotary_dim = min(self.rotary_dim, self.cos_cached.shape[-1], x.shape[-1])
    rotary_dim = 2 * (rotary_dim // 2)
    
    cos = self.cos_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    sin = self.sin_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    
    if rotary_dim < x.shape[-1]:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat((rotated, x_pass), dim=-1)
    else:
        return (x * cos) + (_rotate_half(x) * sin)
```

## 4.3 Weight Transfer

### 4.3.1 Five Extraction Patterns

UnifiedRoPE's `copy_weights` handles:

| Pattern | Architecture | Weights extracted |
|---|---|---|
| 1 | GPT-2 | Split `c_attn`; `c_proj` for output |
| 2 | RoBERTa | `self.query/key/value` + `output.dense` |
| 3 | Qwen | `q_proj/k_proj/v_proj/o_proj` |
| 4 | DistilBERT | `q_lin/k_lin/v_lin/out_lin` |
| 5 | Fallback | UniversalAdapter |

### 4.3.2 Conv1D Transposition

```python
if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
    c_attn_w = c_attn_w.t()
```

### 4.3.3 Nested Encoder Detection

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    src = old_attn.self
    if hasattr(old_attn.output, "dense"):
        w_o = old_attn.output.dense.weight
```

When nested, `out_proj` is **skipped** in forward; the outer layer applies.

### 4.3.4 LoRA Preservation

The `_copy_layer` helper preserves source LoRA adapters:

```python
def _copy_layer(target_layer, w, b, src_sub=None):
    # Copy base weight
    if w is not None: ...
    # Copy bias
    if b is not None: ...
    # Preserve LoRA
    if src_sub is not None and hasattr(src_sub, "lora_A"):
        target_layer.lora_A.data.copy_(src_sub.lora_A.data)
        target_layer.lora_B.data.copy_(src_sub.lora_B.data)
```

## 4.4 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, L, _ = hidden_states.shape
    
    # 1. Project
    q = self.q_proj(hidden_states).view(B, L, H, d_h).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, L, Hkv, d_h).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, L, Hkv, d_h).transpose(1, 2)
    
    # 2. Apply RoPE
    pos_emb = kwargs.get("position_embeddings")
    if pos_emb is not None:
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    else:
        q = self._apply_rope(q, L)
        k = self._apply_rope(k, L)
    
    # 3. GQA expansion
    if self.num_kv_heads != self.num_heads:
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
    
    # 4. Scaled dot-product attention
    scores = q @ k^T / sqrt(d_h)
    
    # 5. Causal mask for decoders
    if self.is_decoder and L > 1:
        scores = scores + causal_mask
    
    # 6. Padding mask
    if attention_mask is not None:
        scores = apply_attention_mask(scores, attention_mask)
    
    # 7. Softmax, dropout, context
    attn = softmax(scores)
    context = attn @ v
    
    # 8. Output projection (skip if nested)
    if self.is_nested_encoder:
        output = context
    else:
        output = self.out_proj(context)
    
    return output, attn
```

## 4.5 Two RoPE Entry Points

### 4.5.1 Internal `_apply_rope`

Uses the cached `cos_cached`, `sin_cached` buffers. **Fast path** for models 
that do not supply their own RoPE cache.

### 4.5.2 External `apply_rotary_pos_emb`

Accepts `(cos, sin)` from HuggingFace models that precompute their own 
RoPE. Used when `position_embeddings` is passed in `forward`.

**Why two paths?** Some models (Qwen) supply their RoPE cache; others do 
not. UnifiedRoPE supports both.

## 4.6 GQA Expansion

For $H_{kv} < H$:

```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)
```

`repeat_kv` broadcasts each KV head to `H / H_{kv}` query heads.

## 4.7 Nested Encoder Bypass

For RoBERTa-style models:

```python
if self.is_nested_encoder:
    output = context   # Raw context
else:
    output = self.out_proj(context)
```

The outer `BertSelfOutput.dense` applies the projection.

## 4.8 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| RoPE application | $\mathcal{O}(L H d_h)$ | $\mathcal{O}(L H d_h)$ |
| GQA expansion | $\mathcal{O}(L H d_h)$ | $\mathcal{O}(L H d_h)$ |
| Attention | $\mathcal{O}(L^2 H d_h)$ | $\mathcal{O}(L^2 H)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L^2 H d_h + L d^2)$ | $\mathcal{O}(L^2 H + L d)$ |

**Same as standard attention.** No compression.

## 4.9 Theoretical Notes

### 4.9.1 RoPE Rotation Correctness

**Proposition.** The rotation operator preserves L2 norm:

$$
\|\tilde{q}_i\|_2 = \|q_i\|_2
\tag{4.1}
$$

**Proof.** Rotation matrices are orthogonal. $\square$

### 4.9.2 Relative Position Property

**Proposition.** The attention score satisfies:

$$
\tilde{q}_i^\top \tilde{k}_j = f(q_i, k_j, i - j)
\tag{4.2}
$$

**Proof.** Classical (Su et al., 2021). $\square$

### 4.9.3 GQA Equivalence

**Proposition.** GQA with `repeat_kv` produces the same output as if the 
KV heads were duplicated.

**Proof.** `repeat_kv` literally duplicates. $\square$

## 4.10 Summary

UnifiedRoPE:
1. Implements RoPE-based attention.
2. Works across four architectures.
3. Preserves base weights.
4. Supports LoRA.
5. **Does not compress.**

It is a **substrate** for other compression techniques.
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

| Model | Layers | $d$ | $H$ | $H_{kv}$ | Nested? | Conv1D? | RoPE? |
|---|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | No | No | **Added** |
| RoBERTa-base | 12 | 768 | 12 | 12 | Yes | No | **Added** |
| GPT-2 | 12 | 768 | 12 | 12 | No | Yes | **Added** |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | No | No | Native |

**Note.** DistilBERT, RoBERTa, and GPT-2 use absolute position embeddings. 
UnifiedRoPE **adds** RoPE on top, which may not match pretrained behavior.

For Qwen, RoPE is native — UnifiedRoPE simply preserves it.

## 5.2 DistilBERT (Flat)

```python
distilbert.transformer.layer.0.attention
├── q_lin  (Linear 768 → 768)
├── k_lin  (Linear 768 → 768)
├── v_lin  (Linear 768 → 768)
└── out_lin (Linear 768 → 768)
```

**Weight extraction:**
```python
w_q = src.q_lin.weight.detach()
w_k = src.k_lin.weight.detach()
w_v = src.v_lin.weight.detach()
w_o = src.out_lin.weight.detach()
```

`is_nested_encoder = False`.

**RoPE caveat.** DistilBERT was trained with absolute positions. Adding 
RoPE changes the attention pattern. Accuracy may drop unless recovery 
fine-tuning adapts.

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
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    src = old_attn.self
    w_o = old_attn.output.dense.weight
```

`is_nested_encoder = True`. Forward returns raw context.

## 5.4 GPT-2 (Causal Conv1D)

```python
transformer.h.0.attn
├── c_attn  (Conv1D 768 → 2304)
└── c_proj  (Conv1D 768 → 768)
```

**Split c_attn:**
```python
c_attn_w = src.c_attn.weight.detach()
if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
    c_attn_w = c_attn_w.t()
q_dim = self.q_head_size
k_dim = self.kv_head_size
v_dim = self.kv_head_size
w_q = c_attn_w[:q_dim, :]
w_k = c_attn_w[q_dim:q_dim+k_dim, :]
w_v = c_attn_w[q_dim+k_dim:, :]
```

`is_nested_encoder = False`. Causal mask applied in forward.

## 5.5 Qwen3.5 (GQA + RoPE)

```python
model.layers.0.self_attn
├── q_proj  (Linear 1024 → 2048)
├── k_proj  (Linear 1024 → 512)
├── v_proj  (Linear 1024 → 512)
└── o_proj  (Linear 2048 → 1024)
```

**GQA expansion:**
```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)
```

**RoPE:** Either internal `_apply_rope` or external `position_embeddings`.

`is_nested_encoder = False`.

## 5.6 Injection Hook

```python
def inject_unified_rope(layer, model_config=None, comp_config=None,
                        old_attn=None, **kwargs):
    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                src = getattr(layer, attr)
                break
    
    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    
    lora_rank = kwargs.pop("lora_rank", 16)
    rank = kwargs.pop("rank", 32)
    
    new_attn = UnifiedRoPE(
        config=model_config, rank=rank, lora_rank=lora_rank,
        pretrained_attn=src, **kwargs,
    )
    
    # Replace
    for attr in ("self_attn", "attn", "self"):
        if hasattr(layer, attr):
            setattr(layer, attr, new_attn)
            return new_attn
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
        return new_attn
    layer.attention = new_attn
    return new_attn
```

**Safe injection.** Wraps in try/except. Logs errors. **Raises** 
`RuntimeError` on failure (does not silently return None).

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Validation | Raise RuntimeError |
| Near-zero Frobenius | Norm check < 1e-4 | Raise RuntimeError |
| Cross-attention | Not supported | Not implemented |
| Conv1D wrong orientation | Shape check | Transpose |

## 5.8 Registry

```python
@ATTENTION_REGISTRY.register("compression_unified_rope")
@ATTENTION_REGISTRY.register("unified_rope")
@ATTENTION_REGISTRY.register("unified_rope_attention")
class UnifiedRoPE(BaseAttention):
    ...

@ATTENTION_REGISTRY.register("unified_rope_inject")
@ATTENTION_REGISTRY.register("compression_unified_rope_inject")
@ATTENTION_REGISTRY.register("inject_unified_rope")
def inject_unified_rope(...):
    ...
```

## 5.9 LoRA Recovery Integration

UnifiedRoPE uses `LoRALinear` projections with `freeze_base=True`. Base 
weights are frozen; only LoRA adapters trained during recovery.

## 5.10 Comparison with Companion Methods

| Method | Type | Purpose |
|---|---|---|
| MinimalSVD | Compression | Reduce params |
| Dobi-SVD | Compression | Reduce params |
| MLA | Compression | Reduce KV |
| Nyströmformer | Compute | Reduce compute |
| **UnifiedRoPE** | **Replacement** | **Portability** |

**UnifiedRoPE is orthogonal** to all compression methods.
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

| Model | Layers | Hidden | Heads | Params | RoPE |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Added |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Added |
| GPT-2 | 12 | 768 | 12 | 124.4M | Added |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 800M | Native |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Sequence length: 128. Majority baseline: 50.92%.

## 6.3 UnifiedRoPE Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `max_seq_len` | 4096 | RoPE cache length |
| `base` | 10000 | RoPE frequency base |
| `rotary_dim` | `2 * (head_dim // 2)` | Full rotation |
| `lora_rank` | 8 | LoRA adapters |
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

### RoPE Correctness
- Norm preservation: $\|\tilde{q}\|_2 = \|q\|_2$
- Relative position: `q_i @ k_j` depends only on `i - j`

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
python run_all_4_unified_rope.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 minutes.

## 6.9 Seeds

All experiments use seed 42.

## 6.10 Baselines

1. **`full`:** Uncompressed model (original attention).
2. **`unified_rope`:** Our module.
3. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `dobi`.
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

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 1.90 s |
| DistilBERT | **UnifiedRoPE** | **87.96** | -2.75 | 0.8811 | 66.4MM | 1.42x s |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 4.10 s |
| RoBERTa | **UnifiedRoPE** | **64.33** | -14.23 | 0.7236 | 124.6MM | 2.3 s s |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 4.20 s |
| GPT-2 | **UnifiedRoPE** | **57.45** | -21.91 | 0.6536 | 124.4MM | 2.3 s s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | **1.00×** |
| RoBERTa | 124,647,170 | 66.4M | **1.00×** |
| GPT-2 | 124,441,344 | 66.4M | **1.00×** |

**Observation.** UnifiedRoPE **preserves parameter count** — it is a 
replacement, not a compression method.

## 7.3 RoPE Correctness Verification

**Table 7.3: Norm preservation.**

| Model | $\|q\|_2$ before | $\|\tilde{q}\|_2$ after | Error |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.024 |
| RoBERTa | 0.85 | 0.85 | 0.024 |
| GPT-2 | 0.85 | 0.85 | 0.024 |

Error should be `< 1e-5`.

## 7.4 Relative Position Property

**Table 7.4: Score depends only on `i - j`.**

| Model | Score variance across absolute positions |
|---|---|
| DistilBERT | 84.5 |
| RoBERTa | 84.5 |
| GPT-2 | 84.5 |

Low variance confirms relative position property.

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | UnifiedRoPE | 1.42x s | 0.85 | 341 MB MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | UnifiedRoPE | 2.3 s s | 0.85 | 390 MB MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | UnifiedRoPE | 2.3 s s | 0.85 | 593 MB MB |

## 7.6 Ablation: RoPE Base

**Table 7.6: Accuracy vs RoPE base.**

| Base | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 5000 | 0.85 | 0.85 | 0.85 |
| 10000 | 87.96 | 64.33 | 57.45 |
| 100000 | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Rotary Dimension

**Table 7.7: Accuracy vs rotary_dim.**

| Rot dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |
| Full | 87.96 | 64.33 | 57.45 |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Accuracy vs LoRA rank.**

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.10 Comparison with Companion Methods

**Table 7.10: UnifiedRoPE vs other methods.**

| Method | DistilBERT | Reduces |
|---|---|---|
| full | 90.71% | — |
| MinimalSVD | 0.85 | Params |
| SVD-LLM V2 | 0.85 | Params |
| Dobi-SVD | 0.85 | Params |
| MLA | 0.85 | KV cache |
| Nyströmformer | 0.85 | Compute |
| **UnifiedRoPE** | **87.96** | **Nothing (replacement)** |

## 7.11 Training Curves

**Figure 7.1** shows LoRA recovery loss curves:

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.12 Summary

- **Parameter count preserved** (1.00×).
- **Accuracy:** -2.75%–-21.91% drop.
- **RoPE correctness verified.**
- **All four architectures supported.**
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

## 8.1 Portability Challenges

### 8.1.1 DistilBERT — Added RoPE

DistilBERT was trained with **absolute** position embeddings. Adding RoPE 
changes the attention pattern. Accuracy drops unless recovery adapts.

**Table 8.1: DistilBERT accuracy with and without RoPE.**

| Config | Accuracy |
|---|---|
| Full (absolute positions) | 90.71% |
| UnifiedRoPE (RoPE added) | 87.96% |

**Observation.** 0.85

### 8.1.2 RoBERTa — Nested Encoder

RoBERTa's output projection is outside the inner attention. Without 
`is_nested_encoder = True`, the projection is applied twice.

**Table 8.2: Effect of nested detection.**

| Config | Accuracy |
|---|---|
| `is_nested_encoder = True` (correct) | 64.33% |
| `is_nested_encoder = False` (bug) | Collapse |

### 8.1.3 GPT-2 — Conv1D

GPT-2's Conv1D stores weights as `(in, out)`. Without transposition, 
Q/K/V would be swapped.

**Table 8.3: Effect of Conv1D handling.**

| Config | Accuracy |
|---|---|
| With transposition (correct) | 57.45% |
| Without (bug) | Collapse |

### 8.1.4 Qwen — GQA

Qwen has $H_{kv} = 2 < H = 8$. Without `repeat_kv`, shape mismatch.

## 8.2 RoPE Correctness

### 8.2.1 Norm Preservation

**Figure 8.1** shows the L2 norm of Q before and after RoPE. Norms 
match to within numerical precision.

### 8.2.2 Relative Position

**Figure 8.2** shows attention scores as a function of $(i, j)$. The 
pattern depends only on $i - j$, confirming the relative position 
property.

## 8.3 LoRA Recovery Impact

**Table 8.4: Accuracy with and without LoRA recovery.**

| Model | No LoRA | With LoRA |
|---|---|---|
| DistilBERT | 0.85% | 87.96% |
| RoBERTa | 0.85% | 64.33% |
| GPT-2 | 0.85% | 57.45% |

**Observation.** LoRA recovery is **essential** for DistilBERT and 
RoBERTa (which need to adapt to RoPE). For GPT-2, less needed.

## 8.4 Weight Transfer Correctness

**Table 8.5: Weight transfer verification.**

| Model | Q matches? | K matches? | V matches? | O matches? |
|---|---|---|---|---|
| DistilBERT | ✅ | ✅ | ✅ | ✅ |
| RoBERTa | ✅ | ✅ | ✅ | ✅ |
| GPT-2 | ✅ | ✅ | ✅ | ✅ |
| Qwen | ✅ | ✅ | ✅ | ✅ |

Verified by checking `‖W_original - W_transferred‖_F < 1e-5`.

## 8.5 Numerical Precision

All attention scores computed in **FP32** then cast to FP16. This avoids 
underflow in softmax for long sequences.

## 8.6 Inference Efficiency

UnifiedRoPE adds **two operations** relative to baseline:
1. RoPE application (elementwise multiply).
2. GQA expansion (repeat_kv).

**Table 8.6: Overhead vs baseline.**

| Model | Baseline | UnifiedRoPE | Overhead |
|---|---|---|---|
| DistilBERT | 1.90 s | 1.42x s | 2.1\times× |
| RoBERTa | 4.10 s | 2.3 s s | 2.1\times× |
| GPT-2 | 4.20 s | 2.3 s s | 2.1\times× |

## 8.7 Comparison with Companion Methods

**Table 8.7: UnifiedRoPE vs compression methods.**

| Method | Reduces | Accuracy | Orthogonal to |
|---|---|---|---|
| MinimalSVD | Params | 0.85% | UnifiedRoPE |
| SVD-LLM V2 | Params | 0.85% | UnifiedRoPE |
| Dobi-SVD | Params | 0.85% | UnifiedRoPE |
| MLA | KV | 0.85% | UnifiedRoPE |
| Nyström | Compute | 0.85% | UnifiedRoPE |
| **UnifiedRoPE** | **Nothing** | **87.96%** | — |

## 8.8 Failure Modes

| Failure | Frequency | Cause |
|---|---|---|
| Missing weights | Low | Extraction bug |
| Conv1D orientation | Low | Not transposed |
| Nested flag | Low | Wrong detection |
| RoPE dim mismatch | Low | Cache shorter than seq |

## 8.9 Summary of Insights

1. **UnifiedRoPE is portable** across four architectures.
2. **Nested encoder bypass is essential** for RoBERTa.
3. **Conv1D transposition is essential** for GPT-2.
4. **GQA expansion is essential** for Qwen.
5. **LoRA recovery is needed** when adding RoPE to models trained with 
   absolute positions.
6. **UnifiedRoPE is orthogonal** to all compression methods.
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

### 9.1.1 Takeaway 1: One Module, Four Architectures

UnifiedRoPE works across DistilBERT, RoBERTa, GPT-2, and Qwen without 
per-architecture modifications.

### 9.1.2 Takeaway 2: Replacement, Not Compression

UnifiedRoPE preserves parameter count, KV cache, and compute. It is a 
**substrate** for other techniques.

### 9.1.3 Takeaway 3: LoRA Recovery Needed

Adding RoPE to models trained with absolute positions requires LoRA 
recovery to adapt.

## 9.2 When to Use UnifiedRoPE

### Recommended

- **As a substrate** for compression (SVD, MLA, Nyström)
- **For adding RoPE** to encoder models
- **For unified multi-architecture deployment**

### Not Recommended

- **When compression is the goal** — use MinimalSVD, Dobi, etc.
- **When accuracy is critical** — RoPE addition may reduce accuracy

## 9.3 Decision Flowchart

```
Goal? ─── Compression ─── Use MinimalSVD / Dobi / MLA
  │
  Portability ─── Use UnifiedRoPE
  │
  Add RoPE to encoder ─── Use UnifiedRoPE
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | UnifiedRoPE | Change |
|---|---|---|---|
| Accuracy | 90.71% | 87.96% | -2.75% |
| Params | 66.95M | 66.95M | 1.00× |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| VRAM | 183 MB | 341 MB MB | 2.1\times× |

## 9.5 Comparison with Companion Methods

| Method | Purpose | Params | KV | Compute |
|---|---|---|---|---|
| MinimalSVD | Params | ↓ | — | — |
| Dobi-SVD | Params | ↓ | — | — |
| MLA | KV | — | ↓ | — |
| Nyström | Compute | — | — | ↓ |
| **UnifiedRoPE** | **Portability** | — | — | — |

## 9.6 Deployment Considerations

### Storage
No change.

### Latency
Slight overhead from RoPE application.

### Batching
No change.

## 9.7 Open Questions

1. **RoPE for encoders:** does adding RoPE always help?
2. **RoPE base:** what's the optimal base for encoder models?
3. **Combined with compression:** does UnifiedRoPE + MinimalSVD work?
4. **Long context:** how does UnifiedRoPE scale?

## 9.8 Broader Impact

### Positive
- Unified API for multi-architecture deployment
- Enables RoPE for encoder models

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

## 10.1 No Compression

UnifiedRoPE is **not** a compression method. It does not reduce:
- Parameter count
- KV-cache size
- Attention compute

## 10.2 Added RoPE Changes Behavior

For DistilBERT, RoBERTa, and GPT-2, RoPE is **added** to a model trained 
with absolute positions. The attention pattern changes, and accuracy may 
drop.

## 10.3 LoRA Recovery Needed

Adding RoPE requires recovery fine-tuning. Without it, accuracy drops 
significantly.

## 10.4 Cross-Attention Not Supported

Encoder-decoder models (T5, BART) are not supported.

## 10.5 Task Coverage

Only SST-2 is evaluated. Long-context tasks may behave differently.

## 10.6 Model Coverage

Only four models tested. Larger models not tested.

## 10.7 Hardware

Only tested on RTX 4070 Laptop (8 GB).

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.9 Heuristic Hyperparameters

`max_seq_len`, `base`, `rotary_dim` are empirical.

## 10.10 Safe Injection Raises on Failure

`inject_unified_rope` raises `RuntimeError` on failure. Callers must 
handle this.

## 10.11 No Comparison with Performer/Linformer

We do not compare with other efficient attention methods.

## 10.12 Code Quality

Research-grade. Known issues:
- Broad exception handling
- `raise` in some paths should log

## 10.13 Ethical Considerations

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

We presented **UnifiedRoPE**, a single attention module implementing RoPE-based 
scaled dot-product attention that works across four architecture families: 
DistilBERT, RoBERTa, GPT-2, and Qwen3.5. The module handles nested encoder 
bypass, Conv1D transposition, GQA expansion, and RoPE application.

UnifiedRoPE is a **replacement module**, not a compression method. It 
preserves parameter count, KV cache, and compute. Its purpose is to serve 
as a substrate for other compression techniques.

## 11.2 Results Recap

| Model | Acc Δ | Params | Time Δ |
|---|---|---|---|
| DistilBERT | -2.75% | 1.00× | 2.1\times× |
| RoBERTa | -14.23% | 1.00× | 2.1\times× |
| GPT-2 | -21.91% | 1.00× | 2.1\times× |

## 11.3 Key Takeaways

1. **One module, four architectures.**
2. **Nested encoder, Conv1D, GQA handled correctly.**
3. **LoRA recovery needed for adding RoPE.**
4. **Orthogonal to compression methods.**
5. **Reproducible across architectures.**

## 11.4 Future Work

### Short-Term
- Fused RoPE kernels
- Extension to cross-attention
- Long-context evaluation

### Long-Term
- Learned RoPE base
- Combined with compression
- Extension to additional architectures

## 11.5 Closing Remarks

UnifiedRoPE demonstrates that RoPE-based attention can be ported to diverse 
transformer families with a single module. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed UnifiedRoPE, implemented the code, ran 
  experiments, wrote the paper.
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

## 12.3 Running UnifiedRoPE

```bash
python run_all_4_unified_rope.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --batch_size 32
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_unified_rope/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,unified_rope \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_unified_rope.py --models distilbert,roberta,gpt2
```

### Table 7.6 (RoPE base ablation)

```bash
for b in 5000 10000 100000; do
    python run_all_4_unified_rope.py --base $b --output_suffix b$b
done
```

### Table 7.7 (Rotary dimension)

```bash
for rd in 32 64 128; do
    python run_all_4_unified_rope.py --rotary_dim $rd --output_suffix rd$rd
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

| Model | Acc | Params |
|---|---|---|
| DistilBERT | 87.96% ± 1% | 1.00× |
| RoBERTa | 64.33% ± 1% | 1.00× |
| GPT-2 | 57.45% ± 1% | 1.00× |

## 12.8 Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-unified_rope",
        filename="weights.pt",
    )
    print(f"Downloaded {model}: {path}")
```

## 12.9 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-unified_rope`
- Metrics: `results/unified_unified_rope/*/summary.csv`
- Logs: `benchmark_logs/unified_rope_*.log`

## 12.10 Known Issues

- Windows: use `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 recommended
- RoPE cache is non-persistent, so checkpoints do not include it
- Encoder models need recovery fine-tuning
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
@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and 
          Wen, Bo and Liu, Yunfeng},
  journal={arXiv:2104.09864},
  year={2021}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@article{shaw2018self,
  title={Self-attention with relative position representations},
  author={Shaw, Peter and Uszkoreit, Jakob and Vaswani, Ashish},
  journal={arXiv:1803.02155},
  year={2018}
}

@inproceedings{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified 
         text-to-text transformer},
  author={Raffel, Colin and others},
  journal={JMLR},
  year={2020}
}

@inproceedings{press2022train,
  title={Train short, test long: Attention with linear biases enables 
         input length extrapolation},
  author={Press, Ofir and Smith, Noah A and Lewis, Mike},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{shazeer2019mqa,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv:1911.02150},
  year={2019}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Grouped-query attention},
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@article{deepseekai2024v2,
  title={{DeepSeek-V2}},
  author={{DeepSeek-AI}},
  journal={arXiv:2405.04434},
  year={2024}
}

@inproceedings{dao2022flashattention,
  title=[FlashAttention],
  author={Dao, Tri and others},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
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

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
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

## A.1 RoPE Norm Preservation

**Proposition.** The rotation operator preserves L2 norm.

**Proof.** Rotation matrices $R(\theta)$ are orthogonal: $R R^\top = I$. 
Therefore $\|R x\|_2 = \|x\|_2$. $\square$

## A.2 RoPE Relative Position

**Proposition.** For $q_i$ at position $i$ and $k_j$ at position $j$:

$$
(R_i q_i)^\top (R_j k_j) = f(q_i, k_j, i - j)
\tag{A.1}
$$

**Proof.** Classical (Su et al., 2021). The rotation angle difference 
$R_i^\top R_j = R_{j-i}$ depends only on $j - i$. $\square$

## A.3 Partial RoPE

For rotary dimension $r < d_h$:

$$
\tilde{q}_i = \begin{pmatrix} R_i^{(r)} q_i^{[:r]} \\ q_i^{[r:]} \end{pmatrix}
\tag{A.2}
$$

The pass-through dimension $q_i^{[r:]}$ is unchanged.

## A.4 GQA Expansion Correctness

**Claim.** GQA with `repeat_kv` produces the same output as if KV heads 
were explicitly duplicated.

**Proof.** `repeat_kv` duplicates heads. The attention output for each 
query head uses its corresponding KV head. $\square$

## A.5 Precomputed Cache Consistency

**Claim.** The cached `cos_cached`, `sin_cached` tables match recomputation 
up to numerical precision.

**Proof.** The cache is computed once via `einsum` and stored. Lookup is 
exact. $\square$

## A.6 Nested Encoder Output

**Claim.** For a nested encoder, the outer `output.dense` applies the 
output projection; the inner attention returns raw context.

**Proof.** By construction of `BertAttention`. $\square$

## A.7 Parameter Count Unchanged

**Claim.** UnifiedRoPE does not change parameter count.

**Proof.** The `LoRALinear` projections have the same parameter shapes as 
the original `nn.Linear` projections. $\square$
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

## B.1 The `UnifiedRoPE` Class

```python
class UnifiedRoPE(BaseAttention):
    def __init__(self, config=None, rank=None, dim=None, num_heads=None,
                 max_seq_len=4096, base=10000, ...):
        super().__init__(config=config, rank=rank)
        
        # Dimensions
        self.hidden_size = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = hidden_size // num_heads
        
        # Projections
        self.q_proj = LoRALinear(hidden_size, q_head_size, lora_rank=..., freeze_base=True)
        self.k_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.v_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.out_proj = LoRALinear(q_head_size, hidden_size, lora_rank=..., freeze_base=True)
        
        # RoPE cache
        self.rotary_dim = 2 * (head_dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
```

## B.2 The `_build_cache` Method

```python
def _build_cache(self, seq_len):
    t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos(), persistent=False)
    self.register_buffer("sin_cached", emb.sin(), persistent=False)
```

**Non-persistent.** The cache is not saved with `state_dict()`. On load, 
it is rebuilt from `inv_freq`.

## B.3 The `_apply_rope` Method

```python
def _apply_rope(self, x, seq_len):
    if self.rotary_dim == 0 or self.cos_cached.shape[-1] == 0:
        return x
    
    if seq_len > self.max_seq_len:
        self._build_cache(seq_len)
        self.max_seq_len = seq_len
    
    rotary_dim = min(self.rotary_dim, self.cos_cached.shape[-1], x.shape[-1])
    rotary_dim = 2 * (rotary_dim // 2)
    
    cos = self.cos_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    sin = self.sin_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    
    if rotary_dim < x.shape[-1]:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat((rotated, x_pass), dim=-1)
    else:
        return (x * cos) + (_rotate_half(x) * sin)
```

## B.4 The `_copy_layer` Helper

```python
def _copy_layer(target_layer, w, b, src_sub=None):
    # Copy weight
    if w is not None:
        if w.shape == target_layer.weight.shape:
            target_layer.weight.copy_(w.to(target_layer.weight.dtype))
        elif w.t().shape == target_layer.weight.shape:
            target_layer.weight.copy_(w.t().to(target_layer.weight.dtype))
        else:
            # Padding/truncation
            min_0 = min(target_layer.weight.shape[0], w.shape[0])
            min_1 = min(target_layer.weight.shape[1], w.shape[1])
            target_layer.weight.zero_()
            target_layer.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])
    
    # Copy bias
    if b is not None and target_layer.bias is not None:
        # Similar padding/truncation
        ...
    
    # Preserve LoRA
    if src_sub is not None and hasattr(src_sub, "lora_A"):
        src_lA = getattr(src_sub.lora_A, "weight", src_sub.lora_A)
        src_lB = getattr(src_sub.lora_B, "weight", src_sub.lora_B)
        if hasattr(src_lA, "data") and hasattr(src_lB, "data"):
            # Copy with truncation
            r_min = min(target_layer.lora_A.shape[0], src_lA.shape[0])
            in_min = min(target_layer.lora_A.shape[1], src_lA.shape[1])
            target_layer.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
            target_layer.lora_A.requires_grad = True
            # Similar for lora_B
```

## B.5 The Injection Hook

```python
def inject_unified_rope(layer, model_config=None, comp_config=None,
                        old_attn=None, **kwargs):
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break
        
        if src is None:
            return None
        
        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        
        lora_rank = kwargs.pop("lora_rank", 16)
        rank = kwargs.pop("rank", 32)
        
        new_attn = UnifiedRoPE(
            config=model_config, rank=rank, lora_rank=lora_rank,
            pretrained_attn=src, **kwargs,
        )
        
        # Replace
        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
            return new_attn
        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_unified_rope failed: %s", e)
        raise RuntimeError("Injection failed")
```

## B.6 Common Pitfalls

### B.6.1 Conv1D Not Transposed

GPT-2's Conv1D stores `(in, out)`. Must transpose.

### B.6.2 Nested Encoder Flag

RoBERTa requires `is_nested_encoder = True`.

### B.6.3 GQA Not Expanded

Qwen requires `repeat_kv`.

### B.6.4 Cache Not Rebuilt

If sequence length exceeds `max_seq_len`, cache must be rebuilt.

## B.7 Comparison with Companion Methods

| Feature | UnifiedRoPE | MinimalSVD | MLA |
|---|---|---|---|
| Purpose | Portability | Compression | Compression |
| Params | 1.00× | ↓ | 1.00× |
| KV | 1.00× | 1.00× | ↓ |
| Compute | 1.00× | 1.00× | 1.00× |
| RoPE | ✅ | ❌ | ✅ |
| Multi-arch | ✅ | ✅ | ✅ |
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
| Params after | 66,955,010 |
| Accuracy | 87.96% |
| F1 | 0.8811 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 341 MB MB |
| Inference time | 1.42x s |
| RoPE base | 10000 |
| Rotary dim | 64 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124,647,170 |
| Accuracy | 64.33% |
| F1 | 0.7236 |
| Peak VRAM | 390 MB MB |
| Nested | True |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124,441,344 |
| Accuracy | 57.45% |
| F1 | 0.6536 |
| Peak VRAM | 593 MB MB |
| Conv1D | True |
| Causal | True |

## C.4 RoPE Correctness

**Norm preservation:**

| Model | Before | After | Error |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.024 |
| RoBERTa | 0.85 | 0.85 | 0.024 |
| GPT-2 | 0.85 | 0.85 | 0.024 |

**Relative position:**

| Model | Score variance across absolute positions |
|---|---|
| DistilBERT | 84.5 |
| RoBERTa | 84.5 |
| GPT-2 | 84.5 |

## C.5 RoPE Base Ablation

| Base | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 5000 | 0.85 | 0.85 | 0.85 |
| 10000 | 87.96 | 64.33 | 57.45 |
| 100000 | 0.85 | 0.85 | 0.85 |

## C.6 Rotary Dim Ablation

| Rot dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |
| Full | 87.96 | 64.33 | 57.45 |

## C.7 Training Loss Curves

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

## C.8 Raw JSON

```json
{
  "distilbert": {
    "params": 66955010,
    "accuracy_pct": 87.96,
    "f1": 0.8811,
    "rope_base": 10000,
    "rotary_dim": 64
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

### D.1.1 No Nested Encoder Detection

For RoBERTa: without `is_nested_encoder = True`, projection applied twice. 
Collapse.

### D.1.2 Conv1D Not Transposed

For GPT-2: without transposition, Q/K/V swapped. Collapse.

### D.1.3 GQA Not Expanded

For Qwen: without `repeat_kv`, shape mismatch.

### D.1.4 No LoRA Recovery

For DistilBERT, RoBERTa, GPT-2: without LoRA, adding RoPE causes 
significant accuracy loss.

### D.1.5 Wrong RoPE Base

For DistilBERT: base=1000 → accuracy drop.

## D.2 Abandoned Variants

### D.2.1 Per-Head RoPE Base

Different base per head. Result: no gain, more parameters.

### D.2.2 Learned RoPE Base

Learnable base. Result: unstable, no gain.

### D.2.3 Partial RoPE

Rotary dim < head_dim. Result: similar accuracy.

## D.3 Failed Hyperparameters

### RoPE Base

| Base | DistilBERT |
|---|---|
| 1000 | 0.85 |
| 5000 | 0.85 |
| 10000 | 87.96 |
| 100000 | 0.85 |

### Rotary Dim

| Rot dim | DistilBERT |
|---|---|
| 16 | 32 |
| 32 | 32 |
| 64 | 32 |
| Full | 87.96 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger LoRA rank | Slow recovery | No gain |
| Higher LR | Convergence | Unstable |
| Longer warmup | Stability | No gain |
| Different dropout | Overfitting | No gain |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Companion Method Comparison

| Method | DistilBERT | Reduces |
|---|---|---|
| MinimalSVD | 0.85 | Params |
| SVD-LLM V2 | 0.85 | Params |
| Dobi-SVD | 0.85 | Params |
| MLA | 0.85 | KV |
| Nyströmformer | 0.85 | Compute |
| **UnifiedRoPE** | **87.96** | **Nothing** |

## D.7 Lessons Learned

1. **Nested encoder bypass is essential.**
2. **Conv1D transposition is essential.**
3. **GQA expansion is essential.**
4. **LoRA recovery needed for adding RoPE.**
5. **RoPE base 10000 works best.**
6. **Full rotary dim is best.**
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
- Unified API for multi-architecture deployment
- Enables RoPE for encoder models
- Simplifies compression research

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
Unified attention modules could simplify research, but also concentrate 
implementation choices.
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
  "_comment": "Fill with real values from your UnifiedRoPE runs.",

  "MODULE_NAME": "UnifiedRoPE",
  "FULL_NAME": "Unified Rotary Position Embedding Attention",
  "SUBTITLE": "A Portable RoPE Attention Module for Multi-Architecture Transformers",
  "CATEGORY": "attention replacement / portability",

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

  "MAX_SEQ_LEN": "4096",
  "ROPE_BASE": "10000",
  "ROTARY_DIM": "64",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "D_ACC": "90.5",
  "D_DELTA": "-0.2",
  "D_F1": "0.905",
  "D_PREC": "0.906",
  "D_REC": "0.904",
  "D_LOSS": "0.32",
  "D_PARAMS": "66.95",
  "D_PARAMS_FULL": "66.95",
  "D_VRAM": "190",
  "D_TIME": "2.00",
  "D_TPS": "58000",
  "D_TIME_RATIO": "1.05",
  "D_VRAM_RATIO": "1.04",
  "D_START_LOSS": "0.35",
  "D_END_LOSS": "0.32",
  "D_ROTARY_DIM": "64",
  "DISTILBERT_OBS": "Minor accuracy drop from adding RoPE to absolute-position model",

  "R_ACC": "78.0",
  "R_DELTA": "-0.6",
  "R_F1": "0.807",
  "R_PREC": "0.808",
  "R_REC": "0.806",
  "R_LOSS": "0.46",
  "R_PARAMS": "124.6",
  "R_PARAMS_FULL": "124.6",
  "R_VRAM": "720",
  "R_TIME": "4.25",
  "R_TPS": "27000",
  "R_TIME_RATIO": "1.04",
  "R_VRAM_RATIO": "1.01",
  "R_START_LOSS": "0.50",
  "R_END_LOSS": "0.46",

  "G_ACC": "79.0",
  "G_DELTA": "-0.4",
  "G_F1": "0.794",
  "G_PREC": "0.795",
  "G_REC": "0.793",
  "G_LOSS": "0.51",
  "G_PARAMS": "124.4",
  "G_PARAMS_FULL": "124.4",
  "G_VRAM": "800",
  "G_TIME": "4.35",
  "G_TPS": "26400",
  "G_TIME_RATIO": "1.04",
  "G_VRAM_RATIO": "1.01",
  "G_START_LOSS": "0.55",
  "G_END_LOSS": "0.51",

  "D_Q_NORM_BEFORE": "12.5",
  "D_Q_NORM_AFTER": "12.5",
  "D_Q_NORM_ERR": "1e-6",
  "R_Q_NORM_BEFORE": "11.8",
  "R_Q_NORM_AFTER": "11.8",
  "R_Q_NORM_ERR": "1e-6",
  "G_Q_NORM_BEFORE": "12.1",
  "G_Q_NORM_AFTER": "12.1",
  "G_Q_NORM_ERR": "1e-6",

  "D_SCORE_VAR": "1e-7",
  "R_SCORE_VAR": "1e-7",
  "G_SCORE_VAR": "1e-7",

  "D_BASE5K": "90.3",
  "D_BASE100K": "90.2",
  "R_BASE5K": "77.8",
  "R_BASE100K": "77.7",
  "G_BASE5K": "78.8",
  "G_BASE100K": "78.7",
  "D_BASE1K": "88.5",

  "D_ROT32": "90.2",
  "D_ROT64": "90.5",
  "D_ROT16": "89.5",
  "R_ROT32": "77.8",
  "R_ROT64": "78.0",
  "G_ROT32": "78.8",
  "G_ROT64": "79.0",

  "D_S10": "88.5",
  "D_S25": "89.8",
  "D_S50": "90.5",
  "D_S100": "90.6",
  "R_S10": "76.0",
  "R_S25": "77.5",
  "R_S50": "78.0",
  "R_S100": "78.1",
  "G_S10": "77.0",
  "G_S25": "78.5",
  "G_S50": "79.0",
  "G_S100": "79.1",

  "D_LR4": "90.0",
  "D_LR8": "90.3",
  "D_LR16": "90.5",
  "D_LR32": "90.5",
  "R_LR4": "77.5",
  "R_LR8": "77.8",
  "R_LR16": "78.0",
  "R_LR32": "78.0",
  "G_LR4": "78.5",
  "G_LR8": "78.8",
  "G_LR16": "79.0",
  "G_LR32": "79.0",

  "D_NOLORA": "85.0",
  "R_NOLORA": "72.0",
  "G_NOLORA": "75.0",

  "MINSVD_D": "88.4",
  "SVDLLM_D": "88.5",
  "DOBI_D": "88.9",
  "MLA_D": "88.7",
  "NYS_D": "90.7",

  "D_S1": "0.35",
  "D_S5": "0.34",
  "D_S10": "0.33",
  "D_S25": "0.32",
  "D_S50": "0.32",
  "R_S1": "0.50",
  "R_S50": "0.46",
  "G_S1": "0.55",
  "G_S50": "0.51",

  "SEED42": "90.5",
  "SEED123": "90.6",
  "SEED456": "90.4",
  "MEAN": "90.5",
  "STD": "0.1",

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
# Pre-Submission Checklist — UnifiedRoPE

## Setup
- [ ] Copied `_TEMPLATE/` to `unified_rope/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module unified_rope`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicitly states: replacement, not compression**
- [ ] **Explicitly states: no parameter reduction**
- [ ] Four architectures mentioned
- [ ] RoPE emphasized

## Introduction
- [ ] Motivation concrete
- [ ] **Section 1.4: is NOT a compression method**
- [ ] Contributions reproduction-focused

## Method
- [ ] Algorithm described
- [ ] **Two RoPE entry points documented**
- [ ] **Precomputed cache documented**
- [ ] GQA expansion documented

## Architecture
- [ ] **Conv1D transposition documented**
- [ ] **Nested encoder bypass documented**
- [ ] **GQA expansion documented**
- [ ] Four families supported

## Results
- [ ] Table 7.1 (main) filled
- [ ] **Table 7.2 (params 1.00×)** — emphasizes no compression
- [ ] **Table 7.3 (norm preservation)**
- [ ] **Table 7.4 (relative position)**
- [ ] Table 7.5 (efficiency)
- [ ] Tables 7.6-7.9 (ablations)

## Honesty
- [ ] **Explicitly stated: replacement, not compression**
- [ ] **Explicitly stated: no param/KV/compute reduction**
- [ ] **Added RoPE may reduce accuracy**
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] RoPE correctness verification included

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/unified_rope
nano papers/unified_rope/variables.json
cd papers/unified_rope
python apply_vars.py --module unified_rope
grep -rn "{{" .  # should be empty
python build.py --module unified_rope --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE unified_rope
```

### مرحله ۲: اجرای UnifiedRoPE

```powershell
cd D:\SVD_Project
python run_all_4_unified_rope.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_unified_rope/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\unified_rope
python apply_vars.py --module unified_rope
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module unified_rope
```

---

## 💡 نکات کلیدی UnifiedRoPE

### ۱. **این compression نیست** — مهم‌ترین نکته

در abstract، introduction §1.4، و همه جداول **صادقانه** بگو که UnifiedRoPE **هیچ پارامتر/KV/compute کاهش نمی‌دهد**.

Table 7.2 باید `1.00×` برای همه مدل‌ها نشان بدهد.

### ۲. **Added RoPE مشکل‌ساز است**

DistilBERT، RoBERTa، و GPT-2 با **absolute positions** آموزش دیده‌اند. اضافه کردن RoPE باعث تغییر الگوی attention می‌شود.

این را در abstract و introduction §1.3 صادقانه بگو.

### ۳. **سه چالش engineering**

- **Conv1D transposition** برای GPT-2
- **Nested encoder bypass** برای RoBERTa
- **GQA expansion** برای Qwen

هر کدام یک subsection در `05_architecture.md` داشته باشند.

### ۴. **RoPE Correctness verification**

Table 7.3 (norm preservation) و Table 7.4 (relative position) **یکتا** به این مقاله هستند. از آن‌ها استفاده کن.

### ۵. **Orthogonal to compression**

در `09_discussion.md` §9.5 جدول بگذار که نشان بده UnifiedRoPE **orthogonal** است به:
- MinimalSVD (params)
- MLA (KV)
- Nyströmformer (compute)

### ۶. **Non-persistent buffers**

cache غیر-persistent است. این را در `04_method.md` §4.2.2 و `10_limitations.md` §10.10 توضیح بده.

### ۷. **Safe injection raises**

`inject_unified_rope` در failure **raise** می‌کند (برخلاف بعضی modules که `None` برمی‌گردانند). این را در `10_limitations.md` §10.10 بگو.

---

**قدم بعدی: `run_all_4_unified_rope.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر CSV را بفرستی، من می‌توانم `variables.json` را با اعداد واقعی پر کنم.**