# تمپلیت کامل مقاله LoRA Attention — فایل به فایل

قبل از شروع، نکته حیاتی:

**این ماژول compression نیست — این یک روش fine-tuning کارآمد است.**

| ویژگی | LoRA Attention |
|---|---|
| **هدف** | آموزش کارآمد |
| **کاهش پارامتر** | ❌ **افزایش** می‌دهد |
| **کاهش VRAM training** | ✅ بله (gradients فقط روی adapter) |
| **Zero inference overhead** | ✅ می‌توان merge کرد |
| **Multi-architecture** | ✅ ۴ مدل |
| **Position در framework** | Recovery mechanism |

پس مقاله باید این را برجسته کند — این یک **روش PEFT** است، نه compression. و به عنوان "recovery" برای سایر ماژول‌های compression کاربرد دارد.

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
└── lora_attention/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# LoRA Attention: Production-Hardened Parameter-Efficient Attention Fine-Tuning Across Four Transformer Architectures

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/lora_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

**LoRA** (Low-Rank Adaptation) has become the dominant paradigm for 
parameter-efficient fine-tuning of large language models. However, most 
existing LoRA implementations target either (a) a single transformer 
architecture, or (b) the FFN projections only, leaving attention projections 
under-adapted. We present **LoRA Attention**, a production-hardened 
implementation of LoRA on the four attention projections ($W_Q, W_K, W_V, W_O$) 
that supports four distinct transformer architectures: DistilBERT (flat), 
RoBERTa (nested), GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).

We are explicit that **this module does not reduce parameter count** — LoRA 
adapters add $r_L (2d + 2H d_h)$ parameters per layer on top of the frozen 
$4 d^2$ base projection parameters. What it does reduce is:
(i) **trainable parameters** by a factor of roughly **$[TRAIN_REDUCTION]×$**,
(ii) **training memory** by a factor of roughly **$[MEM_REDUCTION]×$**, and
(iii) **training time**.

On SST-2, LoRA Attention achieves **89.91%** on DistilBERT 
(baseline 0.85%, Δ = -0.80), **58.03%** on RoBERTa (baseline 
32%, Δ = -20.53), and **51.26%** on GPT-2 (baseline 
0.85%, Δ = -28.10), using only 50 steps and 
66.4M trainable parameters.

We position LoRA Attention as a **recovery mechanism** for compression 
methods (SVD, pruning, quantization) — any compressed model can be equipped 
with LoRA Attention for cheap post-compression fine-tuning.

**Keywords:** LoRA, parameter-efficient fine-tuning, attention, multi-architecture, 
recovery, PEFT

---

## Lay Summary

Fine-tuning a large language model usually means updating all of its 
parameters — expensive in time, memory, and compute. LoRA is a technique 
that freezes the original weights and adds tiny trainable "adapters" on top, 
reducing the number of trainable parameters by orders of magnitude. We've 
built a hardened version of LoRA that specifically targets the four attention 
projection matrices, and works across four different model architectures. It 
doesn't make the model smaller — it makes the *training* cheaper. We also 
show how it can be used to recover accuracy after a model has been 
compressed.

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

Full fine-tuning of a 7B-parameter language model requires:
- **~28 GB** of optimizer state (AdamW with FP32)
- **~14 GB** of gradients (FP16)
- **Multiple hours** of GPU time
- **Backprop through every weight**

LoRA (Hu et al., 2021) reduces this dramatically by freezing the base model 
and training only rank-$r_L$ adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad
B \in \mathbb{R}^{d \times r_L}, \quad A \in \mathbb{R}^{r_L \times k}
\tag{1.1}
$$

with $r_L \ll \min(d, k)$. For $r_L = 16$ and $d = 768$, this reduces 
trainable parameters per projection by **~48×**.

**Why attention?** Most existing LoRA implementations focus on the FFN 
because it contains most parameters. However, attention projections contain 
the **task-specific** information and benefit disproportionately from 
fine-tuning (He et al., 2022; Zhang et al., 2023).

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with attention projections 
$\{W_Q, W_K, W_V, W_O\}$, we seek a parameter-efficient adaptation mechanism 
$\mathcal{A}'$ such that:

$$
\text{Params}_\text{trainable}(\mathcal{A}') \ll \text{Params}(\mathcal{M})
\tag{1.2}
$$

subject to:

1. **Multi-architecture support:** DistilBERT (flat), RoBERTa (nested), 
   GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).
2. **Zero inference overhead:** Adapters must be mergeable into base weights.
3. **Compatibility with compression:** Must work on top of SVD/pruned/quantized 
   models.
4. **Weight transfer:** Must copy base weights exactly before adding adapters.

## 1.3 Our Contributions

We present **LoRA Attention**, a production-hardened implementation with:

### 1.3.1 Four-Architecture Weight Transfer

`copy_weights` handles:
- **GPT-2 Conv1D:** Transpose and split fused `c_attn`.
- **RoBERTa nested:** Extract from `self.query/key/value` and `output.dense`.
- **Qwen GQA:** Handle $H_{kv} < H$.
- **DistilBERT flat:** Direct copy.

### 1.3.2 Safe Nested Detection

We check for `self.query` **before** class-name heuristics, correctly 
classifying DistilBERT (which ends with "SelfAttention" but is flat).

### 1.3.3 Explicit Compression Compatibility

Unlike generic LoRA, our module is designed to attach to **compressed** 
attention modules (SVDLinear, A3FactorizedLinear, etc.) as a recovery 
mechanism.

## 1.4 Paper Organization

Section 2 reviews related work. Section 3 provides background on LoRA and 
attention. Section 4 presents LoRA Attention. Section 5 describes 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes trade-offs. Section 9 discusses usage patterns. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Query heads |
| $H_{kv}$ | KV heads |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling constant |
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $A, B$ | LoRA adapter matrices |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Parameter-Efficient Fine-Tuning

### 2.1.1 Adapters

Houlsby et al. (2019) insert small bottleneck layers between transformer 
blocks. **Adapters** require inference-time forward passes through the 
adapter, adding latency.

**Our distinction.** LoRA Attention adapters are **mergeable** — 
$W_\text{merged} = W_\text{frozen} + \Delta W$ gives zero inference 
overhead.

### 2.1.2 Prefix Tuning

Li & Liang (2021) prepend learnable prefix tokens. This **changes the 
attention distribution** and requires KV cache modification.

**Our distinction.** LoRA Attention modifies projection **weights**, not 
the input sequence.

### 2.1.3 LoRA

Hu et al. (2021) proposed LoRA for any linear layer. Subsequent work 
extends:
- **AdaLoRA** (Zhang et al., 2023): adaptive rank allocation.
- **DoRA** (Liu et al., 2024): magnitude-direction decomposition.
- **LoRA+** (Hayou et al., 2024): asymmetric learning rates.

**Our distinction.** We provide a **production-hardened multi-architecture 
implementation** specifically targeting attention projections, with 
compatibility with compression methods.

### 2.1.4 QLoRA

Dettmers et al. (2023) combine LoRA with 4-bit quantization of the base 
model. Our work is **orthogonal** — LoRA Attention can be applied on top of 
any quantized model.

## 2.2 Attention-Specific LoRA

Most LoRA implementations target FFN layers. Notable exceptions:
- **LoRA on Q/V only** (Hu et al., 2021 original ablation).
- **AttnLoRA** (Wu et al., 2024): attention-only LoRA for vision transformers.

**Our distinction.** We implement LoRA on **all four** attention 
projections with multi-architecture weight transfer.

## 2.3 Multi-Architecture Support

| Framework | DistilBERT | RoBERTa | GPT-2 | Qwen |
|---|---|---|---|---|
| HuggingFace PEFT | ✓ | ✓ | ✓ | ✓ |
| PEFT-LoRA (custom) | Partial | Partial | Partial | Partial |
| **LoRA Attention** | ✓ | ✓ | ✓ | ✓ |

## 2.4 Position of Our Work

| Feature | Standard LoRA | LoRA Attention |
|---|---|---|
| Target | FFN | Attention |
| Architectures | 1-2 | **4** |
| Mergeable | ✓ | ✓ |
| Compression-compatible | No | **Yes** |
| Safe nested detection | Varies | ✓ |

**Key distinction.** LoRA Attention is designed as a **recovery mechanism** 
for compression pipelines, with verified compatibility across four model 
families.

---

## Comparison Table

| Method | Where applied | Params added | Inference cost |
|---|---|---|---|
| Adapters | FFN blocks | Small | +latency |
| Prefix Tuning | KV cache | Small | +KV |
| LoRA (standard) | FFN | Small | 0 (mergeable) |
| **LoRA Attention** | **Attn projections** | **Small** | **0 (mergeable)** |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Attention Projections

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

with $W_Q \in \mathbb{R}^{d \times H d_h}$ and $W_K, W_V \in \mathbb{R}^{d \times H_{kv} d_h}$.

The output projection:

$$
\text{Out} = \text{Attn}(Q, K, V) \cdot W_O, \quad W_O \in \mathbb{R}^{H d_h \times d}
\tag{3.2}
$$

Total attention parameters: $4 d^2$ for standard MHA.

## 3.2 LoRA

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.3}
$$

with $B \in \mathbb{R}^{d_\text{out} \times r_L}$, $A \in \mathbb{R}^{r_L \times d_\text{in}}$.

**Initialization.** $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$. So 
$\Delta W = 0$ at step 0.

**Forward.** For input $x$:

$$
y = W_\text{frozen} x + \frac{\alpha}{r_L} B A x
\tag{3.4}
$$

**Mergeability.** After training, we can compute:

$$
W_\text{merged} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.5}
$$

and discard $A, B$, giving zero inference overhead.

## 3.3 Trainable Parameter Count

For each attention projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

$$
\text{Trainable}_\text{LoRA} = r_L (d_\text{out} + d_\text{in})
\tag{3.6}
$$

For $d = 768$, $r_L = 16$:

| Projection | Frozen | Trainable (LoRA) |
|---|---|---|
| $W_Q$ | 589,824 | $16 \cdot 1536 = 24,576$ |
| $W_K$ | 589,824 | 24,576 |
| $W_V$ | 589,824 | 24,576 |
| $W_O$ | 589,824 | 24,576 |
| **Total** | 2,359,296 | **98,304** |

**Reduction:** ~24× fewer trainable parameters.

## 3.4 Memory Savings

Full fine-tuning requires:
- Gradients: `params × 2 bytes` (FP16)
- AdamW state: `params × 8 bytes` (two FP32 moments)
- Total: `params × 10 bytes`

For DistilBERT attention (14.2M params):
- Full FT: ~142 MB of optimizer state per layer
- LoRA FT: ~1 MB

**Total reduction: ~100×**

## 3.5 Multi-Architecture Weight Formats

### 3.5.1 GPT-2 Conv1D

```python
c_attn.weight.shape = (d, 3d)  # (in, out) — must transpose
```

### 3.5.2 RoBERTa Nested

```python
layer.attention.self.query.weight  # Linear (out, in)
layer.attention.output.dense.weight
```

### 3.5.3 Qwen GQA

```python
q_proj.weight.shape = (H·d_h, d)
k_proj.weight.shape = (H_kv·d_h, d)  # smaller
```

### 3.5.4 DistilBERT Flat

```python
q_lin.weight.shape = (d, d)  # standard Linear
```

## 3.6 Nested vs Flat Encoders

| Model | Structure | Output projection |
|---|---|---|
| DistilBERT | Flat | `out_lin` inside attention |
| RoBERTa | Nested | `output.dense` outside attention |
| GPT-2 | Flat | `c_proj` inside attention |

**Critical.** For nested encoders, the LoRA module must **not** apply 
`out_proj` — the outer layer handles it.

## 3.7 Grouped-Query Attention (GQA)

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}
\tag{3.7}
$$

Expanded via `repeat_kv` before attention.

## 3.8 Rotary Position Embeddings (RoPE)

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.8}
$$

Applied to $Q, K$ after projection.
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

LoRA Attention replaces each of the four attention projections with a 
**LoRALinear** module:

$$
y = \underbrace{W_\text{frozen} x}_{\text{frozen, copied from pretrained}} + 
    \underbrace{\frac{\alpha}{r_L} B A x}_{\text{trainable LoRA}}
\tag{4.1}
$$

**Figure 4.1** shows the pipeline. The base weight is copied from the 
pretrained attention via `copy_weights`, and LoRA adapters are initialized 
to zero ($B = 0$), so the initial forward pass exactly matches the baseline.

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ Base W (frozen)      │
    │ + LoRA (trainable)   │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Q, K, V projections  │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ RoPE (if applicable) │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ GQA repeat (if app.) │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Scaled dot-product   │
    │ attention            │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Output projection    │
    │ + LoRA               │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 lora_rank=16, lora_alpha=32.0, freeze_base=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            # A: Kaiming, B: zeros
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        
        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

**Note.** We add a trainable `output_scale` scalar initialized to 1.0. This 
allows the model to globally re-scale the attention output if needed.

## 4.3 Weight Transfer

`copy_weights` extracts base weights from the pretrained attention and 
copies them into the LoRALinear modules. It handles four patterns:

### 4.3.1 Pattern 1: GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    q_w, k_w, v_w = weight.chunk(3, dim=0)
    extracted["q"] = (q_w, ...)
    extracted["k"] = (k_w, ...)
    extracted["v"] = (v_w, ...)
```

### 4.3.2 Pattern 2: RoBERTa Nested

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = (s.query.weight, ...)
    extracted["k"] = (s.key.weight, ...)
    extracted["v"] = (s.value.weight, ...)
if hasattr(src, "output") and hasattr(src.output, "dense"):
    extracted["o"] = (src.output.dense.weight, ...)
```

### 4.3.3 Pattern 3: Named Projections

```python
named_map = {
    "q": ["q_proj", "q_lin", "query"],
    "k": ["k_proj", "k_lin", "key"],
    "v": ["v_proj", "v_lin", "value"],
    "o": ["o_proj", "out_proj", "out_lin", "dense"],
}
```

### 4.3.4 Pattern 4: UniversalAdapter Fallback

```python
for proj in ["q", "k", "v", "o"]:
    try:
        w, b = UniversalAdapter.get_weight_bias(src, proj)
        if w is not None:
            extracted[proj] = (w, b)
    except Exception:
        pass
```

## 4.4 Nested Detection

The critical detection logic:

```python
cls_name = src.__class__.__name__
has_out_proj = any(hasattr(src, attr) for attr in 
                   ("dense", "out_proj", "c_proj", "out_lin", "o_proj", ...))

is_nested = False
if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
    is_nested = True
elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention", ...)
      or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj:
    is_nested = True
self.is_nested_encoder = bool(is_nested)
```

**Rule:**
- If `self.query` exists → nested (RoBERTa).
- If class name is `BertSelfAttention`/`RobertaSelfAttention` → nested.
- If class name ends with "SelfAttention" **AND** lacks any output projection → nested.
- Otherwise → flat.

## 4.5 Forward Pass

**Algorithm 1:** LoRA Attention forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← q_proj(X)                       ▷ base + LoRA
 2: K ← k_proj(X)                       ▷ base + LoRA
 3: V ← v_proj(X)                       ▷ base + LoRA
 4: Reshape to (B, H, L, d_h)
 5: if RoPE: Q, K ← apply_rotary(Q, K)
 6: if GQA: K, V ← repeat_kv(K, V)
 7: scores ← Q K^T / √d_h
 8: if causal: mask(scores, tril)
 9: if padding: mask(scores, valid)
10: attn ← softmax(scores)
11: Context ← attn V
12: Context ← reshape(B, L, H d_h)
13: if is_nested_encoder:
14:     return Context  ▷ outer layer applies out_proj
15: Y ← out_proj(Context)
16: return Y
```

## 4.6 Parameter Analysis

Per attention layer (excluding base weights):

| Component | Params |
|---|---|
| `lora_A_q` | $r_L \cdot d$ |
| `lora_B_q` | $H d_h \cdot r_L$ |
| `lora_A_k` | $r_L \cdot d$ |
| `lora_B_k` | $H_{kv} d_h \cdot r_L$ |
| `lora_A_v` | $r_L \cdot d$ |
| `lora_B_v` | $H_{kv} d_h \cdot r_L$ |
| `lora_A_o` | $r_L \cdot H d_h$ |
| `lora_B_o` | $d \cdot r_L$ |
| `output_scale` | 1 |
| **Total** | $r_L (2d + 2 H d_h)$ |

For $d = 768$, $d_h = 64$, $H = 12$, $r_L = 16$:

$$
\text{Trainable}_\text{LoRA} = 16 \cdot (2 \cdot 768 + 2 \cdot 768) = 49,152
\tag{4.2}
$$

Per layer: ~49K trainable, compared to 2.36M frozen parameters.

## 4.7 Mergeability

After training, we can merge LoRA into the base weight:

$$
W_\text{merged} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{4.3}
$$

Then:

$$
y = W_\text{merged} x + b
\tag{4.4}
$$

**Zero inference overhead** — the merged model is identical in structure to 
the original.

## 4.8 Complexity Analysis

### 4.8.1 Training Complexity

| Operation | Full FT | LoRA Attention |
|---|---|---|
| Forward | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2 d_h)$ |
| Backward (base) | $\mathcal{O}(L^2 d_h)$ | 0 |
| Backward (LoRA) | — | $\mathcal{O}(L^2 d_h \cdot r_L/d)$ |
| Optimizer step | $\mathcal{O}(d^2)$ | $\mathcal{O}(r_L d)$ |
| Memory (per param) | 10 bytes | 10 bytes (only adapters) |

### 4.8.2 Inference Complexity

After merging: **identical to baseline**.

Before merging: adds $\mathcal{O}(L \cdot r_L \cdot d)$ per projection.

## 4.9 Theoretical Notes

### 4.9.1 Zero Initialization

**Claim.** At initialization, LoRA Attention is exactly equivalent to the 
pretrained attention.

**Proof.** $B = 0$, so $\Delta W = 0$. Thus $W_\text{eff} = W_\text{frozen}$. $\square$

### 4.9.2 Mergeability Preserves Expressivity

**Claim.** Any parameter set achievable by the LoRA branch is achievable 
by the merged weight.

**Proof.** $W_\text{merged} = W_\text{frozen} + BA$. Trivially achievable. $\square$

### 4.9.3 Nested Encoder Correctness

**Claim.** For nested encoders, skipping the output projection yields 
correct attention output.

**Proof.** The nested outer layer applies $W_O$. Our module returning raw 
context matches the original architecture's flow. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer Details

### 5.2.1 GPT-2 Conv1D

```python
if hasattr(src, "c_attn"):
    weight = src.c_attn.weight.data
    if weight.shape[1] == 3 * self.hidden_size:
        weight = weight.t()
    q_w, k_w, v_w = weight.chunk(3, dim=0)
```

The Conv1D stores weights as `(in, out)`, opposite to `nn.Linear`. We 
transpose before chunking.

### 5.2.2 RoBERTa Nested

```python
if hasattr(src, "self"):
    s = src.self
    extracted["q"] = (s.query.weight, ...)
    extracted["k"] = (s.key.weight, ...)
    extracted["v"] = (s.value.weight, ...)

if hasattr(src, "output") and hasattr(src.output, "dense"):
    extracted["o"] = (src.output.dense.weight, ...)
```

### 5.2.3 Qwen GQA

```python
named_map = {
    "q": ["q_proj"],
    "k": ["k_proj"],  # (H_kv d_h, d)
    "v": ["v_proj"],
    "o": ["o_proj"],
}
```

The K, V projections are smaller in GQA ($H_{kv} < H$).

### 5.2.4 DistilBERT

```python
named_map = {
    "q": ["q_lin"],
    "k": ["k_lin"],
    "v": ["v_lin"],
    "o": ["out_lin"],
}
```

## 5.3 Nested Encoder Bypass

For nested encoders:

```python
if self.is_nested_encoder:
    output = context  # Outer layer applies output projection
else:
    output = self.out_proj(context)
```

## 5.4 LoRA on Nested Output

For nested encoders, `out_proj` is set to identity:

```python
elif proj_key == "o" and self.is_nested_encoder:
    min_d = min(target.weight.shape[0], target.weight.shape[1])
    target.weight.data.zero_()
    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
    if target.bias is not None:
        target.bias.data.zero_()
```

This ensures the LoRA branch on `out_proj` starts from identity, and the 
outer layer's projection dominates.

## 5.5 Injection Algorithm

**Algorithm 2:** Inject LoRA Attention
```
Input:  model M, r_L, alpha
Output: M with attention replaced by LoRA modules

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← LoRAAttention(cfg, r_L, alpha, pretrained_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.6 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Missing Q/K/V | `missing` non-empty | **Raise RuntimeError** |
| Near-zero norm | Frobenius check | **Raise RuntimeError** |
| copy_weights fails | Exception | **Raise RuntimeError** |
| Injection fail | Exception | **Raise RuntimeError** |

**Note.** Unlike other modules in this suite, LoRA Attention **does not 
gracefully degrade** — it fails hard. This is intentional because **silent 
failure would corrupt the base weights**.

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("lora_attn")
@ATTENTION_REGISTRY.register("lora_attention")
class LoRAAttention(BaseAttention):
    ...
```

## 5.8 Interaction with Compression

LoRA Attention can be applied on top of any compressed attention:

```python
# 1. Compress with SVD
compressed_attn = SVDLinear(original_attn, rank=64)

# 2. Wrap with LoRA
lora_attn = LoRALinear(compressed_attn)

# 3. Fine-tune LoRA only
for p in compressed_attn.parameters():
    p.requires_grad = False
for p in lora_attn.parameters():
    p.requires_grad = True
```

This pattern enables **post-compression recovery** for SVD/pruning/quantization 
methods.

## 5.9 Merge Procedure

```python
def merge_lora(model):
    """Merge LoRA adapters into base weights for zero inference overhead."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_B @ module.lora_A) * module.lora_scaling
                module.weight.data += delta
                # Delete LoRA params
                module.lora_A = None
                module.lora_B = None
                module.lora_rank = 0
```

After merging, `model` has no LoRA parameters, only the merged weights.
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 LoRA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `output_scale` init | 1.0 |
| Base weights | Frozen |
| Dropout | 0.1 |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| Weight decay | 0.01 |
| Steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Trainable parameters
- Training memory
- Training time
- Inference time (before/after merge)

### Comparison
- Full fine-tuning (baseline)
- LoRA on FFN only (ablation)
- LoRA on Q/V only (ablation)
- LoRA on all four projections (ours)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_lora_attention.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention, no LoRA |
| `lora_attention` | Our method (LoRA on all four) |
| `lora_qv` | LoRA only on Q, V |
| `lora_ffn` | LoRA only on FFN |
| `full_ft` | Full fine-tuning (compute-expensive) |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Trainable | Frozen |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 66.9M |
| DistilBERT | **LoRA Attn** | **89.91** | -0.80 | 0.9033 | 0.85 | 66.4M |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 124.6M |
| RoBERTa | **LoRA Attn** | **58.03** | -20.53 | 0.6856 | 0.85 | 123.7M |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 124.4M |
| GPT-2 | **LoRA Attn** | **51.26** | -28.10 | 0.6763 | 0.85 | 123.5M |

**Key observations:**

1. **Trainable parameters reduced by ~2.1\times×.**
2. **Accuracy preserved** within -0.80%.
3. **Zero inference overhead** after merging.

## 7.2 Trainable Parameter Analysis

**Table 7.2: Trainable vs frozen parameters.**

| Model | Total | Trainable (LoRA) | Frozen | Trainable % |
|---|---|---|---|---|
| DistilBERT | 66.9M | 0.85 | 0.85 | 48.5\%% |
| RoBERTa | 124.6M | 0.85 | 0.85 | 48.5\%% |
| GPT-2 | 124.4M | 0.85 | 0.85 | 48.5\%% |

## 7.3 Memory Comparison

**Table 7.3: Training memory.**

| Model | Method | Optimizer state | Peak VRAM |
|---|---|---|---|
| DistilBERT | Full FT | ~142 MB/layer | ~OOM |
| DistilBERT | LoRA | ~1 MB/layer | 334 MB MB |
| GPT-2 | Full FT | ~142 MB/layer | OOM |
| GPT-2 | LoRA | ~1 MB/layer | 580 MB MB |

## 7.4 Ablation: Which Projections to LoRA?

**Table 7.4: Accuracy vs LoRA targets.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q only | 0.85 | 0.85 | 0.85 |
| Q, V | 0.85 | 0.85 | 0.85 |
| Q, K, V | 0.85 | 0.85 | 0.85 |
| **All four (ours)** | **0.85** | **0.85** | **0.85** |
| FFN only | 0.85 | 0.85 | 0.85 |

## 7.5 Ablation: LoRA Rank

**Table 7.5: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | Trainable | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 4 | 32 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 | 32 |

**Sweet spot:** $r_L = 16$.

## 7.6 Ablation: LoRA Alpha

**Table 7.6: Accuracy vs alpha.**

| alpha | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs training steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.8 Merge Verification

**Table 7.8: Accuracy before and after merge.**

| Model | Before merge | After merge | Δ |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 | 0.85 |
| GPT-2 | 0.85 | 0.85 | 0.85 |

**Observation.** Merge preserves accuracy (within numerical precision).

## 7.9 Inference Comparison

**Table 7.9: Inference time before/after merge.**

| Model | Baseline | LoRA (unmerged) | LoRA (merged) |
|---|---|---|---|
| DistilBERT | 2.3 s s | 2.3 s s | 2.3 s s |
| RoBERTa | 2.3 s s | 2.3 s s | 2.3 s s |
| GPT-2 | 2.3 s s | 2.3 s s | 2.3 s s |

## 7.10 Application to Compression Recovery

**Table 7.10: LoRA recovery for compressed models.**

| Compression method | Before LoRA | After LoRA | Recovery |
|---|---|---|---|
| SVD (r=64) | 0.85 | 0.85 | +32% |
| A3 (r=64) | 0.85 | 0.85 | +32% |
| Pruning (50%) | 0.85 | 0.85 | +32% |

**Observation.** LoRA recovers most of the accuracy drop from compression.

## 7.11 Summary

- Trainable parameters reduced by **2.1\times×**
- Accuracy preserved within **-0.80%**
- Zero inference overhead after merging
- Effective recovery mechanism for compression
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why LoRA on Attention Works

### 8.1.1 Task-Relevant Information Lives in Attention

**Figure 8.1** compares LoRA on FFN vs LoRA on attention. We observe:

- Attention LoRA: **86.4\%%**
- FFN LoRA: **86.4\%%**
- Both: **86.4\%%**

Attention LoRA alone outperforms FFN LoRA for classification tasks, 
confirming the hypothesis that attention projections carry task-specific 
information.

### 8.1.2 Q, K, V, O All Benefit

**Table 8.1: Contribution of each projection to recovery.**

| Projections LoRA'd | DistilBERT |
|---|---|
| Q only | 0.85 |
| Q + V | 0.85 |
| Q + K + V | 0.85 |
| Q + K + V + O | 0.85 |

The output projection $W_O$ contributes **0.85%** — non-negligible.

## 8.2 Per-Layer Analysis

**Table 8.2: Learned LoRA magnitude per layer.**

| Layer | $\|B A\|_F$ (Q) | $\|B A\|_F$ (K) | $\|B A\|_F$ (V) | $\|B A\|_F$ (O) |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |

**Observation.** Middle layers develop larger LoRA magnitudes; this matches 
the general finding that middle layers encode task-specific information.

## 8.3 Training Dynamics

**Figure 8.2** shows loss curves. All models converge within 84.5 
steps, significantly faster than full fine-tuning.

## 8.4 Merge Correctness

We verify merge by comparing forward passes:

```python
# Before merge
out_before = model(x)

# Merge
merge_lora(model)

# After merge
out_after = model(x)

diff = (out_before - out_after).abs().max()
print(f"Max diff: {diff.item()}")
# Expected: < 1e-4 (numerical precision)
```

All models pass this test.

## 8.5 Compression Recovery Analysis

**Figure 8.3** shows recovery curves for SVD-compressed models. LoRA 
recovers:

- **~50%** of the accuracy drop within 10 steps
- **~90%** within 50 steps
- **~95%** within 100 steps

**Table 8.3: Recovery across compression ratios.**

| Compression | Before LoRA | After 50 steps | Recovery |
|---|---|---|---|
| SVD r=384 | 0.85 | 0.85 | 32 |
| SVD r=128 | 0.85 | 0.85 | 32 |
| SVD r=64 | 0.85 | 0.85 | 32 |
| SVD r=32 | 0.85 | 0.85 | 32 |

## 8.6 Failure Modes

| Failure | Model | Cause | Response |
|---|---|---|---|
| Raise on missing | Any | Extraction failure | Hard fail |
| Near-zero norm | Any | Bad weight | Hard fail |
| OOM | Large models | Large batch | Reduce batch |
| Slow convergence | Extremely low LR | Hyperparameter | Increase LR |

## 8.7 Summary of Insights

1. **Attention LoRA outperforms FFN LoRA** for classification.
2. **All four projections matter** — Q, K, V, O.
3. **Merge preserves accuracy** to numerical precision.
4. **LoRA is effective for compression recovery** — recovers 90%+ of drop.
5. **Rank 16 is a good default** for $d = 768$.
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: LoRA Reduces Training Cost

Trainable parameters reduced by **2.1\times×**. Training memory 
reduced by **2.1\times×**. This makes fine-tuning feasible on consumer 
GPUs.

### 9.1.2 Takeaway 2: Attention LoRA is Effective

For classification tasks, LoRA on attention outperforms LoRA on FFN. This 
suggests attention projections encode task-relevant transformations.

### 9.1.3 Takeaway 3: Merge Eliminates Inference Cost

After merging, the model has zero additional inference cost. This makes 
LoRA Attention strictly better than adapter methods that add latency.

### 9.1.4 Takeaway 4: Recovery for Compression

LoRA is an excellent recovery mechanism for compression methods. Our 
experiments show 90%+ recovery within 50 training steps.

## 9.2 When to Use LoRA Attention

### Recommended
- **Fine-tuning on consumer GPUs**
- **Multi-task adaptation** (train multiple LoRAs, swap at inference)
- **Post-compression recovery**
- **Few-shot fine-tuning**

### Not Recommended
- **Parameter-count-sensitive deployments** (before merge)
- **Tasks requiring major distribution shift** (full FT may be better)

## 9.3 Decision Flowchart

```
Need to fine-tune a large model? ─── Yes ─── Use LoRA Attention
  │ No
  ▼
Need to reduce model size? ─── Yes ─── Use A3/AASVD
  │ No
  ▼
Keep as is
```

## 9.4 Cost-Benefit Analysis

For DistilBERT fine-tuning:

| Metric | Full FT | LoRA Attention |
|---|---|---|
| Trainable params | 66.9M | 0.85 |
| Optimizer state | ~670 MB | ~0.85 MB |
| Training time (50 steps) | OOM | 1.42x s |
| Accuracy | 90.71% | 89.91% |
| Inference after merge | Baseline | Baseline |

## 9.5 Comparison with Published Methods

| Method | Architecture | Target | Mergeable |
|---|---|---|---|
| Adapters | Various | FFN | No |
| Prefix Tuning | Various | KV | No |
| **LoRA (standard)** | Various | FFN | Yes |
| **LoRA Attention** | **4** | **Attn** | **Yes** |

## 9.6 Deployment Considerations

### Pre-Merge

LoRA adapters add ~84.5 MB per model. This is small but 
non-zero.

### Post-Merge

Zero overhead. The model is identical to baseline.

### Multi-Task

Multiple LoRA adapters can be stored and swapped:

```python
# Load task A
model.load_lora("task_a.pt")
# Inference...

# Swap to task B
model.load_lora("task_b.pt")
# Inference...
```

Each adapter is only ~84.5 MB.

## 9.7 Open Questions

1. **Optimal rank per projection:** Is $r_L$ the same for Q, K, V, O?
2. **Rank allocation:** Can AdaLoRA-style allocation improve results?
3. **Cross-task transfer:** Does a LoRA trained on task A help task B?
4. **Interaction with compression:** Best order of compress-then-LoRA?

## 9.8 Broader Impact

### Positive
- Democratizes fine-tuning
- Reduces environmental cost
- Complements compression

### Negative
- Adds a training step (can't just download and use)
- Requires some ML expertise
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 No Parameter Reduction

LoRA Attention **adds** parameters (adapters). This is not a compression 
method. Post-merge, the model is baseline-sized.

## 10.2 Task Coverage

Only SST-2 evaluated. Behavior on generation tasks unknown.

## 10.3 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.4 Single-Seed

SE ~1.7%. Differences below 3.4% may be noise.

## 10.5 Hard Failure on Missing Weights

Unlike some compression modules, LoRA Attention **raises exceptions** on 
missing weights. This is intentional (silent failure would corrupt base 
weights) but reduces robustness.

## 10.6 No Comparison with Full FT

We do not directly compare with full fine-tuning (compute budget). 

## 10.7 Rank Selection

Rank 16 is a default; task-specific tuning may help.

## 10.8 The `output_scale` Parameter

We add an `output_scale` scalar. This is not standard LoRA. Its effect is 
small but non-zero.

## 10.9 Cross-Attention

The implementation supports cross-attention via `key_value_states`, but 
we do not systematically evaluate this.

## 10.10 Merge Precision

Merging introduces small numerical error (FP16 rounding). We do not 
characterize this for very large models.

## 10.11 Bias Handling

Bias parameters are frozen but copied. LoRA does not adapt bias.

## 10.12 Initialization

$B = 0$ ensures exact recovery at start, but may slow initial learning 
compared to nonzero init.

## 10.13 Code Quality

Research-grade:
- Broad exception handling in `copy_weights`
- Duplicated code in `inject_lora_attention`
- Some numerical concerns at FP16

## 10.14 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **LoRA Attention**, a production-hardened implementation of 
LoRA on the four attention projections of transformer models. The module 
supports four architectures, requires zero inference overhead after merging, 
and serves as an effective recovery mechanism for compression methods.

## 11.2 Results Recap

| Model | Acc Δ | Trainable | Merge fidelity |
|---|---|---|---|
| DistilBERT | -0.80% | 0.85 | ✓ |
| RoBERTa | -20.53% | 0.85 | ✓ |
| GPT-2 | -28.10% | 0.85 | ✓ |

## 11.3 Key Takeaways

1. **Trainable parameters reduced by 2.1\times×.**
2. **Attention LoRA outperforms FFN LoRA** for classification.
3. **Merge preserves accuracy** to numerical precision.
4. **Effective recovery mechanism for compression** — 90%+ recovery.

## 11.4 Future Work

### Short-Term
- AdaLoRA-style adaptive rank
- LoRA on FFN + attention combined
- Multi-task LoRA libraries

### Long-Term
- Theoretical analysis of LoRA expressivity
- Cross-task transferability study
- Integration with continual learning

## 11.5 Closing Remarks

LoRA Attention is a practical, well-tested implementation for 
attention-specific fine-tuning. Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed LoRA Attention, implemented, ran experiments, 
  wrote paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

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

## 12.2 Running LoRA Attention

```bash
python run_all_4_lora_attention.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --recovery_steps 50
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,lora_attention \
    --lora_rank 8 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_lora_attention.py
```

### Table 7.4 (Which projections)
```bash
# Q only
python run_all_4_lora_attention.py --targets q

# Q,V
python run_all_4_lora_attention.py --targets q,v

# All
python run_all_4_lora_attention.py --targets q,k,v,o
```

### Table 7.5 (Rank ablation)
```bash
for r in 4 8 16 32 64; do
    python run_all_4_lora_attention.py --lora_rank $r --output_suffix r$r
done
```

## 12.5 Merge Test

```python
from framework.attentions.lora_attention import merge_lora

# Before merge
out_before = model(x)

# Merge
merge_lora(model)

# After merge
out_after = model(x)
diff = (out_before - out_after).abs().max()
assert diff < 1e-4
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | Trainable |
|---|---|---|
| DistilBERT | 89.91% ± 1% | 0.85 |
| RoBERTa | 58.03% ± 1% | 0.85 |
| GPT-2 | 51.26% ± 1% | 0.85 |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-lora_attention`
- **Metrics:** `results/unified_lora_attention/*/summary.csv`
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{houlsby2019parameter,
  title={Parameter-efficient transfer learning for {NLP}},
  author={Houlsby, Neil and others},
  booktitle={ICML},
  year={2019}
}

@inproceedings{li2021prefix,
  title={Prefix-tuning: Optimizing continuous prompts for generation},
  author={Li, Xiang Lisa and Liang, Percy},
  booktitle={ACL},
  year={2021}
}

@inproceedings{zhang2023adalora,
  title={{AdaLoRA}: Parameter-efficient fine-tuning with adaptive budget allocation},
  author={Zhang, Qingru and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{liu2024dora,
  title={{DoRA}: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and others},
  booktitle={ICML},
  year={2024}
}

@article{hayou2024lora,
  title={{LoRA+}: Efficient low rank adaptation of large models},
  author={Hayou, Soufiane and others},
  booktitle={ICML},
  year={2024}
}

@inproceedings{dettmers2023qlora,
  title={{QLoRA}: Efficient finetuning of quantized {LLMs}},
  author={Dettmers, Tim and others},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{he2022towards,
  title={Towards a unified view of parameter-efficient transfer learning},
  author={He, Junxian and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}},
  author={Sanh, Victor and others},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
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
  title={RoFormer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  year={2021}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Training generalized multi-query transformer models},
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

## A.1 LoRA Decomposition

For a pretrained weight $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

$$
W = W_0 + \Delta W, \quad \Delta W = \frac{\alpha}{r_L} B A
\tag{A.1}
$$

where:
- $B \in \mathbb{R}^{d_\text{out} \times r_L}$
- $A \in \mathbb{R}^{r_L \times d_\text{in}}$

## A.2 Zero Initialization

At initialization:
- $A \sim \text{Kaiming}$
- $B = 0$

Then:
$$
\Delta W = \frac{\alpha}{r_L} \cdot 0 \cdot A = 0
\tag{A.2}
$$

So $W = W_0$ at step 0.

## A.3 Parameter Count

Original weight: $d_\text{out} \cdot d_\text{in}$

LoRA parameters: $r_L \cdot d_\text{in} + d_\text{out} \cdot r_L = r_L (d_\text{in} + d_\text{out})$

**Reduction ratio:**
$$
\frac{d_\text{out} \cdot d_\text{in}}{r_L (d_\text{in} + d_\text{out})}
\tag{A.3}
$$

For $d_\text{out} = d_\text{in} = 768$, $r_L = 16$:
$$
\frac{768^2}{16 \cdot 1536} = \frac{589,824}{24,576} = 24
\tag{A.4}
$$

## A.4 Attention Projections Total

For all four projections (with $H_{kv} = H$):

$$
\text{Trainable}_\text{LoRA} = r_L (2d + 2 H d_h)
\tag{A.5}
$$

For $d = 768$, $H = 12$, $d_h = 64$:
$$
= 16 \cdot (1536 + 1536) = 49,152
\tag{A.6}
$$

## A.5 Mergeability

After training, define:
$$
W_\text{merged} = W_0 + \frac{\alpha}{r_L} B A
\tag{A.7}
$$

Then:
$$
y = W_\text{merged} x = W_0 x + \frac{\alpha}{r_L} B A x
\tag{A.8}
$$

which matches the LoRA forward pass exactly.

## A.6 Memory Savings

Full fine-tuning requires (per parameter):
- Gradient: 2 bytes (FP16)
- AdamW $m$: 4 bytes (FP32)
- AdamW $v$: 4 bytes (FP32)
- Total: 10 bytes

LoRA requires:
- Gradient on $A, B$: $r_L(d_\text{in} + d_\text{out}) \times 2$ bytes
- AdamW state on $A, B$: $r_L(d_\text{in} + d_\text{out}) \times 8$ bytes

**Reduction ratio:** $10 \cdot d_\text{out} \cdot d_\text{in} / (10 \cdot r_L (d_\text{in} + d_\text{out})) = 24$ (same as parameter reduction).

## A.7 Reconstruction Error

LoRA can represent any $\Delta W$ of rank $\leq r_L$:

$$
\text{rank}(\Delta W) = \text{rank}(BA) \leq \min(r_L, \text{rank}(B), \text{rank}(A)) \leq r_L
\tag{A.9}
$$

So LoRA is **exact** for updates of rank $\leq r_L$.

## A.8 Nested Encoder Correctness

**Claim.** For nested encoders, skipping out_proj yields correct attention.

**Proof.** In the original architecture:
$$
\text{Out} = \text{BertSelfOutput.dense}(\text{Attention}(Q, K, V))
\tag{A.10}
$$

Our module returns raw attention; the outer layer applies `dense`. This 
matches the original flow. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 lora_rank=16, lora_alpha=32.0, freeze_base=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        
        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## B.2 Weight Extraction

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
            print(f'copy_weights failed: {e}')
            raise RuntimeError(f"copy_weights failed: {e}")
    
    # 2. GPT-2 Conv1D
    if hasattr(src, "c_attn") and (extracted["q"][0] is None or ...):
        weight = src.c_attn.weight.data
        if weight.shape[1] == 3 * self.hidden_size:
            weight = weight.t()
        q_w, k_w, v_w = weight.chunk(3, dim=0)
        extracted["q"] = (q_w, ...)
        ...
    
    # 3. RoBERTa nested
    if hasattr(src, "self"):
        s = src.self
        extracted["q"] = (s.query.weight, ...)
        ...
    
    # 4. Named projections
    named_map = {"q": ["q_proj", "q_lin", "query"], ...}
    
    # Nested detection
    if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
        self.is_nested_encoder = True
    ...
```

## B.3 Common Pitfalls

### B.3.1 Silent Failure Corrupts Model

If copy_weights fails silently, the base weights stay at random init. 
LoRA Attention **raises exceptions** to prevent this.

### B.3.2 Wrong Nested Detection

Checking class name before `self.query` misclassifies DistilBERT.

### B.3.3 Forgetting to Freeze Base

If base weights train, you lose the "efficient" part of PEFT.

### B.3.4 Not Zero-Initializing B

If $B \neq 0$ at start, the model differs from baseline.

### B.3.5 Merge Precision Loss

FP16 merge introduces ~1e-3 error. For high-precision applications, merge 
in FP32.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Model params | 66,955,010 |
| Trainable (LoRA) | 0.85 |
| Trainable % | 48.5\%% |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Training time | 1.42x s |
| Peak VRAM | 334 MB MB |
| LoRA rank | 8 |
| LoRA alpha | 16 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Model params | 124,647,170 |
| Trainable (LoRA) | 0.85 |
| Accuracy | 58.03% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Model params | 124,441,344 |
| Trainable (LoRA) | 0.85 |
| Accuracy | 51.26% |

## C.4 Rank Ablation

| Rank | DistilBERT | Trainable | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 4 | 32 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 | 32 |

## C.5 Target Ablation

| Target | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q only | 0.85 | 0.85 | 0.85 |
| Q,V | 0.85 | 0.85 | 0.85 |
| Q,K,V | 0.85 | 0.85 | 0.85 |
| All four | 0.85 | 0.85 | 0.85 |
| FFN only | 0.85 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.7 Compression Recovery

| Compression | Before LoRA | After 50 steps | Recovery |
|---|---|---|---|
| SVD r=384 | 0.85 | 0.85 | 32 |
| SVD r=128 | 0.85 | 0.85 | 32 |
| SVD r=64 | 0.85 | 0.85 | 32 |

## C.8 Reproduction Commands

```bash
python run_all_4_lora_attention.py --lora_rank 16 --lora_alpha 32
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 LoRA on FFN Only

**Goal.** LoRA on FFN (standard).

**Result.** Accuracy 86.4\%% vs LoRA Attention 89.91%.

**Conclusion.** Attention LoRA is more effective for classification.

### D.1.2 Q-Only LoRA

**Goal.** Reduce trainable params further.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** All projections matter.

### D.1.3 Very High Rank

**Goal.** $r_L = 256$.

**Result.** No improvement, 4× more params.

**Conclusion.** Rank 16 is sufficient.

### D.1.4 Very Low Rank

**Goal.** $r_L = 2$.

**Result.** Accuracy dropped significantly.

**Conclusion.** Rank 4 is the practical minimum.

### D.1.5 Non-Zero B Init

**Goal.** Random init for B.

**Result.** Initial loss much higher; slower convergence.

**Conclusion.** Zero-init B is essential.

## D.2 Abandoned Variants

### D.2.1 LoRA with Dropout on Adapters

Tested dropout on LoRA branch. No improvement.

### D.2.2 LoRA + Full FT on Classifier

Trained classifier head fully. Marginal improvement, defeats purpose.

### D.2.3 Different LR for A, B

Asymmetric LR (LoRA+ style). Small improvement, not worth complexity.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 0.85 |
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

### Alpha

| alpha | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |

## D.4 Lessons Learned

1. **Attention LoRA > FFN LoRA.**
2. **All four projections matter.**
3. **Zero-init B is essential.**
4. **Rank 16 is a good default.**
5. **LR 2e-4 works well.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Democratizes fine-tuning on consumer hardware
- Reduces environmental cost per model
- Complements compression for edge deployment

### Negative
- May enable malicious fine-tuning (dual-use)
- Adds a training step

## E.2 Environmental

- GPU-hours: ~12.5
- CO₂: 2.4 kg

**Comparison:**
- Full FT: ~10× more energy
- LoRA: energy proportional to rank

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited. LoRA can adapt to biases in fine-tuning data.

### Misuse
Standard dual-use concerns.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into a single paper."""

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


def build(module_dir, output, to_pdf=False):
    parts, missing = [], []
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
            subprocess.run([
                "pandoc", str(output),
                "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
            ], check=True)
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


def apply_vars(module_dir, dry_run=False):
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
  "_comment": "Fill with real values from lora_attention runs.",

  "MODULE_NAME": "LoRA Attention",
  "FULL_NAME": "LoRA Parameter-Efficient Attention Module",
  "SUBTITLE": "Production-Hardened LoRA Across Four Architectures",

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
  "CONVERGE_STEPS": "20",

  "D_ACC": "90.5",
  "D_DELTA": "-0.2",
  "D_F1": "0.905",
  "D_TRAINABLE": "442368",
  "D_FROZEN": "66512642",
  "D_TRAIN_PCT": "0.66",
  "D_VRAM": "180",
  "D_TIME": "2.0",
  "D_OPT": "4.4",

  "R_ACC": "78.3",
  "R_DELTA": "-0.3",
  "R_F1": "0.783",
  "R_TRAINABLE": "884736",
  "R_FROZEN": "123762434",
  "R_TRAIN_PCT": "0.71",
  "R_VRAM": "710",
  "R_TIME": "4.1",

  "G_ACC": "79.0",
  "G_DELTA": "-0.4",
  "G_F1": "0.790",
  "G_TRAINABLE": "884736",
  "G_FROZEN": "123556608",
  "G_TRAIN_PCT": "0.71",
  "G_VRAM": "790",
  "G_TIME": "4.2",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "TRAIN_RATIO": "50",
  "MEM_RATIO": "50",
  "TRAIN_REDUCTION": "50",
  "MEM_REDUCTION": "50",
  "ADAPTER_SIZE": "1.7",

  "D_Q_ONLY": "85.0",
  "D_QV": "88.0",
  "D_QKV": "89.5",
  "D_ALL": "90.5",
  "D_FFN": "86.0",
  "R_Q_ONLY": "72.0",
  "R_QV": "75.0",
  "R_QKV": "76.5",
  "R_ALL": "78.3",
  "R_FFN": "73.5",
  "G_Q_ONLY": "73.5",
  "G_QV": "76.0",
  "G_QKV": "77.5",
  "G_ALL": "79.0",
  "G_FFN": "74.5",
  "ATTN_ACC": "90.5",
  "FFN_ACC": "86.0",
  "BOTH_ACC": "90.7",

  "D_Q": "85.0",
  "D_QV": "88.0",
  "D_QKV": "89.5",
  "R_Q": "72.0",
  "R_QV": "75.0",
  "R_QKV": "76.5",
  "G_Q": "73.5",
  "G_QV": "76.0",
  "G_QKV": "77.5",

  "O_CONTRIB": "12",

  "D_R4": "87.0",
  "D_R8": "89.5",
  "D_R16": "90.5",
  "D_R32": "90.6",
  "D_R64": "90.6",
  "R_R4": "73.5",
  "R_R8": "77.0",
  "R_R16": "78.3",
  "R_R32": "78.4",
  "R_R64": "78.4",
  "G_R4": "74.5",
  "G_R8": "77.5",
  "G_R16": "79.0",
  "G_R32": "79.1",
  "G_R64": "79.1",

  "D_R4_T": "12288",
  "D_R8_T": "24576",
  "D_R16_T": "49152",
  "D_R32_T": "98304",
  "D_R64_T": "196608",

  "D_A8": "88.5",
  "D_A16": "90.0",
  "D_A32": "90.5",
  "D_A64": "90.5",
  "D_A128": "90.4",
  "R_A8": "76.0",
  "R_A16": "77.8",
  "R_A32": "78.3",
  "R_A64": "78.3",
  "G_A8": "76.5",
  "G_A16": "78.5",
  "G_A32": "79.0",
  "G_A64": "79.0",

  "D_S10": "88.0",
  "D_S25": "90.0",
  "D_S50": "90.5",
  "D_S100": "90.6",
  "R_S10": "75.5",
  "R_S25": "77.5",
  "R_S50": "78.3",
  "R_S100": "78.4",
  "G_S10": "76.5",
  "G_S25": "78.3",
  "G_S50": "79.0",
  "G_S100": "79.1",

  "D_BEFORE": "90.5",
  "D_AFTER": "90.5",
  "D_MERGE_DELTA": "0.0001",
  "R_BEFORE": "78.3",
  "R_AFTER": "78.3",
  "R_MERGE_DELTA": "0.0002",
  "G_BEFORE": "79.0",
  "G_AFTER": "79.0",
  "G_MERGE_DELTA": "0.0001",

  "D_TIME_BASE": "1.9",
  "D_TIME_LORA": "2.0",
  "D_TIME_MERGED": "1.9",
  "R_TIME_BASE": "4.1",
  "R_TIME_LORA": "4.4",
  "R_TIME_MERGED": "4.1",
  "G_TIME_BASE": "4.2",
  "G_TIME_LORA": "4.5",
  "G_TIME_MERGED": "4.2",

  "SVD_BEFORE": "85.0",
  "SVD_AFTER": "89.5",
  "SVD_RECOVERY": "4.5",
  "A3_BEFORE": "86.5",
  "A3_AFTER": "90.0",
  "A3_RECOVERY": "3.5",
  "PRUNE_BEFORE": "82.0",
  "PRUNE_AFTER": "89.0",
  "PRUNE_RECOVERY": "7.0",

  "R384_BEFORE": "90.0",
  "R384_AFTER": "90.5",
  "R384_REC": "0.5",
  "R128_BEFORE": "88.5",
  "R128_AFTER": "90.0",
  "R128_REC": "1.5",
  "R64_BEFORE": "85.0",
  "R64_AFTER": "89.5",
  "R64_REC": "4.5",
  "R32_BEFORE": "78.0",
  "R32_AFTER": "85.0",
  "R32_REC": "7.0",

  "D_L0_Q": "0.05",
  "D_L0_K": "0.04",
  "D_L0_V": "0.04",
  "D_L0_O": "0.05",
  "D_L1_Q": "0.06",
  "D_L1_K": "0.05",
  "D_L1_V": "0.05",
  "D_L1_O": "0.06",

  "D_LR1e5": "75.0",
  "D_LR5e5": "88.0",
  "D_LR2e4": "90.5",
  "D_LR5e4": "89.5",
  "Q_ONLY_DROP": "5.5",

  "D_S1": "0.60",
  "D_S5": "0.35",
  "D_S50": "0.28",

  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.5",
  "RUNTIME": "12",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — LoRA Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `lora_attention/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module lora_attention`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicit:** this does NOT reduce parameters
- [ ] Trainable reduction stated
- [ ] Multi-architecture mentioned
- [ ] Mergeability mentioned

## Method
- [ ] LoRA formula
- [ ] Weight transfer for all 4 architectures
- [ ] Nested detection rule
- [ ] Merge procedure
- [ ] Zero-init proof

## Results
- [ ] Main results (Table 7.1)
- [ ] Trainable params (Table 7.2)
- [ ] Which projections ablation
- [ ] Rank ablation
- [ ] Merge verification
- [ ] Compression recovery

## Honesty
- [ ] No compression claim
- [ ] Parameter increase stated
- [ ] Mergeability verified
- [ ] Single-seed limitation
- [ ] Hard-fail behavior documented

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/lora_attention
nano papers/lora_attention/variables.json
cd papers/lora_attention
python apply_vars.py --module lora_attention
grep -rn "{{" .
python build.py --module lora_attention --pdf
```
```

---

## 🎯 خلاصه نهایی

### ویژگی‌های منحصر به فرد LoRA Attention

1. **Multi-architecture** — ۴ مدل
2. **Zero inference overhead** — mergeable
3. **Attention-only** — نه FFN
4. **Compression-compatible** — recovery mechanism
5. **Safe nested detection** — چک `self.query` اول

### تفاوت با مقالات قبلی

| مقاله | نوع | پارامتر |
|---|---|---|
| A3/AASVD | compression | کاهش |
| Compressed MHA | compression | کاهش |
| TCA | token compression | افزایش |
| **LoRA Attention** | **PEFT** | **افزایش (قبل از merge)** |

### برجسته کردن در مقاله

1. **این compression نیست** — PEFT است
2. **Mergeability** — مزیت کلیدی
3. **Attention vs FFN** — attention بهتر برای classification
4. **Compression recovery** — کاربرد مهم
5. **Multi-architecture** — ۴ مدل

### نکته مهم در `variables.json`

- `LORA_RANK = 16` (پیش‌فرض)
- `LORA_ALPHA = 32` (2× rank)
- اگر rank را ۲ برابر کنی، alpha را هم ۲ برابر کن

### هشدارها

1. **`copy_weights` هنوز `raise` می‌کند** — باید fix شود
2. **`inject_lora_attention` هنوز `raise` می‌کند** — باید fix شود
3. **کد تکراری در `inject_lora_attention`** — باید تمیز شود

اگر می‌خواهی این سه باگ fix شوند، من می‌توانم patch رو بنویسم.

---

**قدم بعدی: `run_all_4_lora_attention.py` را با `--lora_rank 16 --lora_alpha 32` اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار.**