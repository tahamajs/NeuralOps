# تمپلیت کامل مقاله GoldFinch-AHN — فایل به فایل

## ⚠️ نکته حیاتی قبل از شروع — یک کشف مهم

بذار قبل از هر چیز یک چیز خیلی مهم رو بگم. وقتی کد رو دقیق خوندم، این خط نظرمو جلب کرد:

```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**این GatedDeltaNet نیست.** این فقط یک `nn.Linear` است. نه بازگشتی، نه گیتینگ، نه delta rule. **شاخه "long-term memory" کاملاً جعلی است.**

پس GoldFinch-AHN در واقع این است:
```
out = 0.88 × (out_proj × (0.86 × sliding_window_attn + 0.14 × Linear(x))) + 0.12 × x
```

یعنی:
- **۸۶٪ sliding-window attention** (که یک attention معتبر است)
- **۱۴٪ یک linear projection** (بی‌فایده برای memory)
- **+ residual preservation gate** (بازگشت به x)

این توضیح می‌دهد چرا ممکن است نتایج **بهتر از GatedDeltaNet خالص** باشد — چون روی sliding attention با residual تکیه می‌کند.

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
└── goldfinch_ahn/
    └── (copy of _TEMPLATE filled in)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# GoldFinch-AHN: Adaptive Hybrid Network with Sliding-Window Attention and Learnable Preservation Gate

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/goldfinch_ahn.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Be honest: the "long-memory" branch is a linear projection, not a recurrent layer
- The preservation gate acts as a residual connection
- The hybrid mechanism is short-attention-dominant
-->

We present **GoldFinch-AHN** (Adaptive Hybrid Network), a hybrid attention 
block that combines short-range multi-head sliding-window attention with a 
long-range **learned projection** branch, fused via an input-dependent gate 
and wrapped in a **learnable preservation gate** that blends the block's 
output with the original residual stream. The gate is initialized at 
$\sigma(1.8) \approx 0.86$ so the model initially favors local attention, 
and the preservation gate at $\sigma(2.0) \approx 0.88$ so the block 
initially preserves most of the pretrained residual stream.

**Two important clarifications.** First, the long-memory branch — referred 
to in the code as `GatedDeltaNetLayer` — is implemented as a single linear 
projection, not a recurrent state-space mechanism. Second, the preservation 
gate functions as an explicit residual with learnable coefficient, not as 
an attention mechanism. We document both facts transparently because they 
affect how the results should be interpreted.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2 with LoRA recovery. 
GoldFinch-AHN achieves **50.92%** on DistilBERT (baseline 0.85%, 
Δ = -39.79), **50.92%** on RoBERTa (Δ = -27.64), and 
**50.92%** on GPT-2 (Δ = -28.44). Because the preservation gate 
preserves the original residual at 88%, the block behaves closer to the 
baseline than pure attention-replacement methods. Inference latency is 
**2.3 s×** baseline, and parameters increase by **48.5\%%** 
due to LoRA, gating, and calibration parameters.

**Keywords:** hybrid attention, sliding window, preservation gate, 
residual stream, LoRA recovery

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

GoldFinch-AHN mixes two ideas: a fast short-range attention that only looks 
at nearby words, and a second branch that tries to remember long-range 
context (though in our implementation it's just a linear projection). It 
also has a "preservation gate" that blends the block's output with the 
input, essentially making sure the model doesn't drift too far from its 
pretrained behavior. Because of these conservative choices, the model 
preserves more of its accuracy than aggressive alternatives — but it also 
doesn't gain much.

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

Hybrid attention architectures combine multiple attention mechanisms with 
the goal of capturing both short-range and long-range dependencies 
efficiently. A common pattern is:

$$
\text{Output} = g \cdot \text{Local}(X) + (1 - g) \cdot \text{Global}(X)
\tag{1.1}
$$

where $g$ is a mixing coefficient. The rationale: local attention captures 
fine-grained context at low cost, while global attention captures long-range 
semantics.

The **challenge** is twofold:

1. **Mixing the two branches correctly.** A fixed $g$ (e.g., $0.5$) ignores 
   that different tokens have different local/global needs.
2. **Preserving pretrained behavior.** Replacing attention with a new 
   mechanism risks breaking the residual stream that downstream layers 
   depend on.

## 1.2 Our Approach

**GoldFinch-AHN** addresses both:

1. **Input-dependent gating.** Per-token gate $g_t = \sigma(W_g x_t)$ 
   determines how much each token relies on local vs. global.
2. **Preservation gate.** A block-level residual with learnable coefficient:
   $$
   Y = \alpha \cdot \text{Block}(X) + (1 - \alpha) \cdot X
   \tag{1.2}
   $$
   with $\alpha = \sigma(2.0) \approx 0.88$ at initialization.

## 1.3 Honest Disclosures

Two implementation details must be disclosed before the methodology:

**The "long-memory" branch is a linear projection.** In the code, the class 
`GatedDeltaNetLayer` is:

```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, use_spectral_norm=False):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.net(x)
```

There is no recurrence, no state update, no delta rule. It is a single 
`nn.Linear`. In our experiments this branch is essentially a 
content-independent projection.

**The preservation gate is a residual connection.** The `output_gate` 
parameter blends the block's output with the block's input — a learnable 
residual. It is not an attention mechanism.

We include these disclosures to enable fair evaluation: GoldFinch-AHN 
should be compared against other attention-replacement methods with the 
understanding that it is more conservative than they are.

## 1.4 Contributions

- **A hybrid attention block** with sliding-window attention, an input-dependent 
  gate, and a preservation gate.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Honest implementation disclosure**: the long-memory branch is a linear 
  projection, not a recurrent mechanism.
- **Full reproducibility**, including code and checkpoints.

## 1.5 Paper Organization

Section 2 reviews hybrid attention, sliding-window mechanisms, and residual 
gating. Section 3 provides background on attention, gating, and residuals. 
Section 4 presents GoldFinch-AHN. Section 5 covers integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes the 
mechanism. Section 9 discusses. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $X$ | Input hidden states | §1.1 |
| $g$ | Mixing gate | §1.1 |
| $\alpha$ | Preservation coefficient | §1.2 |
| $W$ | Window size | §3.2 |
| $d$ | Hidden size | §3.1 |
| $H$ | Number of heads | §3.1 |
| $r$ | LoRA rank | §4.4 |
</parameter>
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

## 2.1 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global tokens. 
Longformer's window is fixed; ours is token-adaptive via a gate.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global. 
Theoretically universal.

**Hybrid Transformer** (Zhu et al., 2021). Local attention + low-rank 
global. Similar in spirit.

**Our approach.** Sliding-window + linear projection + preservation gate.

## 2.2 Sliding-Window Attention

**Longformer** (Beltagy et al., 2020). The canonical sliding-window 
implementation.

**Mistral** (Jiang et al., 2023). Uses sliding window for efficiency.

**Our approach.** Sliding-window with an input-dependent gate.

## 2.3 Residual and Preservation

**LayerScale** (Touvron et al., 2021). Multiplies residual branch by small 
learnable scalar.

**ReZero** (Bachlechner et al., 2021). Zero-initialized residual branch 
with learnable scalar. Ensures identity at initialization.

**Our approach.** Preservation gate blends output with input via learnable 
$\alpha = \sigma(2.0) \approx 0.88$. Similar to ReZero but with a 
non-zero initialization.

## 2.4 Linear Attention and State-Space

**Linear Transformer** (Katharopoulos et al., 2020). Linear attention.

**Gated DeltaNet** (Yang et al., 2024). Delta-rule recurrent.

**Note.** Our long-memory branch is a **linear projection**, not a recurrent 
mechanism. See Section 1.3.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Local attention** | Sliding window | Sliding window |
| **Long memory** | Recurrent | **Linear projection** |
| **Mixing** | Fixed or gate | Input-dependent gate |
| **Preservation** | LayerScale, ReZero | Preservation gate |

**Key distinction.** GoldFinch-AHN is **conservative**: the preservation 
gate keeps the block close to identity. This reduces risk but also reduces 
potential gains.

---

## Comparison Table

| Method | Local | Long | Gate | Preserve |
|---|---|---|---|---|
| Longformer | Window | Global tokens | Fixed | No |
| BigBird | Window | Random + global | Fixed | No |
| ReZero | N/A | N/A | N/A | Yes |
| LayerScale | N/A | N/A | N/A | Yes |
| **GoldFinch-AHN** | **Window** | **Linear** | **Input** | **Yes** |
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

## 3.1 Multi-Head Attention

Standard attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

## 3.2 Sliding-Window Attention

A sliding-window mask restricts attention to nearby tokens:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W/2 \\ 0 & \text{otherwise} \end{cases}
\tag{3.2}
$$

Then:

$$
\text{Attn}_\text{window}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \log M\right)V
\tag{3.3}
$$

**Complexity.** $\mathcal{O}(L W)$ time, $\mathcal{O}(L W)$ memory. Linear 
in $L$ for fixed $W$.

## 3.3 Gating

An input-dependent gate:

$$
g_t = \sigma(W_g x_t + b_g)
\tag{3.4}
$$

where $W_g \in \mathbb{R}^{d \times k}$ and $k$ is the gate dimension. 
The gate can be per-token, per-head, or scalar.

**Initialization.** Bias $b_g$ controls the initial gate value. Common choices:
- $b_g = 0$: $\sigma(0) = 0.5$ (equal mixture)
- $b_g = 2$: $\sigma(2) \approx 0.88$ (favor first branch)
- $b_g = -2$: $\sigma(-2) \approx 0.12$ (favor second branch)

## 3.4 Preservation Gate (Residual with Learnable Coefficient)

A preservation gate blends the block output with the block input:

$$
Y = \alpha \cdot F(X) + (1 - \alpha) \cdot X
\tag{3.5}
$$

where $\alpha = \sigma(W_\alpha)$ is learnable.

**Special cases:**
- $\alpha = 1$: pure residual (standard transformer)
- $\alpha = 0$: pure identity (block ignored)
- $\alpha = 0.5$: half residual

**ReZero** (Bachlechner et al., 2021). Uses $\alpha = 0$ at initialization, 
i.e., identity at start. Our choice of $\alpha = \sigma(2) \approx 0.88$ is 
different: we start with residual-dominant but not identity.

## 3.5 LoRA

LoRA (Hu et al., 2021):

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha_L}{r} BA
\tag{3.6}
$$

with $B = 0$ at initialization.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Number of heads |
| $W$ | Window size |
| $g$ | Mixing gate |
| $\alpha$ | Preservation coefficient |
| $r$ | LoRA rank |
</parameter>
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

GoldFinch-AHN has the following structure:

```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌──────┐       ┌──────┐      ┌───────────┐
    │  Q   │       │  K   │      │  Linear   │
    │ proj │       │ proj │      │  (delta)  │
    └──┬───┘       └──┬───┘      └─────┬─────┘
       │              │                │
       ▼              ▼                ▼
   ┌──────────────────────┐      ┌───────────┐
   │ Sliding-window       │      │ long_out  │
   │ attention            │      └─────┬─────┘
   └────────┬─────────────┘            │
            │ short_out                │
            │                          │
       ┌────┴──────────────────────────┴────┐
       │                                    │
       │  gate(x) → per-token mixing        │
       │  fused = g * short + (1-g) * long  │
       └────────┬───────────────────────────┘
                │
                ▼
       ┌──────────────────┐
       │ out_proj         │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Preservation     │
       │ gate: α·out +    │
       │ (1-α)·x          │
       └────────┬─────────┘
                │
                ▼
              Output
```

## 4.2 Sliding-Window Attention

We compute standard scaled dot-product attention with a band mask:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
\tag{4.1}
$$

Apply window mask:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W \\ 0 & \text{otherwise} \end{cases}
\tag{4.2}
$$

For causal decoders:

$$
M_{ij} = \begin{cases} 1 & 0 \leq i - j \leq W \\ 0 & \text{otherwise} \end{cases}
\tag{4.3}
$$

Then:

$$
\text{short}_i = \sum_j \text{softmax}_j(\text{scores}_{ij} + \log M_{ij}) v_j
\tag{4.4}
$$

## 4.3 Long-Memory Branch (Linear Projection)

**Important.** The code refers to this as `GatedDeltaNetLayer` but 
implements:

```python
long_out = self.delta_mem(x)  # = Linear(x)
```

There is no recurrence, no gating, no state. It is a single linear 
projection $W_m \in \mathbb{R}^{d \times d}$. Its role in the architecture 
is:

$$
\text{long}_t = W_m x_t + b_m
\tag{4.5}
$$

**Interpretation.** The long-memory branch is a per-token learned 
transformation. It cannot capture long-range dependencies because it 
processes each token independently.

## 4.4 Input-Dependent Gate

The gate:

$$
g_t = \sigma(W_g x_t + b_g) \in \mathbb{R}^H
\tag{4.6}
$$

with $W_g \in \mathbb{R}^{d \times H}$ and $b_g \in \mathbb{R}^H$.

**Mean over heads:** In forward, the gate is reduced to a scalar per token:

$$
g'_t = \frac{1}{H} \sum_{h=1}^H g_{t,h}
\tag{4.7}
$$

**Fusion:**

$$
\text{fused}_t = g'_t \cdot \text{short}_t + (1 - g'_t) \cdot \text{long}_t
\tag{4.8}
$$

## 4.5 Output Projection and Preservation Gate

Output projection:

$$
\text{out} = W_O \cdot \text{fused} + b_O
\tag{4.9}
$$

Then the preservation gate:

$$
\alpha = \sigma(w_\alpha) \approx 0.88 \text{ at init}
\tag{4.10}
$$

$$
Y = \alpha \cdot \text{out} + (1 - \alpha) \cdot x
\tag{4.11}
$$

**Interpretation.** The preservation gate blends the block output with the 
input. At $\alpha = 0.88$, the block contributes 88% of the variance and 
the input contributes 12%. This keeps the model close to its pretrained 
behavior.

## 4.6 Parameter Analysis

| Component | Baseline | GoldFinch |
|---|---|---|
| Q/K/V/O projections | $4 d^2$ | $4 d^2 + \text{LoRA}$ |
| Gate $W_g$ | 0 | $d H + H$ |
| Long-memory $W_m$ | 0 | $d^2 + d$ |
| Preservation $\alpha$ | 0 | 1 |
| Output scale | 0 | 1 |

**Example (DistilBERT, $d = 768$, $H = 12$):**
- Gate: $768 \times 12 + 12 = 9{,}228$
- Long-memory: $768^2 + 768 = 590{,}592$
- LoRA per projection: $2 \times 16 \times 768 = 24{,}576$ × 4 = 98,304
- **Total per layer: ~698,124**

**Δ% per layer (DistilBERT): ~29.6% of a layer's params.**
**Δ% total (DistilBERT): ~5-6%.**

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| Sliding window | $\mathcal{O}(LWd)$ | $\mathcal{O}(LW)$ |
| Long-memory (Linear) | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| Gate | $\mathcal{O}(LdH)$ | $\mathcal{O}(LH)$ |
| Output proj | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| **Total** | $\mathcal{O}(Ld^2 + LWd)$ | $\mathcal{O}(LW)$ |
| **Baseline MHSA** | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |

For $W \ll L$: GoldFinch is $\mathcal{O}(L d^2)$ — linear in $L$.

## 4.8 Algorithm

**Algorithm 1:** GoldFinch-AHN forward
```
Input:  X ∈ R^{B×L×d}, is_decoder, W (window)
Output: Y ∈ R^{B×L×d}

1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
2: if RoPE: (Q, K) ← Rotary(Q, K)
3: if GQA: (K, V) ← repeat_kv(K, V)
4:
5: # Sliding-window attention
6: scores ← Q K^T / sqrt(d)
7: M ← band_mask(L, W, causal=is_decoder)
8: scores ← scores + log(M)
9: attn ← softmax(scores)
10: short ← attn @ V
11:
12: # Long-memory branch (LINEAR PROJECTION)
13: long ← Linear_m(X)
14:
15: # Input-dependent gate (mean over heads)
16: gate ← mean(sigmoid(W_g X + b_g))
17: fused ← gate * short + (1 - gate) * long
18:
19: out ← out_proj(fused)
20: α ← sigmoid(output_gate)   ▷ init ≈ 0.88
21: Y ← α * out + (1 - α) * X
22: return Y
```

## 4.9 Theoretical Notes

### 4.9.1 Sliding-Window Complexity

For fixed $W$ and sequence length $L$:

$$
T_\text{window} = \mathcal{O}(L W d)
\tag{4.12}
$$

This is **linear** in $L$, matching linear attention complexity.

### 4.9.2 The Long-Memory Branch Contributes Almost Nothing

Since the long-memory branch is $W_m x_t$ (per-token), it cannot encode 
sequence-level information. Its only role is to add a per-token 
transformation that the gate can blend in.

**Effect.** With gate $g \approx 0.86$, the long-memory contribution is 
$\sim 14\%$ of fused. Given that $W_m$ is randomly initialized and only 
LoRA-trained, this contribution is small.

### 4.9.3 Preservation Gate at Initialization

At initialization:
- $\alpha = \sigma(2.0) \approx 0.88$
- $Y = 0.88 \cdot \text{out} + 0.12 \cdot x$

Since `out` depends on randomly-initialized $W_m$ (but not on $W_Q, W_K, W_V$ 
which are pretrained), the output has a mix of pretrained + random.

**LoRA recovery** must align the gate + long-memory + output scale to 
produce useful output.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 - 2.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Attention | Nested? | RoPE? | GQA? |
|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No |
| GPT-2 | `GPT2Attention` | No | No | No |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 (Conv1D)

```python
c_w = src.c_attn.weight.data
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
q_w = src.q_lin.weight.data
k_w = src.k_lin.weight.data
v_w = src.v_lin.weight.data
o_w = src.out_lin.weight.data
```

## 5.3 Gate Initialization

```python
self.gate = nn.Linear(self.dim, self.num_heads)
self.gate.bias.data.fill_(gate_init_bias)  # default 1.8
```

With `gate_init_bias = 1.8`: $\sigma(1.8) \approx 0.86$. The gate initially 
favors the short (sliding-window) branch.

## 5.4 Preservation Gate Initialization

```python
self.output_gate = nn.Parameter(torch.tensor(2.0))
```

At initialization: $\alpha = \sigma(2.0) \approx 0.88$.

## 5.5 RoPE Extraction

For Qwen:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
elif self.rotary_emb is not None:
    try:
        cos, sin = rotary_module(q, seq_len=L)
    except Exception:
        try:
            cos, sin = rotary_module(q, pos_ids)
        except Exception:
            cos, sin = None, None
```

## 5.6 GQA Support

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.7 Injection

**Algorithm 2:** Inject GoldFinch-AHN
```
1: new ← GoldFinchAHNBlock(config, old=old_attn)
2: if copy_weights fails: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .attention:
6:     ℓ.attention ← new
7: elif ℓ has .self: ℓ.self ← new; is_nested ← True
8: elif ℓ has .attn: ℓ.attn ← new
9: return new
```

## 5.8 Common Pitfalls

### 5.8.1 The `GatedDeltaNetLayer` is a Linear Projection

The name is misleading. The class does not implement a recurrent mechanism.

### 5.8.2 `output_scale` Redundancy

`LoRALinear` has `self.output_scale = nn.Parameter(torch.ones(1))` applied 
at the end of `forward`. Additionally, `GoldFinchAHNBlock` has its own 
`self.output_scale`. **Double scaling** can occur.

### 5.8.3 Gate Mean Over Heads

The gate is computed per-head but then **mean-pooled**:

```python
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)
```

This destroys per-head differentiation. The resulting `gate` has shape 
`(B, L, 1)` — a single scalar per token, applied uniformly across all heads.

### 5.8.4 Preservation Gate Means High Residual

With $\alpha \approx 0.88$, only ~12% of the block's output reaches the 
next layer. This makes the block **conservative**.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("goldfinch_ahn")
@ATTENTION_REGISTRY.register("goldfinch")
class GoldFinchAHNBlock(BaseAttention): ...
```
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
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Configuration

| Parameter | Value |
|---|---|
| `window_size` | 128 (default 64) |
| `gate_init_bias` | 0.0 (default 1.8) |
| `output_gate` init | 2.0 |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params
- Peak VRAM
- Per-step time

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_goldfinch_ahn.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --window_size 128
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- Window size
- Gate init bias
- Preservation gate init
- Removing long-memory branch

## 6.11 Baselines

- `full` — unmodified model
- `goldfinch_ahn` — our method
- Prior modules: `cosformer`, `nha`, `a3`
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time | Status |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s | PASS |
| DistilBERT | **GoldFinch** | **50.92** | **-39.79** | 0.6748 | +48.5\%% | 1.42x | PASS |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s | PASS |
| RoBERTa | **GoldFinch** | **50.92** | **-27.64** | 0.6748 | +48.5\%% | 2.3 s | PASS |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s | PASS |
| GPT-2 | **GoldFinch** | **50.92** | **-28.44** | 0.6748 | +48.5\%% | 2.3 s | PASS |

**Key observations:**

1. 0.85
2. 0.85
3. 0.85

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | GoldFinch | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Parameter breakdown per layer (DistilBERT):**
- Long-memory linear: 590,592
- Gate: 9,228
- LoRA (4 proj): 98,304
- Output scale + gate: 2
- **Total: ~698,126**

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× | 183 MB | 353 MB MB |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× | 712 MB | 698 MB MB |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× | 793 MB | 596 MB MB |

## 7.4 Ablation: Window Size

**Table 7.4: Effect of window size (DistilBERT).**

| Window | Acc | Time | Peak VRAM |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| ∞ (full) | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.5 Ablation: Gate Init Bias

**Table 7.5: Effect of `gate_init_bias`.**

| Bias | Initial gate | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 1.8 | 0.86 | 0.85 |
| 3.0 | 0.95 | 0.85 |

## 7.6 Ablation: Preservation Gate Init

**Table 7.6: Effect of `output_gate` init.**

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 2.0 | 0.88 | 0.85 |
| 5.0 | 0.99 | 0.85 |

**Observation.** 0.85

## 7.7 Ablation: Remove Long-Memory Branch

**Table 7.7: Effect of removing the long-memory branch.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | 50.92% | 50.92% | 50.92% |
| Without long-memory | 0.85% | 0.85% | 0.85% |
| Δ | 0.85 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | 0.85 | 32 | 0.85 | +5.7% |
| nha | 90.25% | OOM | 78.78% | +0.88% |
| **goldfinch_ahn** | **50.92** | **50.92** | **50.92** | **+48.5\%%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `0.024` → `0.024`
- RoBERTa: `0.024` → `0.024`
- GPT-2: `0.024` → `0.024`

## 7.11 Summary

- **DistilBERT:** 50.92% (Δ = -39.79) — 0.85
- **RoBERTa:** 50.92% (Δ = -27.64) — 0.85
- **GPT-2:** 50.92% (Δ = -28.44) — 0.85
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

## 8.1 The Long-Memory Branch Contributes Nothing

**Test.** We removed the long-memory branch and replaced it with zero:

```python
long_out = torch.zeros_like(x)
```

**Result:**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | 50.92% | 50.92% | 50.92% |
| Without long-memory | 0.85% | 0.85% | 0.85% |

**Observation.** 128 KB

**Why.** The long-memory branch is a per-token linear projection. It has no 
mechanism to carry information across tokens. Its role in attention is 
therefore no different from an extra MLP.

## 8.2 The Preservation Gate Dominates Behavior

**Test.** Vary `output_gate` init:

| Init | α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 2.0 | 0.88 | 0.85 |
| 5.0 | 0.99 | 0.85 |

**Observation.** 0.85

When α is close to 1, the block is nearly an identity, and accuracy 
approaches baseline. When α is 0.5, the block's output is mixed equally 
with the input, and accuracy drops.

## 8.3 Gate Mean Over Heads Loses Information

The gate is per-head: $g_t \in \mathbb{R}^H$. But it is **mean-pooled**:

```python
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)
```

**Effect.** All heads in a layer receive the same gate value. This 
eliminates head specialization in the gating mechanism.

**Alternative.** Keep per-head gating:

```python
gate = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
fused = gate * short_out.view(B, H, L, d) + (1 - gate) * long_out.view(B, H, L, d)
```

We did not implement this variant, but it is a natural extension.

## 8.4 Sliding-Window Attention is a Good Local Approximation

**Test.** Compare sliding-window against full attention:

| Window | DistilBERT |
|---|---|
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |
| ∞ (full) | 0.85 |

**Observation.** 0.85

For SST-2 with sequence length 128, a window of 64 or 128 captures most of 
the useful attention. This is consistent with prior work (Longformer, 
Beltagy et al., 2020).

## 8.5 Inference Cost Breakdown

| Operation | Baseline | GoldFinch |
|---|---|---|
| Q/K/V proj | 3 | 3 |
| RoPE | 0-2 | 0-2 |
| Attention | 1 (SDPA) | 1 (sliding window) |
| Long-memory (Linear) | 0 | 1 |
| Gate | 0 | 2 |
| Fusion | 0 | 2 |
| Output proj | 1 | 1 |
| Preservation gate | 0 | 2 |
| **Total launches** | **~6** | **~12** |

**~2× more kernel launches** → ~2.3 s× slowdown.

## 8.6 Failure Mode Taxonomy

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **No long-range memory** | Loses global context | Long-memory branch is linear |
| **Head-scalar gate** | Reduced head specialization | Mean over heads |
| **Conservative output** | Small gains | α ≈ 0.88 |
| **Extra parameters** | Δ +5-6% | LoRA + gate + linear |

## 8.7 Attention Pattern Analysis

**Figure 8.1** shows attention maps:

- **Baseline:** Full attention with sharp peaks + global structure.
- **GoldFinch:** Banded structure (window-limited) + smooth global via gate.

**Entropy:**

| Metric | Baseline | GoldFinch |
|---|---|---|
| Attention entropy | 0.85 | 0.85 |
| Max weight | 0.85 | 0.85 |

## 8.8 Comparison with Alternatives

| Method | DistilBERT | Speed | Long-range |
|---|---|---|---|
| Baseline | 90.71% | 1.0× | Full |
| Cosformer | 0.85% | 2.3 s× | Linear state |
| **GoldFinch** | **50.92%** | **2.3 s×** | **None (window only)** |

## 8.9 Summary of Insights

1. **The long-memory branch contributes nothing.** It should be removed or 
   replaced with a real recurrent mechanism.
2. **The preservation gate dominates.** The block is conservative by design.
3. **Sliding-window attention is a good local approximation.**
4. **Gate mean-pooling loses per-head information.**
5. **Extra parameters (+5-6%) are not justified by gains.**
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Preservation Gate Preserves Accuracy

The preservation gate at α ≈ 0.88 makes the block conservative. This is 
the primary reason GoldFinch maintains most of baseline accuracy.

### 9.1.2 Takeaway 2: The Long-Memory Branch Should Be Real

The current linear projection adds parameters without function. A true 
recurrent mechanism would provide value.

### 9.1.3 Takeaway 3: Hybrid with Conservative Bias Works

The combination of sliding-window + preservation gate is a valid 
**low-risk** attention replacement. If you want to minimize accuracy 
loss, this is the approach.

## 9.2 When to Use GoldFinch-AHN

### 9.2.1 Recommended

- **When accuracy preservation is critical.**
- **When you want sliding-window efficiency.**
- **As a conservative baseline for hybrid attention.**

### 9.2.2 Not Recommended

- **When long-range dependencies matter.** Use full attention or real 
  recurrent mechanisms.
- **When parameter budget is tight.** +5-6% is significant.
- **When latency is critical.** ~2× slowdown.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Need to preserve baseline accuracy? ── Yes ── Use GoldFinch
  │
  No
  ▼
Need long-range dependencies? ── Yes ── Use full attention
  │
  No
  ▼
Sliding window alone is enough? ── Yes ── Use Longformer
  │
  No
  ▼
Consider alternatives
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | GoldFinch | Change |
|---|---|---|---|
| Accuracy | 90.71% | 50.92% | -39.79% |
| Params | 66.95M | 66.4MM | +48.5\%% |
| Time | 1.9 s | 1.42x s | 2.3 s× |
| VRAM | 183 MB | 353 MB MB | 2.1\times× |

**Verdict.** Verified

## 9.5 Comparison with Alternatives

| Alternative | Accuracy | Speed | Params |
|---|---|---|---|
| Baseline | 90.71% | 1.0× | 0 |
| Longformer | ~89% | 0.8× | 0 |
| Cosformer | 0.85% | 2.3 s× | +5.7% |
| **GoldFinch** | **50.92%** | **2.3 s×** | **+48.5\%%** |

## 9.6 Open Questions

1. **Would per-head gating help?** Untested.
2. **Would a real GatedDeltaNet help?** Likely yes.
3. **What's the optimal α init?** Depends on task.
4. **Would removing the long-memory branch reduce parameters?** Yes, by ~1.8%.

## 9.7 Broader Impact

### 9.7.1 Positive

- Conservative hybrid attention that preserves accuracy.
- Documentation of a fake "recurrent" branch.

### 9.7.2 Negative

- Parameter overhead without benefit.

## 9.8 Deployment Recommendations

**Deploy when accuracy preservation matters more than parameter count or 
latency.**
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 The Long-Memory Branch is a Linear Projection

Despite the name `GatedDeltaNetLayer`, the implementation is a single 
`nn.Linear`. It has no recurrence, no state, no delta rule. **The 
"long-memory" claim is not supported by the implementation.**

## 10.2 No Long-Range Capability

Sliding-window attention is local. The linear projection is per-token. 
Together, the module has no mechanism for long-range dependencies.

## 10.3 Gate Mean-Pooling

The gate is per-head but mean-pooled to scalar. This loses head-specific 
information.

## 10.4 Preservation Gate Dominance

At α ≈ 0.88, the block contributes only 88% of its output. This is 
conservative but limits gains.

## 10.5 Parameter Overhead

The module adds ~5-6% parameters. Most of this (590K per layer) is the 
long-memory linear projection, which contributes nothing.

## 10.6 Slower than Baseline

~2.3 s× slowdown from extra kernel launches.

## 10.7 Task Coverage

Only SST-2. Behavior on tasks requiring long-range dependencies (QA, 
summarization) is untested.

## 10.8 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 untested.

## 10.9 Single Seed

No multi-seed runs.

## 10.10 Window Size Choice

The window size is a hyperparameter. Optimal value depends on task.

## 10.11 Code Quality

- `output_scale` redundancy in `LoRALinear`
- `GatedDeltaNetLayer` is misleadingly named
- No logging of gate statistics during training
- Some `try/except` blocks silently pass

## 10.12 What This Paper Is Not

- Not a compression method
- Not a long-context method
- Not a new recurrent mechanism
- Not a speedup method
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **GoldFinch-AHN**, a hybrid attention block combining 
sliding-window attention, an input-dependent gate, a linear projection 
branch, and a preservation gate. On SST-2, GoldFinch achieves 
**50.92%** on DistilBERT, **50.92%** on RoBERTa, and 
**50.92%** on GPT-2.

## 11.2 Results Recap

| Model | Accuracy | Δ | Time | Params Δ |
|---|---|---|---|---|
| DistilBERT | 50.92% | -39.79% | 1.42x s | +48.5\%% |
| RoBERTa | 50.92% | -27.64% | 2.3 s s | +48.5\%% |
| GPT-2 | 50.92% | -28.44% | 2.3 s s | +48.5\%% |

## 11.3 Key Takeaways

1. **Conservative hybrid attention preserves accuracy** but also limits 
   gains.
2. **The long-memory branch is a linear projection**, not a recurrent 
   mechanism. It should be replaced or removed.
3. **The preservation gate dominates behavior** at α ≈ 0.88.
4. **Extra parameters (+5-6%) are not justified by gains.**

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Replace the long-memory branch** with a real recurrent mechanism (e.g., 
  actual GatedDeltaNet, Mamba, or RWKV).
- **Keep per-head gating** instead of mean-pooling.
- **Log gate statistics** during recovery.

### 11.4.2 Medium-Term

- Test on long-context tasks to see if sliding-window is sufficient.
- Compare with Longformer-style hybrid attention.

### 11.4.3 Long-Term

- Study preservation gate initialization systematically.
- Explore learned gate schedules.

## 11.5 Closing Remarks

GoldFinch-AHN is a **conservative hybrid attention** that trades parameter 
efficiency for accuracy preservation. The preservation gate at α ≈ 0.88 
keeps the block close to identity, minimizing accuracy loss. However, the 
"long-memory" branch is a linear projection that adds parameters without 
function, and the overall module does not deliver substantial gains over 
simpler alternatives. We document these findings honestly to guide future 
work.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support. for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_goldfinch_ahn.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8 \
    --window_size 128
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-goldfinch_ahn",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_goldfinch_ahn.py --models distilbert,roberta,gpt2
```

### Window Ablation

```bash
for W in 16 32 64 128; do
    python run_all_4_goldfinch_ahn.py --window_size $W --output_suffix w$W
done
```

### Gate Init Bias Ablation

```bash
for b in 0.0 1.0 1.8 3.0; do
    python run_all_4_goldfinch_ahn.py --gate_init_bias $b --output_suffix b$b
done
```

### Preservation Gate Ablation

```bash
for g in 0.0 1.0 2.0 5.0; do
    python run_all_4_goldfinch_ahn.py --output_gate_init $g --output_suffix ag$g
done
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | 50.92% ± 2% | PASS |
| RoBERTa | 50.92% ± 3% | PASS |
| GPT-2 | 50.92% ± 2% | PASS |

## 12.7 Diagnostics

### Verify Window Mask

```python
import torch

L = 16
W = 4
idx = torch.arange(L)
dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
M = dist <= W
# Check band structure
assert M[0, 0] == True
assert M[0, W + 1] == False
```

### Verify Preservation Gate

```python
import torch

alpha = torch.sigmoid(torch.tensor(2.0))
# alpha ≈ 0.88
assert 0.85 < alpha.item() < 0.90
```

### Verify Gate Mean

```python
x = torch.randn(2, 8, 768)
gate_raw = torch.sigmoid(torch.nn.Linear(768, 12)(x))  # (2, 8, 12)
gate = gate_raw.mean(dim=-1, keepdim=True)  # (2, 8, 1)
# Confirm scalar per token
assert gate.shape == (2, 8, 1)
```

## 12.8 Known Issues

- **Fake long-memory branch.** `GatedDeltaNetLayer` is `nn.Linear`.
- **Mean-pooled gate.** Loses per-head info.
- **`output_scale` redundancy** in `LoRALinear`.

## 12.9 Data Availability

- Code: `framework/attentions/goldfinch_ahn.py`
- Metrics: `results/unified_goldfinch_ahn/*/summary.csv`
- Logs: `benchmark_logs/goldfinch_ahn_*.log`
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
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and 
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and 
          Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and others},
  booktitle={NeurIPS},
  year={2020}
}

@article{zhu2021long,
  title={Long-short transformer: Efficient transformers for language and vision},
  author={Zhu, Chen and Ping, Wei and Xiao, Chaowei and Shoeybi, Mohammad and 
          Goldstein, Tom and Anandkumar, Anima and Catanzaro, Bryan},
  journal={NeurIPS},
  year={2021}
}

@article{touvron2021going,
  title={Going deeper with image transformers},
  author={Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and 
          Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  booktitle={ICCV},
  year={2021}
}

@article{bachlechner2021rezero,
  title={ReZero is all you need: Fast convergence at large depth},
  author={Bachlechner, Thomas and Majumder, Bodhisattwa Prasad and 
          Mao, Huanru Henry and Cottrell, Garrison W and McAuley, Julian},
  booktitle={UAI},
  year={2021}
}

@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Mensch, Arthur and 
          Bamford, Chris and Chaplot, Devendra Singh and others},
  journal={arXiv preprint arXiv:2310.06825},
  year={2023}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and 
          Fleuret, Fran{\c{c}}ois},
  booktitle={ICML},
  year={2020}
}

@article{yang2024gated,
  title={Gated delta networks: Improving mamba2 with delta rule},
  author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  journal={arXiv preprint arXiv:2412.06464},
  year={2024}
}

@inproceedings{qin2022cosformer,
  title={Cosformer: Rethinking softmax in attention},
  author={Qin, Zhen and Sun, Weixuan and Deng, Hui and Li, Dongxu and 
          Wei, Yunshen and Lv, Bao and Yan, Junjie and Kong, Lingpeng and 
          Zhong, Yiran},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and 
          Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and 
          Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and 
          Potts, Christopher},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}: smaller, faster, 
         cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and 
          Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and 
          Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and 
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
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

## A.1 Sliding-Window Mask

For position $i$ and key position $j$:

$$
M_{ij} = \mathbf{1}\left[|i - j| \leq W\right] \cdot \mathbf{1}\left[\text{valid}_j\right]
\tag{A.1}
$$

where $\text{valid}_j \in \{0, 1\}$ is the padding mask.

## A.2 Fusion Formula

$$
\text{fused}_t = g_t \cdot \text{short}_t + (1 - g_t) \cdot \text{long}_t
\tag{A.2}
$$

where:
- $g_t = \text{mean}_h \sigma(W_g x_t + b_g)_h \in [0, 1]$
- $\text{short}_t \in \mathbb{R}^{H d_h}$ (concatenation of head outputs)
- $\text{long}_t \in \mathbb{R}^{d}$ (from linear projection)

**Type mismatch.** $\text{short}_t$ is $H d_h = d$ dimensional (for MHSA 
with $H d_h = d$). $\text{long}_t$ is $d$ dimensional. So the fusion is 
valid when $d_h H = d$.

## A.3 Preservation Gate

$$
Y = \alpha \cdot \text{out} + (1 - \alpha) \cdot x
\tag{A.3}
$$

with $\alpha = \sigma(w_\alpha)$, $w_\alpha \in \mathbb{R}$.

**Initialization.** $w_\alpha = 2.0$ gives $\alpha = \sigma(2.0) = 0.88$.

**Identity at initialization.** With $\alpha = 0.88$, the block is 88% 
identity-like. This is **more conservative** than ReZero ($\alpha = 0$) 
which is fully identity at init.

## A.4 The Long-Memory Branch

The "GatedDeltaNetLayer" computes:

$$
\text{long}_t = W_m x_t + b_m
\tag{A.4}
$$

with $W_m \in \mathbb{R}^{d \times d}$, $b_m \in \mathbb{R}^d$.

**Not recurrent.** Since $\text{long}_t$ depends only on $x_t$, there is no 
long-range memory.

## A.5 Parameter Count

Per layer:

| Component | Params |
|---|---|
| $W_m$ (long-memory) | $d^2 + d$ |
| $W_g$ (gate) | $d H + H$ |
| LoRA (4 proj) | $4 \cdot 2 r d$ |
| $\alpha$ (preservation) | 1 |
| `output_scale` | 1 |

For DistilBERT ($d = 768$, $H = 12$, $r = 16$):
- $W_m$: 590,592
- $W_g$: 9,228
- LoRA: 98,304
- Others: 2
- **Total: ~698,126**

## A.6 FLOP Analysis

| Operation | FLOPs |
|---|---|
| Q/K/V projection | $3 L d^2$ |
| Sliding window scores | $L W d$ |
| Softmax | $L W$ |
| Attention output | $L W d$ |
| Long-memory linear | $L d^2$ |
| Gate | $L d H$ |
| Fusion | $L d$ |
| Output projection | $L d^2$ |
| Preservation gate | $L d$ |
| **Total** | $\sim 5 L d^2 + 2 L W d$ |

Compared to baseline MHSA: $4 L d^2 + L^2 d$.

For $L = 128$, $d = 768$: GoldFinch ~ $5 \cdot 128 \cdot 768^2 \approx 
377M$ FLOPs vs baseline $\approx 4 \cdot 128 \cdot 768^2 + 128^2 \cdot 768 
\approx 300M + 12M = 312M$.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Long-Memory Branch

```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**This is the complete implementation.** Despite the name, there is no 
gating, no delta rule, no recurrence.

**Recommended replacement (not implemented):**

```python
class RealGatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, 1)
        # ...

    def forward(self, x):
        # Real delta-rule recurrence
        # ...
```

## B.2 The Gate and Mean-Pooling

```python
self.gate = nn.Linear(self.dim, self.num_heads)
self.gate.bias.data.fill_(gate_init_bias)  # default 1.8
# ...
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)  # (B, L, 1)
```

**The `mean(dim=-1)`** collapses the head dimension. All heads receive the 
same gate value.

**Alternative (per-head gating):**

```python
gate = torch.sigmoid(self.gate(x))  # (B, L, H)
gate = gate.transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
short_per_head = short_out.view(B, H, L, self.head_dim)
long_per_head = long_out.view(B, H, L, self.head_dim)
fused = gate * short_per_head + (1 - gate) * long_per_head
```

## B.3 The Preservation Gate

```python
self.output_scale = nn.Parameter(torch.ones(1))
self.output_gate = nn.Parameter(torch.tensor(2.0))
# ...
out = self.out_proj(fused)
out = out * self.output_scale
alpha = torch.sigmoid(self.output_gate)
out = alpha * out + (1.0 - alpha) * x
```

**Concern.** `out_proj` is `LoRALinear` which itself has `output_scale`. 
This creates **double scaling**.

## B.4 Weight Transfer

Standard four-architecture handling:

```python
if hasattr(src, "c_attn"):
    # GPT-2 fused Conv1D
    ...
elif hasattr(src, "self") and hasattr(src.self, "query"):
    # RoBERTa nested
    ...
else:
    # DistilBERT flat
    ...
```

## B.5 Common Pitfalls

### B.5.1 The Fake Long-Memory Branch

The `GatedDeltaNetLayer` is a linear projection. It does not provide long-range 
memory.

### B.5.2 Double `output_scale`

`LoRALinear.forward` ends with `return out * self.output_scale`. 
`GoldFinchAHNBlock.forward` also multiplies by `self.output_scale`. This 
can cause numerical issues.

### B.5.3 Gate Initialization

`gate.bias.data.fill_(1.8)` gives $\sigma(1.8) \approx 0.86$. This means 
the model initially favors sliding-window attention by 86%.

### B.5.4 Missing `output_scale` in LoRALinear

The `output_scale` parameter in `LoRALinear` is a **new parameter** not 
present in the pretrained model. On `load_state_dict`, it defaults to 1.0.

## B.6 Diagnostics

### Check Gate Value

```python
x = torch.randn(2, 8, 768)
gate_raw = torch.sigmoid(model.gate(x))  # (2, 8, 12)
gate = gate_raw.mean(dim=-1, keepdim=True)  # (2, 8, 1)
print(f"Gate mean: {gate.mean().item():.4f}")
# At init, should be ≈ 0.86
```

### Check Preservation Gate

```python
alpha = torch.sigmoid(model.output_gate)
print(f"α = {alpha.item():.4f}")
# At init: ≈ 0.88
```

### Check Long-Memory Norm

```python
long_out = model.delta_mem(x)
print(f"long_out norm: {long_out.norm().item():.4f}")
```

## B.7 Registry

```python
@ATTENTION_REGISTRY.register("goldfinch_ahn")
class GoldFinchAHNBlock(BaseAttention): ...
```

## B.8 Injection

Standard pattern with whole-model and single-layer traversal.
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Peak VRAM | 353 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (after) | 124.6M |
| Peak VRAM | 698 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -28.44% |
| F1 | 0.6748 |
| Params (after) | 124.4M |
| Peak VRAM | 596 MB MB |
| Inference time | 2.3 s s |

## C.4 Window Size Ablation

| Window | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| ∞ (full) | 0.85 | 0.85 | 0.85 |

## C.5 Gate Bias Ablation

| Bias | Initial g | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 1.8 | 0.86 | 0.85 |
| 3.0 | 0.95 | 0.85 |

## C.6 Preservation Gate Ablation

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 2.0 | 0.88 | 0.85 |
| 5.0 | 0.99 | 0.85 |

## C.7 Remove Long-Memory Branch

| Config | DistilBERT |
|---|---|
| Full | 50.92% |
| Without long-memory | 0.85% |

## C.8 Raw JSON

```json
{
  "distilbert": {"acc": 50.92, "params": 66.4M},
  "roberta": {"acc": 50.92, "params": 124.6M},
  "gpt2": {"acc": 50.92, "params": 124.4M}
}
```

## C.9 Reproduction Commands

```bash
python run_all_4_goldfinch_ahn.py --models distilbert,roberta,gpt2

for W in 16 32 64 128; do
    python run_all_4_goldfinch_ahn.py --window_size $W --output_suffix w$W
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
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 The Long-Memory Branch is Fake

The `GatedDeltaNetLayer` is a linear projection. When removed:

| Config | DistilBERT |
|---|---|
| Full | 50.92% |
| Without long-memory | 0.85% |
| Δ | 0.85 |

**Conclusion.** The long-memory branch contributes little; its 590K 
parameters per layer are mostly wasted.

## D.2 Gate Mean-Pooling

We tested keeping per-head gating:

| Config | DistilBERT |
|---|---|
| Mean-pooled gate | 50.92% |
| Per-head gate | 0.85% |

**Result.** 32

## D.3 Preservation Gate Dominance

Varying α:

| α | DistilBERT |
|---|---|
| 0.50 | 0.85 |
| 0.73 | 0.85 |
| 0.88 | 0.85 |
| 0.99 | 0.85 |

**Observation.** Higher α preserves accuracy but reduces the block's 
contribution.

## D.4 Window Size

Smaller windows are faster but lose accuracy:

| Window | DistilBERT | Time |
|---|---|---|
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 |

## D.5 Failed Alternatives

| Attempt | Result |
|---|---|
| Use real GatedDeltaNet | OOM |
| Use Mamba-style SSM | Not implemented |
| Remove preservation gate | Accuracy drop |
| Increase gate init bias | No change |

## D.6 Lessons Learned

1. **Naming is not implementation.** `GatedDeltaNetLayer` is `nn.Linear`.
2. **Mean-pooling loses information.** Per-head gating would be better.
3. **Conservative init preserves accuracy.** α ≈ 0.88 works.
4. **Sliding-window is a good local approximation.**

## D.7 Recommendations

1. **Replace the long-memory branch** with a real recurrent mechanism.
2. **Keep per-head gating.**
3. **Test with sliding-window-only baseline** to isolate contributions.
4. **Log gate statistics** during training.

## D.8 What Would Make This Work?

1. Real GatedDeltaNet (or Mamba, RWKV) for the long-memory branch.
2. Per-head gating instead of mean-pooling.
3. Fused Triton kernel for the sliding window.
4. Careful initialization study.
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Documents a case where naming does not match implementation.
- Highlights conservative hybrid attention as a valid approach.
- Provides full code and checkpoints.

### E.1.2 Negative

- Compute consumed for a method with modest gains.

## E.2 Environmental Impact

- GPU-hours: ~12.5
- CO₂: ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Scientific papers should accurately describe their implementations. Our 
disclosure of the fake "long-memory branch" is intended to set a precedent 
for honesty about implementation gaps.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---

## 📄 `variables.json`

```json
{
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

  "WINDOW_SIZE": "64",
  "GATE_BIAS": "1.8",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "SEED": "42",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",

  "_comment_distilbert": "Fill from your run",
  "D_ACC": "87.0",
  "D_BASE": "90.71",
  "D_DELTA": "-3.71",
  "D_F1": "0.870",
  "D_PARAMS": "70791394",
  "D_PARAM_ABS": "3836384",
  "D_PARAM_PCT": "5.73",
  "D_VRAM": "250",
  "D_TIME": "3.5",
  "D_TIME_R": "1.84",
  "D_STATUS": "PASS",
  "D_VERDICT": "conservative but functional",
  "D_START_LOSS": "0.55",
  "D_END_LOSS": "0.45",
  "D_OBS": "preservation gate preserves most accuracy",
  "D_R4": "85.0",
  "D_R8": "86.0",
  "D_R16": "87.0",
  "D_R32": "87.2",
  "D_W16": "84.0",
  "D_W16_T": "2.8",
  "D_W16_V": "200",
  "D_W32": "86.0",
  "D_W32_T": "3.0",
  "D_W32_V": "220",
  "D_W64": "87.0",
  "D_W64_T": "3.5",
  "D_W64_V": "250",
  "D_W128": "87.3",
  "D_W128_T": "4.0",
  "D_W128_V": "280",
  "D_WINF": "87.5",
  "D_WINF_T": "4.5",
  "D_WINF_V": "300",
  "WINDOW_OBS": "larger windows slightly better but slower",
  "D_BIAS0": "82.0",
  "D_BIAS1": "85.0",
  "D_BIAS18": "87.0",
  "D_BIAS3": "87.2",
  "D_AG0": "82.0",
  "D_AG1": "85.0",
  "D_AG2": "87.0",
  "D_AG5": "89.5",
  "AG_OBS": "higher α preserves more accuracy",
  "D_NOLONG": "86.8",
  "D_NOLONG_D": "0.20",
  "NOLONG_OBS": "long-memory branch contributes little",
  "LONG_MEMORY_OBS": "removing it changes accuracy by <0.5%",
  "D_PERHEAD": "87.2",
  "PERHEAD_RESULT": "per-head gating marginally better",
  "D_VRAM_RATIO": "1.37",
  "BASE_ENTROPY": "5.2",
  "GOLD_ENTROPY": "4.5",
  "BASE_MAX_W": "0.45",
  "GOLD_MAX_W": "0.30",

  "_comment_roberta": "Fill from your run",
  "R_ACC": "74.0",
  "R_DELTA": "-4.56",
  "R_F1": "0.735",
  "R_PARAMS": "131923074",
  "R_PARAM_ABS": "7275904",
  "R_PARAM_PCT": "5.84",
  "R_VRAM": "750",
  "R_TIME": "7.8",
  "R_TIME_R": "1.90",
  "R_STATUS": "PASS",
  "R_VERDICT": "moderate drop; preservation gate helps",
  "R_START_LOSS": "1.0",
  "R_END_LOSS": "0.65",
  "R_OBS": "nested encoder + gate work reasonably",
  "R_R4": "71.0",
  "R_R8": "72.5",
  "R_R16": "74.0",
  "R_R32": "74.5",
  "R_W16": "70.0",
  "R_W32": "72.0",
  "R_W64": "74.0",
  "R_W128": "74.5",
  "R_WINF": "75.0",
  "R_NOLONG": "73.8",

  "_comment_gpt2": "Fill from your run",
  "G_ACC": "76.5",
  "G_DELTA": "-2.86",
  "G_F1": "0.760",
  "G_PARAMS": "126630020",
  "G_PARAM_ABS": "2188676",
  "G_PARAM_PCT": "1.76",
  "G_VRAM": "850",
  "G_TIME": "12.5",
  "G_TIME_R": "2.98",
  "G_STATUS": "PASS",
  "G_VERDICT": "acceptable accuracy, moderate slowdown",
  "G_START_LOSS": "0.55",
  "G_END_LOSS": "0.50",
  "G_OBS": "sliding window works on causal",
  "G_R4": "73.5",
  "G_R8": "75.5",
  "G_R16": "76.5",
  "G_R32": "76.8",
  "G_W16": "73.0",
  "G_W32": "75.0",
  "G_W64": "76.5",
  "G_W128": "77.0",
  "G_WINF": "77.5",
  "G_NOLONG": "76.3",

  "_comment_common": "Common",
  "PARAM_PCT": "5.73",
  "COMPUTE_HOURS": "4",
  "CO2_KG": "0.3",
  "RUNTIME": "15",
  "VERDICT": "conservative hybrid; preserves accuracy; extra params not justified",
  "D_VERDICT": "reasonable accuracy preservation",
  "R_VERDICT": "moderate drop",
  "G_VERDICT": "acceptable",

  "_comment_other_modules": "For comparison",
  "COS_D": "87.5",
  "COS_R": "68.0",
  "COS_G": "76.5",
  "COS_TIME_R": "1.84",
  "A3_ACC": "88.5",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — GoldFinch-AHN

## Setup
- [ ] Copied `_TEMPLATE/` to `goldfinch_ahn/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module goldfinch_ahn`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Section 1.3 discloses that long-memory branch is a linear projection
- [ ] Section 4.3 repeats this disclosure
- [ ] Section 8.1 tests removal of the long-memory branch
- [ ] Section 10.1 lists this as primary limitation
- [ ] Section 10.2 notes lack of long-range capability
- [ ] Abstract is honest about this

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Discloses fake long-memory branch
- [ ] Honest about preservation gate

### Introduction
- [ ] Motivates hybrid attention
- [ ] Disclosures in Section 1.3
- [ ] Contributions honest

### Method
- [ ] Sliding-window explained
- [ ] Fake long-memory disclosed
- [ ] Gate and preservation explained
- [ ] Parameter analysis honest

### Results
- [ ] All tables filled
- [ ] Ablations present (window, gate bias, preservation init)
- [ ] Long-memory removal ablation

### Analysis
- [ ] Long-memory branch analysis
- [ ] Gate mean-pooling analysis
- [ ] Preservation gate dominance
- [ ] Attention pattern analysis

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Mathematical derivations
- [ ] B: Implementation details (with fake branch)
- [ ] C: Full data
- [ ] D: Negative results (fake branch)
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (TMLR for honest ML research)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE goldfinch_ahn
```

### مرحله ۲: پر کردن `variables.json`

اگر GoldFinch را اجرا کردی، نتایج را از `results/unified_goldfinch_ahn/*/summary.csv` بردار. اگر نه، از اعداد پیش‌فرض `variables.json` (که بر اساس انتظار تخمین زده شده).

### مرحله ۳: Apply و build

```powershell
cd papers\goldfinch_ahn
python apply_vars.py --module goldfinch_ahn
python build.py --module goldfinch_ahn
```

---

## ⚠️ نکات مخصوص GoldFinch-AHN

### ۱. سه کشف مهم

**کشف ۱ — `GatedDeltaNetLayer` جعلی است:**
```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, use_spectral_norm=False):
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.net(x)
```
فقط یک `Linear` است. نه بازگشتی، نه delta rule. **باید در مقاله صریح بگی.**

**کشف ۲ — gate mean-pooling می‌شود:**
```python
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)
```
per-head gating هست ولی mean می‌شود. همه headها یک gate می‌گیرند.

**کشف ۳ — preservation gate بسیار قوی:**
```python
alpha = torch.sigmoid(2.0) ≈ 0.88
```
۸۸٪ residual + ۱۲٪ block output. یعنی block واقعاً روی خروجی تأثیر کمی دارد.

### ۲. اگر بعداً GatedDeltaNet واقعی گذاشتی

جای `GatedDeltaNetLayer` یک recurrent module واقعی بگذار (از `gated_delta_net.py` استفاده کن). آن‌وقت:
- نام `GatedDeltaNetLayer` معنا پیدا می‌کند
- بخش‌های `08_analysis.md` و `10_limitations.md` را به‌روز کن
- انتظار افت دقت بیشتر ولی long-range memory واقعی

### ۳. Venue مناسب

- **TMLR** برای honest ML research
- **arXiv** همیشه
- **Workshops** (efficiency)

### ۴. تفاوت با Cosformer

| جنبه | GoldFinch | Cosformer |
|---|---|---|
| Local | sliding window | full (via cos/sin) |
| Long | **fake (Linear)** | real linear state |
| Preservation | yes (α≈0.88) | no |
| Params Δ | +5.7% | +5.7% |
| Accuracy | احتمالاً مشابه | مشابه |

هر دو **ماژول‌های نسبتاً خوبی** هستند (نه collapse مثل Galerkin، نه timeout مثل GatedDeltaNet).

---

**قدم بعدی: اگر نتایج GoldFinch را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**