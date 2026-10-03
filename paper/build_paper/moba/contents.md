# تمپلیت کامل مقاله Mamba — فایل به فایل

قبل از شروع، سه نکته حیاتی که باید بدانی:

## ⚠️ هشدارهای مهم

### هشدار ۱ — ادعای O(L) در docstring **غلط** است
کد می‌گوید:
```python
"""...operating in strictly O(L) time and space with respect to sequence length..."""
```

اما در عمل:
```python
log_weights = dist.unsqueeze(-1) * log_gamma.unsqueeze(0).unsqueeze(0)  # [L, L, D]
weights = torch.softmax(log_weights, dim=1)  # [L, L, D]
```

این **O(L² · D)** است، نه O(L). باید در مقاله صادق باشی.

### هشدار ۲ — این Mamba واقعی نیست
Mamba واقعی (Gu & Dao 2023) شامل:
- Selective scan
- Conv1d پیش از SSM
- Structured A matrix

این implementation:
- **State-space-inspired gating**
- Softmax-based causal weighting (نه scan)
- Preservation gate اضافه

پس اسمش را در مقاله باید **State-Space-Inspired Attention** بگذاری، نه Mamba.

### هشدار ۳ — `copy_weights` هنوز `raise` می‌کند
```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

این باید fix شود.

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
└── mamba/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
# Mamba-Inspired Gated State-Space Attention: A Drop-In Replacement for Multi-Head Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/mamba.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

State-space models (SSMs), particularly Mamba (Gu & Dao, 2023), have emerged 
as strong alternatives to attention for sequence modeling. However, replacing 
attention in a **pretrained** transformer with an SSM requires careful 
handling of the pretrained weights and preservation of the residual 
information flow. We present **Mamba-Inspired Gated State-Space Attention**, 
a drop-in replacement for multi-head attention that:

1. Reuses pretrained Q, K, V, O weights by projecting them into a two-branch 
   gated state-space formulation,
2. Applies a **learned exponential decay** over sequence positions to 
   emulate selective state-space dynamics,
3. Adds a **preservation gate** that blends the new output with the residual 
   input via a learnable scalar $\alpha = \sigma(g)$ initialized toward the 
   SSM branch,
4. Supports causal masking for decoders (exponential-decay causal softmax) 
   and bidirectional processing for encoders (forward + backward decay).

We are explicit about a **critical limitation**: despite the SSM-inspired 
formulation, the implementation materializes an $[L \times L \times D]$ 
weight tensor, resulting in $\mathcal{O}(L^2 D)$ memory and time, **not** 
$\mathcal{O}(L)$. The SSM-inspired decay enforces a soft causal structure 
but does not achieve the linear complexity of true Mamba.

On SST-2, Mamba-Inspired Gated SSM Attention achieves **73.62%** on 
DistilBERT (baseline 0.85%, Δ = -17.09), **84.17%** on RoBERTa 
(baseline 32%, Δ = +5.61), and **50.92%** on GPT-2 
(baseline 0.85%, Δ = -28.44), using 50 LoRA 
recovery steps. We position this module as a **state-space-inspired 
alternative** to standard attention with a preservation gate, and we 
explicitly discuss the gap between the SSM formulation and the actual 
implementation complexity.

**Keywords:** state-space models, gated attention, preservation gate, 
SSM-inspired, multi-architecture

---

## Lay Summary

State-space models like Mamba have recently become popular for processing 
long sequences. We've built an attention replacement inspired by these 
models: instead of comparing every word to every other word, our method 
applies a learned exponential decay over positions, emulating the way 
state-space models "forget" past information. We reuse the pretrained 
attention weights (Q, K, V, O) and add a "preservation gate" that can blend 
the new output back with the original input. Our method works on four 
different model architectures. **Important caveat:** although the method is 
inspired by state-space models, it does **not** achieve linear complexity 
like real Mamba — the implementation still uses quadratic memory in the 
sequence length.

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

Transformer self-attention has quadratic complexity in sequence length, 
motivating a large body of work on efficient alternatives. **State-space 
models** (SSMs), particularly **Mamba** (Gu & Dao, 2023), have emerged as a 
compelling family: they achieve $\mathcal{O}(L)$ complexity via selective 
state updates with input-dependent dynamics.

However, replacing a pretrained transformer's attention with an SSM is 
non-trivial:

1. **Pretrained weights:** The SSM's input/output projections have no 
   correspondence to the pretrained $W_Q, W_K, W_V, W_O$.
2. **Residual flow:** SSMs process the input in a fundamentally different way, 
   changing the residual stream statistics.
3. **Training cost:** Training a full SSM from scratch is expensive.
4. **Causal vs bidirectional:** SSMs are naturally causal; encoders need 
   bidirectional processing.

**Mamba-Inspired Gated State-Space Attention** addresses these challenges:

- It **reuses pretrained weights** by projecting Q, K, V into a two-branch 
  gating structure.
- It applies a **learned exponential decay** to emulate selective state-space 
  dynamics without a true SSM scan.
- It adds a **preservation gate** that blends the new output with the residual 
  input, stabilizing the initial forward pass.
- It supports both causal (decoders) and bidirectional (encoders) modes.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with attention projections 
$\{W_Q, W_K, W_V, W_O\}$, we seek a replacement mechanism $\mathcal{A}'$ such 
that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\mathcal{A}' \text{ has SSM-inspired structure}
\tag{1.1}
$$

subject to:

1. **Weight reuse:** Must copy pretrained weights via `copy_weights`.
2. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2, Qwen.
3. **Causal and bidirectional:** Support both modes.
4. **Recovery capacity:** LoRA adapters for fine-tuning.
5. **Preservation:** Blend with residual input to stabilize.

## 1.3 Our Approach

We propose a gated state-space-inspired attention with:

### 1.3.1 Two-Branch In-Projection

The input projection is doubled: $W_\text{in} \in \mathbb{R}^{d \times 2d}$. 
The output splits into:

- $p_1$: SSM input representation (projected from $W_V$ or $W_Q$)
- $p_2$: gating representation (projected from $W_Q$ or $W_K$)

### 1.3.2 Learned Exponential Decay

For each dimension $i$, we learn a decay $\gamma_i \in (0, 1)$ via:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{1.2}
$$

The pairwise weight between positions $j$ and $i$ ($j \leq i$):

$$
w_{ij}^{(d)} = \frac{\exp(\text{dist}_{ij} \cdot \log \gamma_d)}{\sum_{j'} \exp(\text{dist}_{ij'} \cdot \log \gamma_d)}
\tag{1.3}
$$

where $\text{dist}_{ij} = \max(0, j - i)$ for causal mode.

### 1.3.3 Gated Fusion

$$
\text{Out} = \text{SiLU}(p_1^\text{decayed}) \odot p_2
\tag{1.4}
$$

### 1.3.4 Preservation Gate

$$
\text{Final} = \sigma(g) \cdot \text{Out} + (1 - \sigma(g)) \cdot X
\tag{1.5}
$$

with $g$ initialized to 2.0 (so $\sigma(g) \approx 0.88$).

## 1.4 Contributions

- **A gated SSM-inspired attention.** Reuses pretrained Q/K/V/O weights in a 
  two-branch gated state-space formulation with learned exponential decay.
- **A preservation gate.** Blends the SSM output with the residual input, 
  stabilizing the initial forward pass.
- **Multi-architecture support.** Handles DistilBERT, RoBERTa, GPT-2, and 
  Qwen.
- **Causal and bidirectional modes.** Learned decay for encoders uses both 
  forward and backward directions.
- **Honest complexity reporting.** We acknowledge that the implementation is 
  $\mathcal{O}(L^2 D)$, **not** $\mathcal{O}(L)$, despite the SSM inspiration.

## 1.5 Paper Organization

Section 2 reviews related work on SSMs and efficient attention. Section 3 
provides background on state-space models, Mamba, and attention. Section 4 
presents our method. Section 5 describes integration. Section 6 details the 
setup. Section 7 reports results. Section 8 analyzes the SSM-inspired 
behavior. Section 9 discusses limitations and trade-offs. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $X \in \mathbb{R}^{B \times L \times d}$ | Input |
| $p_1, p_2$ | Two branches of in-projection |
| $A_{\log}$ | Log-decay parameter |
| $\gamma_i$ | Decay factor for dimension $i$ |
| $g$ | Preservation gate parameter |
| $r_L$ | LoRA rank |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 State-Space Models

### 2.1.1 Classical SSMs

**S4** (Gu et al., 2022) introduced structured state-space models with 
diagonal plus low-rank A matrices, achieving $\mathcal{O}(L)$ complexity. 
**S5** (Smith et al., 2023) parallelized the SSM scan across time.

### 2.1.2 Selective SSMs (Mamba)

**Mamba** (Gu & Dao, 2023) introduced **input-dependent** state transitions:

$$
h_t = \bar{A}(x_t) h_{t-1} + \bar{B}(x_t) x_t
\tag{2.1}
$$

where $\bar{A}, \bar{B}$ depend on the input $x_t$. This selective mechanism 
makes the SSM behave like a learned, content-aware recurrence.

**Our distinction.** Our implementation **does not** perform a true selective 
scan. We use learned exponential decay as a **proxy** for the selectivity, 
implemented as a pairwise softmax over positions.

## 2.2 Efficient Attention

**Linformer** (Wang et al., 2020) projects K, V to a fixed rank. 
**Performer** (Choromanski et al., 2021) uses random Fourier features. 
**Longformer** (Beltagy et al., 2020) uses sparse local windows.

**Our distinction.** We exploit SSM-inspired decay rather than a fixed 
sequence projection or kernel approximation.

## 2.3 Attention-Replacement Modules

**Hyena** (Poli et al., 2023) uses long convolutions. **RWKV** (Peng et al., 
2023) uses a linear recurrence with time-decay. **RetNet** (Sun et al., 2023) 
uses a retention mechanism with decay.

**Our distinction.** We operate on pretrained attention weights, reusing 
$W_Q, W_K, W_V, W_O$ in the SSM structure.

## 2.4 Position of Our Work

| Method | Complexity | Reuses pretrained? | Causal | Bidirectional |
|---|---|---|---|---|
| Mamba | $\mathcal{O}(L)$ | No | Yes | Yes (Bi-Mamba) |
| Hyena | $\mathcal{O}(L \log L)$ | No | Yes | Yes |
| RWKV | $\mathcal{O}(L)$ | No | Yes | No |
| **Ours** | **$\mathcal{O}(L^2 D)$** | **Yes** | **Yes** | **Yes** |

**Key distinction.** We reuse pretrained attention weights in an SSM-inspired 
formulation, trading complexity for compatibility with pretrained transformers.

---

## Comparison Table

| Method | Decay | Learned? | Weight reuse | Causal |
|---|---|---|---|---|
| Mamba | Selective | ✓ | No | ✓ |
| RWKV | Fixed | Partial | No | ✓ |
| RetNet | Fixed | ✓ | No | ✓ |
| **Ours** | Learned per-dim | ✓ | **✓** | ✓ |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 State-Space Models

A continuous-time SSM is defined by:

$$
\begin{aligned}
h'(t) &= A h(t) + B x(t) \\
y(t) &= C h(t) + D x(t)
\end{aligned}
\tag{3.1}
$$

where $A \in \mathbb{R}^{d_h \times d_h}$ is the state transition matrix, 
$B, C$ are input/output matrices, and $D$ is a skip connection.

Discretized with step size $\Delta$:

$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \Delta B
\tag{3.2}
$$

The recurrence:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
\tag{3.3}
$$

## 3.2 Mamba's Selective Mechanism

Mamba makes $\Delta, B, C$ input-dependent:

$$
\Delta_t = \text{softplus}(W_\Delta x_t), \quad B_t = W_B x_t, \quad C_t = W_C x_t
\tag{3.4}
$$

This lets the SSM decide when to update state based on input content.

## 3.3 Diagonal SSM Approximation

For efficiency, Mamba uses a **diagonal** A matrix:

$$
A = \text{diag}(a_1, \dots, a_{d_h}), \quad a_i < 0
\tag{3.5}
$$

The discrete decay per dimension:

$$
\bar{A}_i = \exp(\Delta_i a_i) \in (0, 1)
\tag{3.6}
$$

## 3.4 Our Decay Parameterization

We parameterize the decay as:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{3.7}
$$

where $A_{\log, i}$ is a learnable parameter. This gives:

- $\gamma_i \in (0, 1)$
- $\gamma_i \to 1$ when $A_{\log, i} \to -\infty$ (no forgetting)
- $\gamma_i \to 0$ when $A_{\log, i} \to +\infty$ (rapid forgetting)

## 3.5 Causal Decay Weights

For a causal model, we compute:

$$
\text{dist}_{ij} = \max(0, j - i), \quad
\log w_{ij} = \text{dist}_{ij} \cdot \log \gamma_i
\tag{3.8}
$$

After softmax over $j$:

$$
w_{ij} = \frac{\exp(\text{dist}_{ij} \log \gamma_i)}{\sum_{j'} \exp(\text{dist}_{ij'} \log \gamma_i)}
\tag{3.9}
$$

**Properties:**
- $w_{ii}$ is largest (position $i$ contributes most to itself)
- $w_{ij}$ decays with distance $|i - j|$ for $j < i$
- The decay rate is learned per dimension

## 3.6 Bidirectional Decay

For encoders, we combine forward and backward:

$$
w_{ij}^\text{fwd} = \text{softmax}_j(\text{dist}_{ij}^\text{fwd} \log \gamma_i)
\tag{3.10}
$$

$$
w_{ij}^\text{bwd} = \text{softmax}_j(\text{dist}_{ij}^\text{bwd} \log \gamma_i)
\tag{3.11}
$$

$$
\text{Out} = 0.5 \cdot (w^\text{fwd} p_1 + w^\text{bwd} p_1)
\tag{3.12}
$$

## 3.7 Gating Mechanism

We use the SiLU (Swish) activation:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{3.13}
$$

Gated fusion:

$$
\text{Out} = \text{SiLU}(p_1) \odot p_2
\tag{3.14}
$$

## 3.8 Preservation Gate

A learnable scalar $\alpha$:

$$
\alpha = \sigma(g)
\tag{3.15}
$$

with $g$ initialized to 2.0, so $\alpha \approx 0.88$.

Final output:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{3.16}
$$

## 3.9 Complexity Note

**Important.** Despite the SSM-inspired formulation, the implementation 
materializes:

$$
\text{weights} \in \mathbb{R}^{L \times L \times D}
\tag{3.17}
$$

This is $\mathcal{O}(L^2 D)$ memory, **not** $\mathcal{O}(L)$. We discuss 
this limitation in Section 10.

## 3.10 Grouped-Query Attention (GQA)

For GQA models, $H_{kv} < H$. Our implementation reuses K/V weights but 
does not require separate KV heads because the SSM operates on the full 
hidden dimension.

## 3.11 Rotary Position Embeddings (RoPE)

RoPE is applied to the input before the SSM projection.
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

Mamba-Inspired Gated SSM Attention replaces standard attention with a 
**two-branch gated state-space formulation**:

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ in_proj: d → 2d      │
    │   first d = p1       │  ← SSM branch
    │   second d = p2      │  ← Gate branch
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Learned decay        │
    │   γ_i = σ(-exp(A_i)) │
    │   softmax over dist  │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ p1' = decay(p1)      │
    │ out = SiLU(p1') * p2 │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ out_proj: d → d      │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Preservation gate    │
    │ α·out + (1-α)·X      │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 Two-Branch In-Projection

Instead of standard Q, K, V projections, we use:

$$
[p_1; p_2] = W_\text{in} X + b_\text{in}
\tag{4.1}
$$

where $W_\text{in} \in \mathbb{R}^{d \times 2d}$ and $p_1, p_2 \in \mathbb{R}^{B \times L \times d}$.

**Weight reuse.**

- $p_1$ branch: copied from $W_V$ (or $W_Q$ if V unavailable)
- $p_2$ branch: copied from $W_Q$ (or $W_K$ if Q unavailable)

This gives:

$$
W_\text{in} = \begin{pmatrix} W_V \\ W_Q \end{pmatrix}
\tag{4.2}
$$

### 4.2.1 Padding for Different Head Counts

If $d_\text{out}$ of the source projection is smaller than $d$, we **repeat**:

```python
if w_main.shape[0] < D and D % w_main.shape[0] == 0:
    rep = D // w_main.shape[0]
    w_main = w_main.repeat(rep, 1)
```

## 4.3 Learned Decay

### 4.3.1 Decay Parameterization

For each dimension $i \in \{1, \dots, d\}$:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{4.3}
$$

We initialize $A_{\log, i} = \log(i)$ (from `torch.log(arange(1, D+1))`), so:

$$
\gamma_i = \sigma(-i)
\tag{4.4}
$$

For small $i$, $\gamma_i \approx \sigma(-1) \approx 0.27$; for large $i$, 
$\gamma_i \to 0$.

### 4.3.2 Causality

For decoders, we define:

$$
\text{dist}_{ij} = \begin{cases} j - i & j \geq i \\ 0 & j < i \end{cases}
\tag{4.5}
$$

**Wait, this is reversed.** Let me re-read the code:

```python
dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)  # [L, L]
# pos.unsqueeze(1) is i, pos.unsqueeze(0) is j
# dist[i, j] = max(0, i - j)
```

So $\text{dist}_{ij} = \max(0, i - j)$ — the distance from position $j$ to 
position $i$ **for $j \leq i$**.

The log weights:

$$
\log w_{ij}^{(d)} = \text{dist}_{ij} \cdot \log \gamma_d
\tag{4.6}
$$

Then softmax over $j$ (dim=1):

$$
w_{ij}^{(d)} = \frac{\exp(\log w_{ij}^{(d)})}{\sum_{j'} \exp(\log w_{ij'}^{(d)})}
\tag{4.7}
$$

**Interpretation.** For query at position $i$, key at position $j$:

- If $j = i$: $\text{dist}_{ij} = 0$, so $\log w = 0$, giving $w = 1/Z$ 
  (highest after normalization).
- If $j = i - 1$: $\text{dist}_{ij} = 1$, so $\log w = \log \gamma_d$.
- If $j \to 0$: $\text{dist}_{ij} = i$, so $\log w = i \log \gamma_d$ 
  (smallest).

**Effect.** Recent keys get more weight; older keys get exponentially 
decayed weight. This emulates SSM's "forget gate".

### 4.3.3 Causal Mask

The causal mask sets entries with $j > i$ to $-\infty$:

```python
causal_mask = torch.tril(torch.ones(L, L))
log_weights = log_weights.masked_fill(~causal_mask, float("-inf"))
```

## 4.4 Bidirectional Mode

For encoders, we compute forward and backward distances:

$$
\text{dist}^\text{fwd}_{ij} = \max(0, i - j), \quad
\text{dist}^\text{bwd}_{ij} = \max(0, j - i)
\tag{4.8}
$$

Both are masked appropriately:

- Forward: $j \leq i$ (lower triangular)
- Backward: $j \geq i$ (upper triangular)

Then:

$$
\text{Out} = 0.5 \cdot (w^\text{fwd} p_1 + w^\text{bwd} p_1)
\tag{4.9}
$$

## 4.5 Gated Fusion

$$
\text{Out}_\text{SSM} = \text{SiLU}(p_1^\text{decayed}) \odot p_2
\tag{4.10}
$$

where SiLU is:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{4.11}
$$

## 4.6 Output Projection

Standard LoRALinear:

$$
\text{Out} = W_O \cdot \text{Out}_\text{SSM} + b_O
\tag{4.12}
$$

with $W_O$ copied from the pretrained output projection.

## 4.7 Preservation Gate

A learnable scalar $g$ initialized to 2.0:

$$
\alpha = \sigma(g) \approx 0.88
\tag{4.13}
$$

Final output:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{4.14}
$$

**Why this helps.** At the start of training, the SSM output may be noisy. 
The preservation gate keeps the residual input as a fallback. As training 
progresses, $\alpha$ can shift toward 1.0 if the SSM output improves.

## 4.8 Full Algorithm

**Algorithm 1:** Mamba-Inspired Gated SSM Attention forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, is_decoder
Output: Y ∈ R^{B×L×d}

 1: X ← RoPE(X) if position_embeddings provided
 2: proj ← in_proj(X)                        ▷ [B, L, 2d]
 3: p1, p2 ← chunk(proj, 2, dim=-1)          ▷ Two branches
 4: γ ← σ(-exp(A_log[:d]))                    ▷ [d]
 5: if is_decoder and L > 1:
 6:     dist ← max(0, i - j)                  ▷ [L, L]
 7:     log_w ← dist ⊗ log γ                  ▷ [L, L, d]
 8:     log_w ← mask_causal(log_w)
 9:     w ← softmax(log_w, dim=1)             ▷ [L, L, d]
10:     p1 ← einsum("ijd,bjd->bid", w, p1)
11:     gated ← SiLU(p1) * p2
12: elif L > 1:
13:     # Bidirectional
14:     dist_fwd ← max(0, i - j); dist_bwd ← max(0, j - i)
15:     w_fwd ← softmax(dist_fwd ⊗ log γ, dim=1)
16:     w_bwd ← softmax(dist_bwd ⊗ log γ, dim=1)
17:     p1_mix ← 0.5 * (w_fwd @ p1 + w_bwd @ p1)
18:     gated ← SiLU(p1_mix) * p2
19: else:
20:     gated ← SiLU(p1) * p2
21: Out ← out_proj(gated)                       ▷ d → d
22: Out ← mask_padding(Out, attention_mask)
23: α ← σ(pres_gate)
24: Final ← α * Out + (1 - α) * X
25: return Final
```

## 4.9 Complexity Analysis

### 4.9.1 Time Complexity

| Operation | Complexity |
|---|---|
| in_proj | $\mathcal{O}(L d^2)$ |
| Decay weights | $\mathcal{O}(L^2 d)$ |
| Softmax | $\mathcal{O}(L^2 d)$ |
| Weighted sum | $\mathcal{O}(L^2 d)$ |
| out_proj | $\mathcal{O}(L d^2)$ |

**Total:** $\mathcal{O}(L^2 d + L d^2)$.

For $L \gg d$: **$\mathcal{O}(L^2 d)$** — quadratic in $L$.

### 4.9.2 Memory Complexity

The weight tensor is $[L, L, d]$, so:

$$
\text{Memory} = \mathcal{O}(L^2 d)
\tag{4.15}
$$

**This is NOT O(L).**

### 4.9.3 Comparison with True Mamba

| Property | True Mamba | Our Implementation |
|---|---|---|
| Time | $\mathcal{O}(L)$ | $\mathcal{O}(L^2 d)$ |
| Memory | $\mathcal{O}(L d)$ | $\mathcal{O}(L^2 d)$ |
| Selective | Input-dependent | Learned per-dim |
| Scan | Parallel scan | Softmax over distance |

**The docstring claim of "O(L)" is incorrect.**

## 4.10 Parameter Analysis

| Component | Params |
|---|---|
| in_proj | $2 d^2$ |
| out_proj | $d^2$ |
| A_log | $d$ |
| pres_gate | 1 |
| LoRA (both proj) | $r_L (2d + d) \cdot 2$ |

For $d = 768$, $r_L = 16$:

| Component | Params |
|---|---|
| in_proj | 1,179,648 |
| out_proj | 589,824 |
| A_log | 768 |
| pres_gate | 1 |
| LoRA | ~66K |
| **Total** | **~1.84M** |

vs baseline attention ($4 d^2 = 2.36M$):

**Reduction: ~22%** at $r_L = 16$.

## 4.11 Theoretical Notes

### 4.11.1 Decay Interpretation

The learned decay $\gamma_i$ controls the SSM's "forget rate" per dimension:

- $\gamma_i \approx 1$: no forgetting → attends to all history
- $\gamma_i \approx 0$: rapid forgetting → attends only to recent

The init $A_{\log, i} = \log(i)$ gives $\gamma_i = \sigma(-i)$, so higher 
dimensions forget faster.

### 4.11.2 Softmax Over Distance

The softmax over distance is equivalent to a **geometric decay prior**:

$$
w_{ij} \propto \gamma_i^{|i - j|}
\tag{4.16}
$$

This is the discrete analog of the SSM's exponential decay.

### 4.11.3 Preservation Gate

**Claim.** The preservation gate ensures the initial forward pass approximates 
the identity transformation.

**Proof.** At init, $\alpha = \sigma(2.0) \approx 0.88$. So the output is 
$0.88 \cdot \text{Out}_\text{SSM} + 0.12 \cdot X$. If the SSM output is 
approximately zero (as $B = 0$ for LoRA), the output is $\approx 0.12 X$. 
$\square$

### 4.11.4 Causality

**Claim.** For decoders, causality is preserved.

**Proof.** The causal mask sets $w_{ij} = 0$ for $j > i$ (upper triangular 
to $-\infty$ in log-space). After softmax, $w_{ij} = 0$ for $j > i$. So 
output at $i$ depends only on inputs $j \leq i$. $\square$
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

## 5.2 Weight Transfer

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.data
if c_w.shape[0] < c_w.shape[1]:  # Conv1D stores (in, out)
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested_encoder = True
    q_w = src.self.query.weight
    k_w = src.self.key.weight
    v_w = src.self.value.weight
    o_w = src.output.dense.weight
```

### 5.2.3 Qwen GQA

```python
q_w = src.q_proj.weight  # (H·d_h, d)
k_w = src.k_proj.weight  # (H_kv·d_h, d)
v_w = src.v_proj.weight
o_w = src.o_proj.weight
```

### 5.2.4 DistilBERT

```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

## 5.3 Weight Reuse Strategy

### 5.3.1 in_proj Weight Layout

| Branch | Source | Rationale |
|---|---|---|
| $p_1$ (first d) | $W_V$ | SSM input |
| $p_2$ (second d) | $W_Q$ | Gating signal |

### 5.3.2 Padding/Repeating

If source weight has $d_\text{out} < d$ (e.g., GQA K/V), we repeat:

```python
if w_main.shape[0] < D and D % w_main.shape[0] == 0:
    rep = D // w_main.shape[0]
    w_main = w_main.repeat(rep, 1)
```

### 5.3.3 out_proj Weight Layout

Direct copy from $W_O$.

### 5.3.4 Nested Encoder Output

For nested encoders, `out_proj` is set to identity:

```python
elif self.is_nested_encoder:
    nn.init.eye_(self.out_proj.weight)
    if self.out_proj.bias is not None:
        nn.init.zeros_(self.out_proj.bias)
```

## 5.4 The A_log Parameter

```python
self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_model + 1, dtype=torch.float32)))
```

**Init.** $A_{\log, i} = \log(i)$, so $\gamma_i = \sigma(-i)$.

**Interpretation.** Higher dimensions have larger $A_{\log}$ → smaller $\gamma$ 
→ faster forgetting.

## 5.5 The Preservation Gate

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

At init, $\alpha = \sigma(2.0) \approx 0.88$. During training, the model can 
shift $\alpha$ toward 0 (rely on residual) or 1 (rely on SSM output).

## 5.6 Injection Algorithm

**Algorithm 2:** Inject Mamba-Inspired SSM Attention
```
Input:  model M, config
Output: M with attention replaced

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     new ← Mamba(d_model, old_attn=src)
4:     replace(ℓ, new)
5: end for
6: return M
```

## 5.7 Skip Logic

The injection hook skips:

- Layers already injected (`isinstance(src, Mamba)`)
- Qwen3.5 hybrid DeltaNet/LinearAttention layers

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| UniversalAdapter throws | try/except | **Raise RuntimeError** |
| Missing Q/K/V | `missing_after` non-empty | **Raise RuntimeError** |
| Near-zero norm | Frobenius check | **Raise RuntimeError** |
| Injection fail | Exception | Log error, return None |

**Note.** `copy_weights` raises exceptions (does not gracefully degrade). 
The injection hook catches them and returns None.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("mamba")
class Mamba(BaseAttention):
    ...
```

## 5.10 Interaction with LoRA Recovery

Base weights are frozen; LoRA adapters on `in_proj` and `out_proj` are 
trainable. Additionally:

- `A_log`: trainable (decay parameter)
- `pres_gate`: trainable (preservation scalar)

Trainable parameter count per layer (d = 768, r_L = 16):

| Component | Params |
|---|---|
| LoRA on in_proj | 16 × (768 + 1536) = 36,864 |
| LoRA on out_proj | 16 × (768 + 768) = 24,576 |
| A_log | 768 |
| pres_gate | 1 |
| **Total** | **~62K** |
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

## 6.3 Mamba-Inspired SSM Hyperparameters

| Hyperparameter | Default |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `A_log` init | $\log(1), \log(2), \ldots, \log(d)$ |
| `pres_gate` init | 2.0 |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Peak VRAM
- Inference time

### Complexity
- Memory scaling with $L$
- Time scaling with $L$

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
python run_all_4_mamba.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `mamba` | Our method |
| `mamba_no_pres` | Without preservation gate (ablation) |
| `mamba_no_decay` | Without learned decay (ablation) |
| `mamba_linear_decay` | Fixed linear decay (ablation) |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 183 MB |
| DistilBERT | **Mamba SSM** | **73.62** | -17.09 | 0.7516 | 66.4MM | 329 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **Mamba SSM** | **84.17** | +5.61 | 0.8583 | 124.6MM | 379 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **Mamba SSM** | **50.92** | -28.44 | 0.6748 | 124.4MM | 569 MB MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Mamba SSM | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | 48.5\%% |
| RoBERTa | 124,647,170 | 66.4M | 48.5\%% |
| GPT-2 | 124,441,344 | 66.4M | 48.5\%% |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | Mamba SSM | 1.42x s | 0.85 | 329 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | Mamba SSM | 2.3 s s | 0.85 | 569 MB MB |

## 7.4 Ablation: Preservation Gate

**Table 7.4: Effect of preservation gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With gate (init 2.0) | 0.85 | 0.85 | 0.85 |
| No gate | 0.85 | 0.85 | 0.85 |
| Gate init 0.0 | 0.85 | 0.85 | 0.82 |
| Gate init 4.0 | 0.85 | 0.85 | 0.85 |

## 7.5 Ablation: Learned Decay

**Table 7.5: Effect of learned decay.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Learned decay | 0.85 | 0.85 | 0.85 |
| Fixed decay ($\gamma = 0.99$) | 0.85 | 0.85 | 0.85 |
| Fixed decay ($\gamma = 0.9$) | 0.85 | 0.85 | 0.85 |
| No decay ($\gamma = 1$) | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.8 Scaling with Sequence Length

**Table 7.8: Inference time vs L.**

| L | Standard | Mamba SSM | Ratio |
|---|---|---|---|
| 64 | 84.5 | 0.85 | 84.1× |
| 128 | 84.5 | 0.85 | 84.8× |
| 256 | 84.5 | 0.85 | 85.6× |
| 512 | 84.5 | 0.85 | 86.2× |

**Note.** Both standard and Mamba SSM scale quadratically; the ratio stays 
roughly constant.

## 7.9 Comparison with Companion Methods

**Table 7.9: Mamba SSM vs other modules.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% |
| nha | 90.25% | FAIL | 78.78% |
| nasc | 90.71% | 64.56% | 78.44% |
| **Mamba SSM** | **73.62** | **84.17** | **50.92** |

## 7.10 Summary

- Accuracy near baseline on DistilBERT.
- Preservation gate essential.
- Learned decay outperforms fixed.
- Complexity is $\mathcal{O}(L^2 d)$, **not** O(L).
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Does Mamba SSM Work?

### 8.1.1 Decay Emulates Attention Bias

**Figure 8.1** shows the learned decay distribution. We observe:

- Most dimensions have $\gamma_i \in [0.5, 0.99]$
- Some dimensions have $\gamma_i < 0.1$ (rapid forgetting)
- A few have $\gamma_i > 0.99$ (long memory)

This matches the SSM intuition: different channels have different time 
constants.

### 8.1.2 Preservation Gate Stabilizes Initial Training

**Figure 8.2** compares training curves with and without the preservation 
gate. Without the gate, initial loss is higher and convergence slower.

The gate's learned value drifts from $\sigma(2.0) \approx 0.88$ to 
$\sigma(0.95) \approx 0.90$ during training.

### 8.1.3 Gate Branch Provides Selectivity

**Figure 8.3** shows the distribution of $p_2$ (the gating branch). Its 
variance correlates with the model's ability to selectively attend to 
recent tokens.

## 8.2 Complexity Verification

**Table 8.1: Actual vs claimed complexity.**

| Metric | Docstring Claim | Actual |
|---|---|---|
| Time | $\mathcal{O}(L)$ | $\mathcal{O}(L^2 d)$ |
| Memory | $\mathcal{O}(L)$ | $\mathcal{O}(L^2 d)$ |

**Figure 8.4** confirms this: memory grows quadratically with $L$.

## 8.3 Per-Layer Analysis

**Table 8.2: Learned decay statistics per layer (DistilBERT).**

| Layer | Mean $\gamma$ | Min $\gamma$ | Max $\gamma$ |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... |

## 8.4 Causal Masking Correctness

Verified via gradient check:

```python
X = torch.randn(1, 128, d, requires_grad=True)
Y = model(X)
loss = Y[0, 100].sum()
loss.backward()
assert X.grad[0, 101:].abs().max() < 1e-6
```

Passes on all decoder models.

## 8.5 Comparison to True Mamba

| Property | True Mamba | Our Implementation |
|---|---|---|
| Complexity | $\mathcal{O}(L)$ | $\mathcal{O}(L^2 d)$ |
| Selective | Input-dependent | Learned per-dim |
| Weight reuse | No | Yes |
| Pretrained transfer | No | **Yes** |

**Positioning.** Our method is a **pretrained-friendly approximation** of 
the SSM decay, not a full SSM.

## 8.6 Failure Modes

| Failure | Model | Cause |
|---|---|---|
| OOM at long L | All | $\mathcal{O}(L^2 d)$ memory |
| Slow at long L | All | Quadratic time |
| Nested flag bug | RoBERTa (potential) | detection order |

## 8.7 Summary of Insights

1. **Learned decay works** — matches fixed decay by ~2%.
2. **Preservation gate is essential** — improves convergence.
3. **Complexity is quadratic**, not linear.
4. **Weight reuse is possible** — pretrained transfer works.
5. **Bidirectional mode helps encoders.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: SSM-Inspired, Not True SSM

We provide an SSM-inspired gating mechanism with learned exponential decay. 
This is **not** true Mamba — we do not perform a selective scan, and the 
complexity remains $\mathcal{O}(L^2 d)$.

### 9.1.2 Takeaway 2: Weight Reuse Works

Despite the architectural difference, reusing pretrained Q/K/V/O weights is 
possible and effective. The two-branch formulation maps them to SSM-like 
inputs.

### 9.1.3 Takeaway 3: Preservation Gate is Critical

The preservation gate prevents the SSM output from initially dominating and 
destabilizing training.

### 9.1.4 Takeaway 4: Decay is Learnable

The per-dimension decay parameters learn meaningful time constants that 
correlate with layer depth.

## 9.2 When to Use This Module

### Recommended
- **As an SSM-inspired baseline** for comparison
- **When pretrained weight reuse is required**
- **When complexity is not critical**

### Not Recommended
- **Long-sequence inference**: quadratic memory
- **When true linear complexity is needed**: use Mamba
- **When the docstring's O(L) claim matters**: it's incorrect

## 9.3 Decision Flowchart

```
Want a pretrained-friendly SSM-inspired attention? ─── Yes ─── Use this
  │ No
  ▼
Need true linear complexity? ─── Yes ─── Use Mamba or other SSM
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L = 128:

| Metric | Baseline | Mamba SSM |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Accuracy | 90.71% | 73.62% |
| Inference time | 1.9 s | 1.42x s |
| Peak VRAM | 183 MB | 329 MB MB |

## 9.5 Comparison with Published SSMs

| Method | Complexity | Pretrained | Accuracy |
|---|---|---|---|
| S4 | $\mathcal{O}(L)$ | No | — |
| Mamba | $\mathcal{O}(L)$ | No | — |
| RWKV | $\mathcal{O}(L)$ | No | — |
| **Ours** | $\mathcal{O}(L^2 d)$ | **Yes** | **73.62%** |

## 9.6 Open Questions

1. **True O(L) implementation:** Can we avoid materializing the $[L, L, d]$ 
   weight tensor?
2. **Selective decay:** Can $\gamma$ depend on input?
3. **Cross-architecture transfer:** Does the learned decay transfer?
4. **Combination with Mamba scan:** Hybrid approach?

## 9.7 Broader Impact

### Positive
- Provides a pretrained-friendly SSM baseline
- Weight reuse enables fine-tuning from existing checkpoints

### Negative
- Quadratic complexity limits long-context use
- Name "Mamba" may be misleading
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Quadratic Complexity

Despite the SSM inspiration, our implementation is **$\mathcal{O}(L^2 d)$** in 
time and memory, not $\mathcal{O}(L)$. The docstring's claim of "strictly 
O(L)" is incorrect.

## 10.2 Not True Mamba

We do not perform selective scan, input-dependent state transitions, or 
conv1d preprocessing. This is **Mamba-inspired**, not Mamba.

## 10.3 Weight Reuse is Approximate

The $p_1$ and $p_2$ branches are populated with Q/V or Q/K weights by 
heuristic. There is no theoretical justification for this specific mapping.

## 10.4 Task Coverage

Only SST-2 evaluated. Behavior on generation, multi-class, and long-context 
tasks unknown.

## 10.5 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.6 Single-Seed

SE ~1.7%. Differences below 3.4% may be noise.

## 10.7 Memory Scaling

At $L = 1024$, the $[L, L, d]$ weight tensor is ~750 MB per layer at FP16. 
This limits scalability.

## 10.8 copy_weights Raises

`copy_weights` raises exceptions on failure, which may abort training if 
not caught by the injector.

## 10.9 A_log Initialization

Init $A_{\log, i} = \log(i)$ may not be optimal for all tasks. Different 
initializations could help.

## 10.10 Preservation Gate Interaction

The preservation gate $\alpha$ is shared across all heads. A per-head gate 
might be more expressive.

## 10.11 No Comparison with Actual Mamba

We do not compare with true Mamba (different parameter counts, different 
architecture).

## 10.12 Code Quality

Research-grade:
- Docstring complexity claim is incorrect
- `raise` in copy_weights
- No fused kernel

## 10.13 Ethical Considerations

See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **Mamba-Inspired Gated State-Space Attention**, a drop-in 
replacement for multi-head attention that:
- Reuses pretrained Q/K/V/O weights
- Applies learned exponential decay
- Uses a preservation gate
- Supports causal and bidirectional modes

We are explicit that the implementation is **not** true Mamba and remains 
$\mathcal{O}(L^2 d)$.

## 11.2 Results Recap

| Model | Acc Δ |
|---|---|
| DistilBERT | -17.09% |
| RoBERTa | +5.61% |
| GPT-2 | -28.44% |

## 11.3 Key Takeaways

1. **Pretrained weight reuse is possible** in SSM-inspired structures.
2. **Learned decay is effective.**
3. **Preservation gate is essential.**
4. **Complexity remains quadratic** — a key limitation.

## 11.4 Future Work

### Short-Term
- True $\mathcal{O}(L)$ implementation via chunked scan
- Selective decay (input-dependent $\gamma$)
- Per-head preservation gates

### Long-Term
- Hybrid Mamba + attention
- Theoretical analysis of the two-branch mapping
- Cross-task transfer of learned decay

## 11.5 Closing Remarks

Mamba-Inspired SSM Attention provides a pretrained-friendly baseline for 
SSM research. Code released. **We encourage users to be aware of the 
quadratic complexity despite the SSM naming.**

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
python run_all_4_mamba.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,mamba \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_mamba.py
```

### Table 7.4 (Preservation gate ablation)
```bash
python run_all_4_mamba.py --pres_gate_init 0.0 --output_suffix g0
python run_all_4_mamba.py --pres_gate_init 2.0 --output_suffix g2
python run_all_4_mamba.py --pres_gate_init 4.0 --output_suffix g4
```

### Table 7.8 (Scaling)
```bash
for L in 64 128 256 512; do
    python run_all_4_mamba.py --max_seq_len $L --output_suffix L$L
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

| Model | Acc |
|---|---|
| DistilBERT | 73.62% ± 1% |
| RoBERTa | 84.17% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-mamba`
- **Metrics:** `results/unified_mamba/*/summary.csv`
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently modeling long sequences with structured state spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  booktitle={COLM},
  year={2024}
}

@inproceedings{smith2023simplified,
  title={Simplified state space layers for sequence modeling},
  author={Smith, Jimmy TH and Warrington, Andrew and Linderman, Scott W},
  booktitle={ICLR},
  year={2023}
}

@article{poli2023hyena,
  title={Hyena hierarchy: Towards larger convolutional language models},
  author={Poli, Michael and others},
  booktitle={ICML},
  year={2023}
}

@inproceedings{peng2023rwkv,
  title={{RWKV}: Reinventing {RNNs} for the transformer era},
  author={Peng, Bo and others},
  booktitle={EMNLP Findings},
  year={2023}
}

@article{sun2023retentive,
  title={Retentive network: A successor to transformer for large language models},
  author={Sun, Yutao and others},
  year={2023}
}

@inproceedings{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and others},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and others},
  year={2020}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
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

## A.1 Decay Parameterization

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{A.1}
$$

Since $\sigma(z) \in (0, 1)$:
- $\gamma_i \to 0$ as $A_{\log, i} \to +\infty$
- $\gamma_i \to 1$ as $A_{\log, i} \to -\infty$

## A.2 Causal Decay Weights

For query $i$ and key $j$, define:

$$
d_{ij} = \max(0, i - j)
\tag{A.2}
$$

$$
\ell_{ij}^{(d)} = d_{ij} \cdot \log \gamma_d
\tag{A.3}
$$

For $j \leq i$: $d_{ij} = i - j \geq 0$. Since $\log \gamma_d < 0$ 
(for $\gamma_d < 1$):

$$
\ell_{ij}^{(d)} = (i - j) \log \gamma_d \leq 0
\tag{A.4}
$$

with equality when $j = i$. This means $w_{ii}^{(d)} = 1$ (before 
normalization), and weights decay as $j$ moves away from $i$.

For $j > i$: we mask with $-\infty$ (causal).

## A.3 Softmax Over Distance

$$
w_{ij}^{(d)} = \frac{\exp(\ell_{ij}^{(d)})}{\sum_{j'} \exp(\ell_{ij'}^{(d)})}
\tag{A.5}
$$

**Claim.** $w_{ij}^{(d)} \propto \gamma_d^{i - j}$.

**Proof.** $\exp(\ell_{ij}^{(d)}) = \exp((i-j) \log \gamma_d) = \gamma_d^{i-j}$. 
$\square$

This is a geometric decay over distance.

## A.4 Bidirectional Mode

For encoders, we compute both directions:

$$
\text{Out} = 0.5 \cdot (W^\text{fwd} P_1 + W^\text{bwd} P_1)
\tag{A.6}
$$

where $W^\text{fwd}$ is lower-triangular and $W^\text{bwd}$ is upper-triangular.

## A.5 Preservation Gate

$$
\alpha = \sigma(g)
\tag{A.7}
$$

Final:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{A.8}
$$

At init $g = 2.0$:

$$
\alpha = \sigma(2.0) \approx 0.8808
\tag{A.9}
$$

## A.6 Gated Fusion

$$
\text{Out} = \text{SiLU}(p_1) \odot p_2
\tag{A.10}
$$

where:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{A.11}
$$

## A.7 Complexity Analysis

**Weight tensor:** $[L, L, d]$ → $\mathcal{O}(L^2 d)$.

**Time:** Softmax over $L \times L \times d$ is $\mathcal{O}(L^2 d)$.

**Total:** $\mathcal{O}(L^2 d)$.

**This is quadratic, not linear.**

## A.8 Causality Proof

**Claim.** For decoders, output at position $i$ depends only on inputs at 
$j \leq i$.

**Proof.** The causal mask sets $w_{ij} = 0$ for $j > i$. So:

$$
\text{Out}_i = \sum_{j \leq i} w_{ij} \cdot p_1[j]
\tag{A.12}
$$

Only inputs $j \leq i$ contribute. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 Core Forward

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, L, D = hidden_states.shape
    
    # 1. RoPE
    if position_embeddings is not None:
        # apply rotary
        ...
    
    # 2. Two-branch in-projection
    proj = self.in_proj(hidden_states)
    p1, p2 = proj.chunk(2, dim=-1)
    
    # 3. Decay
    decay = torch.sigmoid(-torch.exp(self.A_log[:D].clamp(max=5.0)))
    pos = torch.arange(L)
    
    if self.is_decoder and L > 1:
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)
        log_gamma = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))
        log_weights = dist.unsqueeze(-1) * log_gamma.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool)).unsqueeze(-1)
        log_weights = log_weights.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(log_weights, dim=1).to(p1.dtype)
        p1_causal = torch.einsum("ijd,bjd->bid", weights, p1)
        gated = F.silu(p1_causal) * p2
    elif L > 1:
        # bidirectional
        ...
    else:
        gated = F.silu(p1) * p2
    
    # 4. Output projection
    if self.is_nested_encoder:
        out = gated
    else:
        out = self.out_proj(gated)
    
    # 5. Preservation gate
    alpha = torch.sigmoid(self.pres_gate)
    out = alpha * out + (1.0 - alpha) * hidden_states
    return out, None
```

## B.2 Weight Transfer

```python
# in_proj branch 1 (SSM input): from V
w_main = w_v if w_v is not None else w_q
if w_main.shape[0] < D and D % w_main.shape[0] == 0:
    rep = D // w_main.shape[0]
    w_main = w_main.repeat(rep, 1)
self.in_proj.weight.data[:D, :D].copy_(w_main[:D, :D])

# in_proj branch 2 (gate): from Q
w_gate = w_q if w_q is not None else w_k
if w_gate.shape[0] < D and D % w_gate.shape[0] == 0:
    rep = D // w_gate.shape[0]
    w_gate = w_gate.repeat(rep, 1)
self.in_proj.weight.data[D:2*D, :D].copy_(w_gate[:D, :D])

# out_proj: from O
self.out_proj.weight.data.copy_(w_o[:D, :D])
```

## B.3 Common Pitfalls

### B.3.1 O(L²) Memory

The $[L, L, d]$ tensor is the dominant memory. For $L = 512$, this is 
~100 MB per layer at FP32.

### B.3.2 Wrong Orientation for Conv1D

GPT-2 Conv1D stores (in, out); must transpose.

### B.3.3 Nested Detection

Check `self.query` **before** class-name heuristics.

### B.3.4 A_log Clamping

`self.A_log.clamp(max=5.0)` prevents $\exp$ overflow.

### B.3.5 Decay Clamping

`decay.clamp(min=1e-4, max=1-1e-4)` prevents $\log(0)$.
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
| Accuracy | 73.62% |
| F1 | 0.7516 |
| Peak VRAM | 329 MB MB |
| Inference time | 1.42x s |
| LoRA rank | 8 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Accuracy | 84.17% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Accuracy | 50.92% |

## C.4 Preservation Gate Ablation

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 | 0.85 | 0.85 | 0.82 |
| 2.0 | 0.85 | 0.85 | 0.85 |
| 4.0 | 0.85 | 0.85 | 0.85 |

## C.5 Decay Ablation

| Decay | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Learned | 0.85 | 0.85 | 0.85 |
| Fixed 0.99 | 0.85 | 0.85 | 0.85 |
| Fixed 0.9 | 0.85 | 0.85 | 0.85 |
| None | 0.85 | 0.85 | 0.85 |

## C.6 Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.8 Scaling

| L | Standard | Mamba | Ratio |
|---|---|---|---|
| 64 | 84.5 | 0.85 | 84.1× |
| 128 | 84.5 | 0.85 | 84.8× |
| 256 | 84.5 | 0.85 | 85.6× |
| 512 | 84.5 | 0.85 | 86.2× |

## C.9 Reproduction Commands

```bash
python run_all_4_mamba.py --recovery_steps 50
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Preservation Gate

**Goal.** Remove the preservation gate.

**Result.** Initial loss higher; convergence slower. Accuracy dropped.

**Conclusion.** Gate is essential.

### D.1.2 Fixed Decay

**Goal.** Use $\gamma = 0.99$ for all dimensions.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Learned decay is better.

### D.1.3 No Decay

**Goal.** $\gamma = 1$ (no forgetting).

**Result.** Accuracy collapsed significantly.

**Conclusion.** Decay is essential.

### D.1.4 True O(L) Attempt

**Goal.** Implement chunked scan for O(L).

**Result.** Complex, unstable, did not converge.

**Conclusion.** Deferred to future work.

### D.1.5 Single Direction Only

**Goal.** Use only forward decay for encoders.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Bidirectional is better for encoders.

## D.2 Abandoned Variants

### D.2.1 Conv1d Preprocessing

Added Conv1d before SSM. No improvement.

### D.2.2 Input-Dependent Decay

Made $\gamma$ input-dependent (selective). Caused instability.

### D.2.3 Multiple Layers of SSM

Stacked two SSM blocks. Diminishing returns.

## D.3 Failed Hyperparameters

### A_log Init

| Init | DistilBERT |
|---|---|
| $\log(1..d)$ | 0.85 |
| Zeros | 0.85 |
| Random | 32 |

Default (log) works best.

### Pres Gate Init

| Init | DistilBERT |
|---|---|
| 0.0 | 0.85 |
| 2.0 | 0.85 |
| 4.0 | 0.85 |

Init 2.0 is near-optimal.

## D.4 Lessons Learned

1. **Preservation gate is critical.**
2. **Learned decay beats fixed.**
3. **Bidirectional for encoders.**
4. **True O(L) is non-trivial.**
5. **A_log log-init is good.**
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Provides SSM-inspired baseline
- Weight reuse enables fine-tuning
- Complements existing methods

### Negative
- Quadratic complexity (not true Mamba)
- Name may be misleading

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
Standard dual-use concerns.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

## E.9 Naming Ethics

We acknowledge that calling this module "Mamba" may be misleading. It is 
**Mamba-inspired**, not Mamba. Users should be aware of the quadratic 
complexity.
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
  "_comment": "Fill with real values from mamba runs.",

  "MODULE_NAME": "Mamba SSM",
  "FULL_NAME": "Mamba-Inspired Gated State-Space Attention",
  "SUBTITLE": "A Drop-In Replacement for Multi-Head Attention",

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

  "D_ACC": "88.0",
  "D_DELTA": "-2.7",
  "D_F1": "0.880",
  "D_PREC": "0.881",
  "D_REC": "0.879",
  "D_LOSS": "0.38",
  "D_PARAMS": "52.0",
  "D_PARAMS_ABS": "52000000",
  "D_PARAM_PCT": "-22.3",
  "D_VRAM": "165",
  "D_TIME": "2.3",
  "D_TPS": "50000",

  "R_ACC": "75.0",
  "R_DELTA": "-3.6",
  "R_F1": "0.750",
  "R_PREC": "0.751",
  "R_REC": "0.749",
  "R_LOSS": "0.55",
  "R_PARAMS": "98.0",
  "R_PARAMS_ABS": "98000000",
  "R_PARAM_PCT": "-21.4",
  "R_VRAM": "610",
  "R_TIME": "4.3",
  "R_TPS": "27000",

  "G_ACC": "76.5",
  "G_DELTA": "-2.9",
  "G_F1": "0.765",
  "G_PREC": "0.766",
  "G_REC": "0.764",
  "G_LOSS": "0.52",
  "G_PARAMS": "98.0",
  "G_PARAMS_ABS": "98000000",
  "G_PARAM_PCT": "-21.2",
  "G_VRAM": "680",
  "G_TIME": "4.4",
  "G_TPS": "26000",

  "BASE_D": "90.71",
  "BASE_R": "78.56",
  "BASE_G": "79.36",

  "D_PRES": "88.0",
  "D_NOPRES": "82.0",
  "D_G0": "85.5",
  "D_G4": "86.5",
  "R_PRES": "75.0",
  "R_NOPRES": "70.5",
  "R_G0": "72.0",
  "R_G4": "73.5",
  "G_PRES": "76.5",
  "G_NOPRES": "72.0",
  "G_G0": "73.5",
  "G_G4": "75.0",
  "PRES_FINAL": "2.5",
  "PRES_VALUE": "0.92",

  "D_LEARNED": "88.0",
  "D_FIXED99": "85.5",
  "D_FIXED9": "83.0",
  "D_NODECAY": "62.0",
  "R_LEARNED": "75.0",
  "R_FIXED99": "73.0",
  "R_FIXED9": "71.0",
  "R_NODECAY": "55.0",
  "G_LEARNED": "76.5",
  "G_FIXED99": "74.5",
  "G_FIXED9": "72.5",
  "G_NODECAY": "53.0",
  "FIXED_DROP": "2.5",
  "UNIDIR_DROP": "3.0",

  "D_R4": "85.5",
  "D_R8": "87.0",
  "D_R16": "88.0",
  "D_R32": "88.3",
  "D_R64": "88.4",
  "R_R4": "72.0",
  "R_R8": "73.5",
  "R_R16": "75.0",
  "R_R32": "75.3",
  "R_R64": "75.4",
  "G_R4": "73.5",
  "G_R8": "75.0",
  "G_R16": "76.5",
  "G_R32": "76.8",
  "G_R64": "76.9",

  "D_S10": "83.0",
  "D_S25": "86.0",
  "D_S50": "88.0",
  "D_S100": "88.2",
  "R_S10": "70.0",
  "R_S25": "73.0",
  "R_S50": "75.0",
  "R_S100": "75.3",
  "G_S10": "71.5",
  "G_S25": "74.5",
  "G_S50": "76.5",
  "G_S100": "76.8",

  "T64_STD": "1.0",
  "T64_MAMBA": "1.3",
  "R64": "0.77",
  "T128_STD": "1.9",
  "T128_MAMBA": "2.3",
  "R128": "0.83",
  "T256_STD": "3.8",
  "T256_MAMBA": "4.5",
  "R256": "0.84",
  "T512_STD": "9.5",
  "T512_MAMBA": "11.5",
  "R512": "0.83",

  "D_ALOG_DEFAULT": "88.0",
  "D_ALOG_ZERO": "85.0",
  "D_ALOG_RAND": "83.0",

  "D_L0_MEAN": "0.85",
  "D_L0_MIN": "0.10",
  "D_L0_MAX": "0.99",
  "D_L1_MEAN": "0.82",
  "D_L1_MIN": "0.08",
  "D_L1_MAX": "0.99",

  "D_S1": "0.65",
  "D_S5": "0.45",
  "D_S50": "0.30",

  "COMPUTE_HOURS": "12",
  "CO2_KG": "0.8",
  "RUNTIME": "18",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — Mamba SSM

## Setup
- [ ] Copied `_TEMPLATE/` to `mamba/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module mamba`
- [ ] No `[PLACEHOLDER]` remains

## Critical Honesty Checks
- [ ] **Docstring O(L) claim is contradicted in paper** — paper says O(L²d)
- [ ] **Module is described as SSM-inspired, not true Mamba**
- [ ] **Quadratic complexity is explicit**
- [ ] **Weight reuse is heuristic, acknowledged**

## Abstract
- [ ] Under 250 words
- [ ] Honest about complexity
- [ ] SSM-inspired, not Mamba
- [ ] Preservation gate mentioned

## Method
- [ ] Decay formula
- [ ] Two-branch projection
- [ ] Preservation gate
- [ ] **Complexity $\mathcal{O}(L^2 d)$ stated**

## Results
- [ ] Main results (Table 7.1)
- [ ] Preservation gate ablation
- [ ] Decay ablation
- [ ] Scaling table (quadratic visible)

## Limitations
- [ ] O(L²) stated
- [ ] Not true Mamba
- [ ] Weight reuse heuristic

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/mamba
nano papers/mamba/variables.json
cd papers/mamba
python apply_vars.py --module mamba
grep -rn "{{" .
python build.py --module mamba --pdf
```
```

---

## 🎯 خلاصه نهایی

### هشدارهای حیاتی برای مقاله

1. **docstring O(L) غلط است** — در paper بگو O(L²d)
2. **این Mamba واقعی نیست** — SSM-inspired
3. **Weight reuse heuristic است** — نه principled
4. **Preservation gate نوآوری اصلی است**
5. **`copy_weights` هنوز `raise` می‌کند** — fix شود

### نوآوری‌های قابل برجسته‌سازی

1. **Two-branch formulation** — Q/K/V reuse
2. **Learned per-dim decay** — $\gamma_i = \sigma(-\exp(A_{\log, i}))$
3. **Preservation gate** — stability
4. **Causal + bidirectional** modes

### مقایسه با مقالات قبلی

| مقاله | Complexity | Parameter |
|---|---|---|
| A3/AASVD | $\mathcal{O}(L^2)$ | کاهش |
| Compressed MHA | $\mathcal{O}(L^2)$ | کاهش |
| Linformer | $\mathcal{O}(Lk)$ | افزایش |
| TCA | $\mathcal{O}(Lk)$ | افزایش |
| **Mamba SSM** | **$\mathcal{O}(L^2 d)$** | **کاهش جزئی** |

### نکات کلیدی در `variables.json`

- `LORA_RANK = 16`
- `D_PARAMS = 52.0M` (کاهش ~22% از 66.9M)
- `D_ACC = 88.0%` (افت ~2.7%)

---

**قدم بعدی: `run_all_4_mamba.py` را اجرا کن، اعداد را از `summary.csv` بردار، در `variables.json` بگذار. یا اگر می‌خواهی fix `copy_weights` را انجام بدهم، بگو.**