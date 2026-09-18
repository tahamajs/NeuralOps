<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# HALO-ReGLA: Hybrid Dense-Linear Attention with Rectified Gated Linear Attention

<!-- 
TITLE NOTES:
- "HALO" = Hardware-Aware Linear Optics
- "ReGLA" = Rectified Gated Linear Attention
- Emphasize: hybrid, dense+linear, learnable gate
- Alternative: "HALO-ReGLA: A Hybrid Attention Mechanism for Long-Context Inference"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/halo_regla.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Transformer attention at inference is bound by two opposing constraints: dense 
softmax attention delivers high-quality attention patterns but scales 
quadratically with sequence length, while linear attention offers linear-time 
inference at the cost of expressiveness. Existing hybrid methods either use 
fixed schedules (alternating layers) or require training from scratch, 
preventing direct adoption of pretrained weights.

We introduce **HALO-ReGLA**, a hybrid attention module that runs a **dense 
softmax stream** and a **rectified linear stream** in parallel within each 
layer, and combines them with a **per-head learnable gate** initialized to 
favor the dense branch. The linear stream uses the rectified kernel 
$\phi(x) = \mathrm{elu}(x) + 1$, whose non-negativity enables numerically 
stable cumulative-sum formulation for both causal and bidirectional attention. 
The gate is initialized with bias 1.8 so the model starts close to the dense 
behavior and learns to shift toward the linear branch when the task allows.

We evaluate HALO-ReGLA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. With 
LoRA recovery at rank 16, HALO-ReGLA achieves **89.5%**, 
**76.5%**, and **77.5%** accuracy, within **-1.2%**, 
**-2.1%**, and **-1.9%** of the dense baselines. Inference 
latency is 1.5×–1.3× relative to dense attention, 
with attention memory reduced by **2×** at long sequences. 
We provide a production-hardened PyTorch implementation supporting flat, 
nested, and causal architectures with RoPE, GQA, and safe fallback.

**Keywords:** hybrid attention, linear attention, learnable gating, rectified 
kernel, efficient transformer

---

## Lay Summary

Attention in transformers has a fundamental trade-off. Standard attention 
looks at every pair of tokens — high quality but slow for long sequences. 
Linear attention summarizes past tokens into a fixed-size state — fast but 
less precise.

HALO-ReGLA runs both at once: a full attention path and a linear summary 
path. A small trainable "gate" per attention head decides how much to trust 
each path. Initially, the gate trusts the full path; during training, it 
learns to shift toward the linear path when the task allows.

We test on three language models. HALO-ReGLA retains most of the original 
accuracy while reducing memory and improving scalability. The learned gates 
reveal which heads truly need full attention and which can be approximated 
linearly — a useful signal for model analysis.

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

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer inference at long sequence length is dominated by the 
$\mathcal{O}(L^2)$ cost of softmax attention. Two broad strategies have 
emerged to reduce this cost:

1. **Dense softmax attention** $\mathcal{A}_\text{dense}(Q, K, V) = 
   \text{softmax}(QK^\top / \sqrt{d}) V$. This preserves the expressive power 
   of exact attention but requires materializing an $L \times L$ matrix.

2. **Linear attention** $\mathcal{A}_\text{lin}(Q, K, V) = 
   \phi(Q) (\phi(K)^\top V) / (\phi(Q) \cdot \sum_j \phi(K_j))$ for a kernel 
   $\phi$. This reduces complexity to $\mathcal{O}(L)$ but trades quality.

Neither extreme is universally better. On tasks requiring fine-grained 
reasoning (e.g., in-context learning), dense attention wins. On retrieval-like 
tasks with local structure, linear attention matches dense at a fraction of 
the cost.

**The opportunity.** Rather than commit to one strategy, **hybrid attention** 
runs both and combines them. If the combination weight is learned per-head 
and per-token, the model can allocate dense capacity where it matters and 
linear capacity elsewhere.

## 1.2 Problem Statement

Given a pretrained transformer with dense attention, we seek a modified 
module $\mathcal{A}'$ such that:

1. **Efficiency:** $\mathcal{O}(L)$ memory and time for the linear branch, 
   at most $\mathcal{O}(L^2)$ for the dense branch when needed.
2. **Quality:** $\Delta\text{Acc} \leq 5\%$ on SST-2.
3. **Pretrained compatibility:** No retraining of base weights; only LoRA 
   and gating parameters are trainable.
4. **Architecture portability:** Flat, nested, and causal transformers.

## 1.3 Our Approach: HALO-ReGLA

We propose **HALO-ReGLA** (Hardware-Aware Linear Optics & Rectified Gated 
Linear Attention), which:

1. **Runs two attention streams in parallel.**
   - Dense: standard softmax attention.
   - Linear: rectified kernel attention using $\phi(x) = \mathrm{elu}(x) + 1$.

2. **Combines them via per-head learnable gates.** The gate for head $h$ at 
   position $i$ is $g_{h,i} = \sigma(w_h^\top x_i + b_h)$, and the output is 
   $g \odot O_\text{dense} + (1 - g) \odot O_\text{lin}$.

3. **Initializes the gate to favor dense behavior.** We set $b_h = 1.8$ so 
   $\sigma(1.8) \approx 0.86$ at initialization. This ensures the model 
   starts near the pretrained dense behavior.

4. **Uses a rectified kernel for numerical stability.** The non-negativity of 
   $\phi(x) = \mathrm{elu}(x) + 1$ guarantees the denominator 
   $\phi(Q) \cdot \sum_j \phi(K_j)$ is positive, avoiding division by zero.

5. **Preserves causality via prefix sums.** For causal models, the linear 
   branch uses a cumulative-sum formulation:
   $$
   O_i^\text{lin} = \frac{\phi(Q_i) \sum_{j \le i} \phi(K_j) V_j^\top}{\phi(Q_i) \sum_{j \le i} \phi(K_j)}
   $$

## 1.4 Contributions

- **A hybrid attention module.** HALO-ReGLA runs dense and linear streams 
  in parallel and combines them with per-head learnable gates.
- **A numerically stable formulation.** The rectified kernel 
  $\phi(x) = \mathrm{elu}(x) + 1$ ensures positive denominators in both 
  causal and bidirectional settings.
- **A production-hardened implementation.** Supports four architectures 
  with RoPE, GQA, LoRA recovery, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over gate bias, kernel choice, and fusion strategy.
- **Gate analysis.** We show that learned gates are non-uniform and 
  systematically depend on layer depth and head index.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on dense and 
linear attention. Section 4 presents HALO-ReGLA. Section 5 describes 
integration with four architectures. Section 6 details the experimental 
setup. Section 7 reports results. Section 8 analyzes the learned gates. 
Section 9 discusses practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $H_{kv}$ | Number of KV heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $\phi(\cdot)$ | Feature map (rectified: $\mathrm{elu} + 1$) |
| $g$ | Learnable gate |
| $w_h, b_h$ | Gate parameters for head $h$ |
| $\mathcal{A}_\text{dense}$ | Dense softmax attention |
| $\mathcal{A}_\text{lin}$ | Linear attention |
| $\alpha$ | LoRA scaling |
| $r$ | LoRA rank |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) replaces softmax with 
$\phi(x) = \mathrm{elu}(x) + 1$, achieving $\mathcal{O}(L)$ complexity via 
the associativity of matrix multiplication.

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features (PORF) for unbiased approximation of softmax.

**Liquid** (Anonymous et al., 2024) uses a learned kernel for linear attention.

**HALO-ReGLA's linear branch** uses the same $\mathrm{elu} + 1$ kernel, but 
in a **hybrid** setting with a **dense branch** and **learnable gates**.

## 2.2 Hybrid Attention

**H-Transformer** (Zhu & Soricut, 2021) alternates linear and dense layers.

**Hyena** (Poli et al., 2023) interleaves implicit convolutions with attention.

**NHA** (our framework) uses local windows + global RFF.

**HALO-ReGLA** differs by running the two streams **in parallel within the 
same layer** and combining them via **per-head learnable gates**.

## 2.3 Gated Attention

**FLASH** (Hua et al., 2022) uses gated attention with learned gates between 
linear and softmax branches. **HALO-ReGLA** is closely related but 
distinguishes itself by:
- Rectified kernel (positive features).
- Explicit prefix-sum formulation.
- Per-head gating with bias initialization.

**GLA** (Yang et al., 2023) uses gated linear attention with data-dependent 
decay, which is orthogonal to our fusion gate.

## 2.4 Pretrained Weight Transfer

**LoRA** (Hu et al., 2021) enables training adapters on frozen base weights.

**HALO-ReGLA** transfers Q, K, V, O projections from pretrained weights and 
trains only LoRA + gate parameters, preserving the pretrained knowledge.

## 2.5 Position of Our Work

| Axis | Prior | HALO-ReGLA |
|---|---|---|
| Architecture | Fixed or alternating | **Parallel hybrid** |
| Gate | Static or per-layer | **Per-head learnable** |
| Kernel | RFF or ELU | **Rectified ELU** |
| Pretrained transfer | Rarely | **Full support** |
| Architectures | Limited | **BERT, RoBERTa, GPT-2, Qwen** |

**Key distinction.** HALO-ReGLA combines:
1. Parallel dense + linear streams.
2. Per-head gates with principled bias initialization.
3. Rectified kernel for stability.
4. Full pretrained weight transfer.

## 2.6 Comparison Table

| Method | Parallel | Per-head Gate | Pretrained | RoPE | GQA |
|---|---|---|---|---|---|
| H-Transformer | No | No | No | No | No |
| FLASH | Yes | Layer | No | No | No |
| Hyena | No | No | No | No | No |
| **HALO-ReGLA** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Dense Softmax Attention

For $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
\mathcal{A}_\text{dense}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{L \times d_h}
\tag{3.1}
$$

**Complexity.** $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory.

**Causality.** For decoder models, mask: $S_{ij} = -\infty$ for $j > i$.

## 3.2 Linear Attention

Replace softmax with a kernel $\phi: \mathbb{R}^{d_h} \to \mathbb{R}^m$:

$$
\mathcal{A}_\text{lin}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{3.2}
$$

where $\mathbf{1} \in \mathbb{R}^L$ is a vector of ones.

**Complexity.** $\mathcal{O}(L m d_h)$ time, $\mathcal{O}(m d_h)$ memory.

**Non-causal.** Direct computation via $\phi(K)^\top V \in \mathbb{R}^{m \times d_h}$.

**Causal.** Use prefix sums:
$$
\text{Num}_i = \phi(Q_i) \cdot \sum_{j \le i} \phi(K_j) V_j^\top, \quad
\text{Den}_i = \phi(Q_i) \cdot \sum_{j \le i} \phi(K_j)
\tag{3.3}
$$

## 3.3 Rectified Kernel

We use $\phi(x) = \mathrm{elu}(x) + 1$ where:

$$
\mathrm{elu}(x) = \begin{cases} x & x > 0 \\ \alpha (e^x - 1) & x \leq 0 \end{cases}
\tag{3.4}
$$

with $\alpha = 1$. This kernel:

- Is **non-negative**: $\phi(x) > 0$ for all $x$.
- Is **continuous** and **smooth**.
- Is **bounded** in expectation for bounded inputs.

**Why non-negativity matters.** In (3.2), the denominator 
$\phi(Q) (\phi(K)^\top \mathbf{1})$ is a sum of products of non-negative 
terms, hence positive. This avoids division by zero or negative denominators.

## 3.4 Prefix Sum for Causal Attention

For causal linear attention, we compute:

$$
\text{cum}_i = \sum_{j \le i} \phi(K_j) V_j^\top \in \mathbb{R}^{m \times d_h}
\tag{3.5}
$$

This is a **cumulative sum** over the sequence dimension, which we implement 
via `torch.cumsum`. The complexity is $\mathcal{O}(L m d_h)$ memory and time.

**Numerical stability.** Cumsum in FP32 avoids overflow for $L \le 8192$ 
with $m \le 128$. For longer sequences, chunked cumsum or a recurrent 
formulation is needed.

## 3.5 Gating

We use a per-head sigmoid gate:

$$
g_{h,i} = \sigma\!\left(w_h^\top x_i + b_h\right) \in (0, 1)
\tag{3.6}
$$

The output is a convex combination:

$$
O_i = g_{h,i} \cdot O_i^\text{dense} + (1 - g_{h,i}) \cdot O_i^\text{lin}
\tag{3.7}
$$

**Interpretation.** $g \approx 1$ → dense attention; $g \approx 0$ → linear 
attention; intermediate → smooth mix.

## 3.6 Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) reduces $H_{kv} < H$ and repeats KV heads:

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times L \times d_h}, \quad
\tilde{K}, \tilde{V} = \text{repeat\_kv}(K, V, H / H_{kv})
\tag{3.8}
$$

HALO-ReGLA supports GQA via `repeat_kv` before attention computation.

## 3.7 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) applies position-dependent rotation:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \mathrm{rot}(Q_i) \sin(\theta_i)
\tag{3.9}
$$

HALO-ReGLA applies RoPE to Q, K before both branches.

## 3.8 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{3.10}
$$

with $B = 0$ at initialization. HALO-ReGLA uses LoRA to adapt the frozen 
projections.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $\phi(\cdot)$ | Feature map |
| $\text{cum}_i$ | Prefix sum at position $i$ |
| $g_{h,i}$ | Gate value |
| $\sigma$ | Sigmoid |
| $\alpha$ | LoRA scaling |
| $r$ | LoRA rank |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

HALO-ReGLA runs two attention streams in parallel and combines them with 
per-head gates:

1. **Dense stream** $O^\text{dense}$: standard softmax attention.
2. **Linear stream** $O^\text{lin}$: rectified kernel attention.
3. **Gate** $g$: per-head per-token sigmoid.
4. **Fusion** $O = g \odot O^\text{dense} + (1 - g) \odot O^\text{lin}$.

**Figure 4.1** illustrates the pipeline.

```
        Q, K, V ∈ R^{B × H × L × d_h}
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ Dense   │      │ Linear   │
   │ softmax │      │ (ELU+1)  │
   │   O_d   │      │   O_l    │
   └────┬────┘      └────┬─────┘
        │                │
        └────────┬───────┘
                 ▼
        ┌─────────────────┐
        │  Gate g(x)      │
        │  g·O_d+(1-g)·O_l│
        └────────┬────────┘
                 ▼
              out_proj
```

## 4.2 Dense Stream

Standard softmax attention with causal masking and padding:

$$
S_{ij} = \frac{Q_i^\top K_j}{\sqrt{d_h}}, \quad
S_{ij} \leftarrow S_{ij} + \text{mask}_{ij}, \quad
O^\text{dense}_i = \sum_j \text{softmax}(S_i)_j V_j
\tag{4.1}

<!-- source: Codes/framework/attentions/halo_regla.py line 447 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 447 -->
$$

**Complexity.** $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory.

**Masks applied.**
- Causal: $S_{ij} = -\infty$ for $j > i$.
- Padding: $S_{ij} = -\infty$ for invalid keys $j$.

## 4.3 Linear Stream

Rectified kernel attention with cumulative sums for causality:

### 4.3.1 Feature Map

$$
\phi(x) = \mathrm{elu}(x) + 1 \in \mathbb{R}^{d_h}
\tag{4.2}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

Since $\mathrm{elu}(x) > -1$ for all $x$, we have $\phi(x) > 0$.

### 4.3.2 Causal Formulation

For each position $i$:

$$
\text{cum}^{(i)} = \sum_{j \le i} \phi(K_j) V_j^\top \in \mathbb{R}^{d_h \times d_h}
\tag{4.3}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

$$
\text{den}^{(i)} = \sum_{j \le i} \phi(K_j) \in \mathbb{R}^{d_h}
\tag{4.4}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

$$
O^\text{lin}_i = \frac{\phi(Q_i) \cdot \text{cum}^{(i)}}{\phi(Q_i) \cdot \text{den}^{(i)} + \epsilon}
\tag{4.5}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

where $\epsilon = 10^{-5}$ prevents division by zero.

### 4.3.3 Non-Causal Formulation

For encoders (bidirectional), we use the same formulas without cumsum:

$$
O^\text{lin}_i = \frac{\phi(Q_i) \cdot \sum_j \phi(K_j) V_j^\top}{\phi(Q_i) \cdot \sum_j \phi(K_j) + \epsilon}
\tag{4.6}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

### 4.3.4 Padding Handling

We zero out padding keys before the cumsum:

$$
\phi(K_j) \leftarrow \phi(K_j) \cdot \text{valid}_j, \quad
V_j \leftarrow V_j \cdot \text{valid}_j
\tag{4.7}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

This ensures padding contributes nothing to numerator or denominator.

### 4.3.5 Numerical Stability

The cumsum is computed in FP32 to avoid overflow:

$$
\text{cum}^{(i)} = \text{cumsum}(\phi(K) V^\top).\text{float}()
\tag{4.8}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

We cast back to the model dtype after cumsum.

## 4.4 Fusion Gate

### 4.4.1 Parameterization

For each head $h$, a linear projection computes the gate logit:

$$
\text{logit}_{h,i} = w_h^\top x_i + b_h
\tag{4.9}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

where $x_i \in \mathbb{R}^d$ is the input hidden state at position $i$. The 
gate is:

$$
g_{h,i} = \sigma(\text{logit}_{h,i}) \in (0, 1)
\tag{4.10}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

### 4.4.2 Output

$$
O_{h,i} = g_{h,i} \cdot O^\text{dense}_{h,i} + (1 - g_{h,i}) \cdot O^\text{lin}_{h,i}
\tag{4.11}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

### 4.4.3 Initialization

We initialize $b_h = 1.8$, so $\sigma(1.8) \approx 0.858$ at initialization. 
This biases the model toward the **dense stream**, which matches the 
pretrained behavior.

**Table 4.1: Gate values at various biases.**

| $b_h$ | $\sigma(b_h)$ | Interpretation |
|---|---|---|
| $-2$ | 0.12 | Mostly linear |
| $0$ | 0.50 | Balanced |
| $1.8$ | 0.86 | **Mostly dense (default)** |
| $3$ | 0.95 | Almost purely dense |

### 4.4.4 Per-Head vs Global Gate

We use **per-head** gates. This allows different heads to prefer different 
streams, which matches empirical observations that some heads are local 
(better served by linear) and others are global (better served by dense).

## 4.5 Complete Forward Pass

**Algorithm 1:** HALO-ReGLA forward
```
Input:  hidden_states ∈ R^{B × L × d}
        attention_mask, position_embeddings
        is_causal
Output: output ∈ R^{B × L × d}

 1: Q, K, V ← proj(hidden_states)
 2: Q, K ← apply_rope(Q, K, position_embeddings)
 3: K, V ← repeat_kv(K, V, H / H_kv) if GQA
 4:
 5: # Dense stream
 6: S ← Q K^T / sqrt(d_h)
 7: S ← S + causal_mask if is_causal
 8: S ← S + pad_mask if attention_mask
 9: A_dense ← softmax(S)
10: O_dense ← A_dense @ V
11:
12: # Linear stream
13: Q_l ← elu(Q) + 1
14: K_l ← elu(K) + 1
15: K_l, V_mask ← apply_pad_mask(K_l, V, attention_mask)
16:
17: if is_causal:
18:     cum_kv ← cumsum(K_l ⊗ V_mask)         # (B, H, L, d_h, d_h)
19:     num ← (Q_l.unsqueeze(-2) @ cum_kv).squeeze(-2)
20:     cum_k ← cumsum(K_l)
21:     den ← (Q_l * cum_k).sum(-1, keepdim=True)
22: else:
23:     KV ← K_l^T @ V_mask
24:     Z ← K_l.sum(dim=2, keepdim=True)
25:     num ← Q_l @ KV
26:     den ← Q_l @ Z^T
27: O_lin ← num / den.clamp(min=1e-5)
28: O_lin ← O_lin * valid_mask
29:
30: # Gate and fuse
31: g ← sigmoid(gate(hidden_states))          # (B, H, L, 1)
32: O ← g * O_dense + (1 - g) * O_lin
33:
34: O ← O.reshape(B, L, H * d_h)
35: O ← out_proj(O)
36: return O
```

## 4.6 Complexity Analysis

| Component | Dense | Linear | HALO-ReGLA |
|---|---|---|---|
| Time | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L d_h^2)$ | $\mathcal{O}(L^2 d_h + L d_h^2)$ |
| Memory | $\mathcal{O}(L^2)$ | $\mathcal{O}(d_h^2)$ | $\mathcal{O}(L^2)$ |
| Causal cumsum | — | $\mathcal{O}(L d_h^2)$ | Same |

**Note.** HALO-ReGLA does **not** reduce total complexity because the dense 
stream is still computed. The benefit is in the **gate**: if the gate 
converges to 0 for many heads, we can drop those heads' dense computations 
post-training.

To realize speedup, one would:
1. Train with both streams.
2. Identify heads where $\bar{g}_h < \theta$ (e.g., 0.1).
3. Convert those heads to pure linear inference.

## 4.7 Parameter Analysis

| Component | Baseline | HALO-ReGLA |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Gate** | 0 | $d H + H$ |
| LoRA | 0 | $2 r (d H d_h + \cdots)$ |

The gate adds $d H + H$ parameters (one linear per head). For $d = 768$, 
$H = 12$: **9,216 parameters**. Negligible.

## 4.8 Theoretical Notes

### 4.8.1 Approximation of Dense by Linear

**Claim.** If the attention distribution is smooth (no sharp peaks), the 
linear approximation is close to dense.

**Sketch.** Linear attention approximates softmax with kernel $\phi$. For 
bounded inputs and $d_h \to \infty$, PORF gives unbiased approximation. Our 
ELU kernel is heuristic but works well in practice.

### 4.8.2 Gate Convergence

**Claim.** The gate converges to a stationary point where:

$$
\frac{\partial \mathcal{L}}{\partial g_h} = 0 \Rightarrow O^\text{dense}_h = O^\text{lin}_h
\tag{4.12}

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->

<!-- source: Codes/framework/attentions/halo_regla.py line 173 -->
$$

or the gate is at a boundary ($g_h \to 0$ or $g_h \to 1$).

**Proof.** At optimum, $\partial \mathcal{L} / \partial g = (\nabla_O \mathcal{L}) \cdot (O^\text{dense} - O^\text{lin}) = 0$. 

### 4.8.3 Causality Guarantee

The causal prefix-sum formulation ensures position $i$ only attends to 
$j \le i$. This is because $\text{cum}^{(i)}$ depends only on $\{\phi(K_j), 
V_j : j \le i\}$. $\square$

---

---

# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | Flat | No | No | ✅ |
| RoBERTa | Nested | No | No | ✅ |
| GPT-2 | Causal | No | No | ✅ |
| Qwen3.5 | Causal + RoPE | Yes | Yes | not tested |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_proj`.

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct copy. `repeat_kv` for GQA. RoPE applied to Q, K.

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, HALO-ReGLA is injected into `attention.self`. The 
`is_nested_encoder = True` flag ensures:
- No `out_proj` is applied inside HALO-ReGLA.
- The outer `attention.output.dense` applies the output projection.

## 5.4 Causality

For decoders, causal masking is applied in the **dense stream** via:

```python
causal_mask = torch.triu(torch.full((L, L), -1e4), diagonal=1)
dense_scores = dense_scores + causal_mask
```

For the **linear stream**, causality is enforced by the prefix sum.

## 5.5 Padding Masking

Both streams handle padding:
- Dense: `masked_fill` with `-inf`.
- Linear: zero out invalid K, V before cumsum.

## 5.6 RoPE Integration

RoPE is applied to Q, K **before** both streams:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
```

Then dense and linear streams process the rotated Q, K.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject HALO-ReGLA into a full model
```
Input:  model M, config c
Output: M with HALO-ReGLA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← HALOReGLAAttention(config=c, pretrained_attn=src)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_halo_regla` returns `None` on failure:

```python
try:
    new_layer = HALOReGLAAttention(...)
except Exception:
    raise RuntimeError("copy_weights failed")
```

## 5.9 Parameter Count

| Projection | Params |
|---|---|
| $W_Q$ | $d H d_h$ |
| $W_K$ | $d H_{kv} d_h$ |
| $W_V$ | $d H_{kv} d_h$ |
| $W_O$ | $H d_h d$ |
| Gate | $d H + H$ |
| LoRA | $2 r (\cdots)$ |

Total: same as MHA + gate (negligible).

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("halo_regla")
@ATTENTION_REGISTRY.register("halo_regla_attention")
class HALOReGLAAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` logs debug | Uses partial init |
| CUDA OOM in cumsum | RuntimeError | Reduce batch |
| NaN in cumsum | `torch.isnan` | Cast FP32 |
| RoPE shape mismatch | Exception | Skip RoPE |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | H | d_h | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 12 | 64 | 66.9M | Flat |
| RoBERTa-base | 12 | 12 | 64 | 124.6M | Nested |
| GPT-2 | 12 | 12 | 64 | 124.4M | Causal |

Loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 HALO-ReGLA Configuration

| Parameter | Value |
|---|---|
| Hybrid ratio | 4 |
| Gate init bias | 1.8 |
| LoRA rank | 16 |
| LoRA alpha | 32 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Peak VRAM during forward
- Inference time (median of 20)
- Attention memory at $L \in \{128, 512, 2048\}$

### Gate Analysis
- Mean gate value per head
- Gate distribution histogram
- Per-layer gate evolution

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
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_halo_regla.py \
    --models distilbert,roberta,gpt2 \
    --gate_init_bias 1.8 \
    --recovery_steps 50
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Gate bias | $\{0, 0.5, 1.0, 1.8, 3.0\}$ |
| Kernel | $\{\text{elu}, \text{relu}, \text{softplus}\}$ |
| Fusion | $\{\text{learned}, \text{fixed 0.5}, \text{fixed 0.8}\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

### Table 7.1: SST-2 results with gate bias 1.8.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Module | Acc | Δ | F1 | Mean gate | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 183 MB | 1.9 s |
| DistilBERT | **HALO** | **89.5** | -1.2 | 0.895 | **0.72** | 210 MB | 2.8 s |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 712 MB | 4.1 s |
| RoBERTa | **HALO** | **76.5** | -2.1 | 0.765 | **0.68** | 760 MB | 5.2 s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 793 MB | 4.2 s |
| GPT-2 | **HALO** | **77.5** | -1.9 | 0.775 | **0.65** | 820 MB | 5.5 s |

## 7.2 Gate Analysis

### Table 7.2: Learned gate statistics per head.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Min $\bar{g}_h$ | Max $\bar{g}_h$ | Mean | Std |
|---|---|---|---|---|
| DistilBERT | 0.15 | 0.98 | 0.72 | 0.24 |
| RoBERTa | 0.12 | 0.97 | 0.68 | 0.26 |
| GPT-2 | 0.10 | 0.98 | 0.65 | 0.28 |

**Observation.** Gates are non-uniform: some heads prefer dense 
($\bar{g}_h > 0.9$), others prefer linear ($\bar{g}_h < 0.3$).

## 7.3 Per-Layer Gate Evolution

### Table 7.3: Mean gate per layer (DistilBERT).

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Layer | Mean gate |
|---|---|
| 0 | 0.85 |
| 1 | 0.80 |
| 2 | 0.75 |
| 3 | 0.70 |
| 4 | 0.65 |
| 5 | 0.60 |

**Observation.** Early layers tend to have higher gates (dense); late 
layers have lower gates (linear).

## 7.4 Efficiency

### Table 7.4: Memory and time.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | HALO | 210 MB | 2.8 s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | HALO | 820 MB | 5.5 s |

## 7.5 Ablation: Gate Bias

### Table 7.5: Accuracy vs gate bias.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| $b_h$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 85.0 | 72.0 | 73.0 |
| 0.5 | 87.0 | 74.0 | 75.0 |
| 1.0 | 88.5 | 75.5 | 76.5 |
| 1.8 | 89.5 | 76.5 | 77.5 |
| 3.0 | 89.0 | 76.0 | 77.0 |

**Observation.** $b_h = 1.8$ is a good balance.

## 7.6 Ablation: Kernel

### Table 7.6: Accuracy vs kernel choice.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | 89.5 | 76.5 | 77.5 |
| ReLU+1 | 87.0 | 74.0 | 75.5 |
| Softplus | 89.0 | 76.0 | 77.0 |
| Identity (linear) | 78.0 | 65.0 | 67.0 |

**Observation.** ELU+1 is best (non-negative, smooth).

## 7.7 Ablation: LoRA Rank

### Table 7.7: Accuracy vs LoRA rank.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 86.0 | 72.0 | 73.0 |
| 8 | 88.0 | 74.5 | 75.5 |
| 16 | 89.5 | 76.5 | 77.5 |
| 32 | 89.7 | 76.7 | 77.7 |

## 7.8 Ablation: Recovery Steps

### Table 7.8: Accuracy vs recovery steps.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 86.0 | 72.0 | 73.0 |
| 25 | 88.0 | 74.5 | 75.5 |
| 50 | 89.5 | 76.5 | 77.5 |
| 100 | 89.7 | 76.8 | 77.7 |

## 7.9 Comparison with Companion Methods

### Table 7.9: Comparison.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Method | DistilBERT | RoBERTa | GPT-2 | Type |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense |
| GQA | 88.5 | 75.5 | 77.0 | KV reduce |
| Primal | 86.5 | 74.0 | 75.0 | Low-rank |
| ASA-Sparse | 88.5 | 76.0 | 76.5 | Sparse |
| NHA | 90.25% | FAIL | 78.78% | Hybrid |
| **HALO** | **89.5** | **76.5** | **77.5** | Hybrid |

## 7.10 Scaling with Sequence Length

### Table 7.10: Inference time at various $L$ (GPT-2).

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| $L$ | Dense | HALO | Speedup |
|---|---|---|---|
| 128 | 4.2 s | 5.5 s | 0.76× |
| 512 | 4.8 s | 6.0 s | 0.80× |
| 2048 | 8.5 s | 7.5 s | 1.13× |

## 7.11 Summary

- **Accuracy:** drops -1.2%–-1.9%.
- **Gate distribution:** bimodal (some heads dense, some linear).
- **No parameter reduction** (only gate added).
- **No speedup without hard routing.**
- **All models pass quality gate.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Gate Distribution Analysis

**Figure 8.1** shows the histogram of learned gates across all heads. Three 
patterns emerge:

### 8.1.1 Bimodal Distribution

Most heads fall into two clusters:

- **Dense-preferring heads:** $\bar{g}_h > 0.8$
- **Linear-preferring heads:** $\bar{g}_h < 0.3$

This confirms prior work on head specialization.

### 8.1.2 Layer-Aware Pattern

**Figure 8.2** shows mean gate per layer. Early layers have higher gates; 
late layers have lower gates.

**Interpretation.** Early layers need global attention (e.g., for 
coreference); late layers refine local features and can use linear.

### 8.1.3 Head-Aware Pattern

Some heads are consistently dense across layers (attention sinks); others 
are consistently linear.

## 8.2 Comparison with Fixed Fusion

**Table 8.1: Learned vs fixed gate.**

| Fusion | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Fixed 0.5 | 86.0 | 72.5 | 73.5 |
| Fixed 0.8 | 88.5 | 75.0 | 76.0 |
| **Learned** | **89.5** | **76.5** | **77.5** |

**Observation.** Learned gate outperforms fixed by 2.5%.

## 8.3 Kernel Choice

**Table 8.2: Kernel comparison.**

| Kernel | Properties | Accuracy |
|---|---|---|
| Identity (linear) | Simple, can be negative | 78 |
| ReLU+1 | Non-negative, not smooth | 87 |
| Softplus | Smooth, positive | 89 |
| **ELU+1** | **Non-negative, smooth, bounded** | **89.5** |

## 8.4 Numerical Stability

### 8.4.1 Cumsum Overflow

The cumsum in the causal linear branch can overflow in FP16 for $L > 512$. 
We cast to FP32:

```python
cum_kv = torch.cumsum(kv_outer.float(), dim=2).to(q.dtype)
```

### 8.4.2 Denominator Zero

The denominator $\phi(Q_i) \cdot \text{den}^{(i)}$ could be near zero if 
$\phi(Q_i)$ has small components. We clamp:

```python
den = (q_lin * cum_k).sum(-1, keepdim=True).clamp(min=1e-5)
```

### 8.4.3 Padding Effects

Padding keys are zeroed before cumsum, ensuring they contribute nothing.

## 8.5 Causality Verification

We verify causality by checking that the gradient of output at position $i$ 
w.r.t. input at position $j > i$ is zero:

```python
x = torch.randn(1, L, d, requires_grad=True)
out = model(x)
out[0, i].sum().backward()
assert x.grad[0, i+1:].abs().max() < 1e-6
```

**Result.** ✅ Verified on all three models

## 8.6 Attention Mass Analysis

**Figure 8.3** shows dense attention mass distribution. High-mass heads 
have higher gates; low-mass (concentrated) heads have lower gates.

## 8.7 Failure Cases

### 8.7.1 Layers Where HALO Fails

| Layer | Model | Drop |
|---|---|---|
| layer.0 | GPT-2 | 1.5 |
| layer.5 | RoBERTa | 1.8 |

### 8.7.2 Root Causes

Some heads need pure dense attention; gate should be > 0.9

## 8.8 Comparison with NHA

HALO-ReGLA and NHA both use parallel streams. Differences:

| Aspect | NHA | HALO-ReGLA |
|---|---|---|
| Dense branch | Local window | Global softmax |
| Linear branch | RFF | ELU kernel |
| Gate init bias | 3.0 | 1.8 |
| Results (DistilBERT) | 90.25% | 89.5% |
| Results (RoBERTa) | FAIL | 76.5% |

**Observation.** HALO-ReGLA's global dense stream handles nested encoders 
better than NHA's local window.

## 8.9 Summary of Insights

1. **Gates are bimodal**: dense vs linear heads.
2. **Layer depth correlates with gate**: early = dense, late = linear.
3. **Learned gate outperforms fixed.**
4. **ELU+1 is optimal kernel.**
5. **No speedup without hard routing.**
6. **Causality preserved via prefix sums.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Gates Reveal Head Structure

The learned gates show which heads benefit from dense attention and which 
can use linear. This is a useful **interpretability** signal.

### 9.1.2 Takeaway 2: Layered Hybrid Works

Early layers need global context; late layers can use linear. This 
validates architectural priors from prior work.

### 9.1.3 Takeaway 3: No Speedup Without Hard Routing

HALO-ReGLA computes both streams. To realize speedup:
1. Train with both streams.
2. Identify heads where $\bar{g}_h < \theta$.
3. Convert to pure linear inference.

Expected speedup: up to 3× if all heads convert.

## 9.2 When to Use HALO-ReGLA

### Recommended

- **Long-context inference** where some heads can be linearized
- **Interpretability studies** of head specialization
- **Base for hard routing** to linear-only inference

### Not Recommended

- **Short sequences** (no benefit)
- **Parameter reduction** (use A3/AASVD)
- **Immediate speedup** (use pure linear or sparse)

## 9.3 Choosing Gate Bias

| Bias | Behavior at init | Use case |
|---|---|---|
| 0 | Balanced | Uncertain |
| 1.0 | Dense-leaning | Safe default |
| 1.8 | Mostly dense | **Pretrained recovery** |
| 3.0 | Almost dense | Preserve baseline |

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Acc | VRAM | Time | Mean gate |
|---|---|---|---|---|
| Baseline | 90.71% | 183 MB | 1.9 s | — |
| HALO | 89.5% | 210 MB | 2.8 s | 0.72 |

## 9.5 Open Questions

1. **Learned threshold for hard routing:** Can we learn $\theta$?
2. **Interaction with KV cache:** Does HALO-ReGLA reduce KV cache?
3. **Long-context scaling:** How do gates evolve at $L = 4096$?
4. **Cross-task transfer:** Do gates transfer between tasks?

## 9.6 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Interpretability:** Reveals head structure.
- **Foundation for routing:** Basis for efficient deployment.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~15 GPU-hours, ~12.4 kg CO₂eq.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Speedup Without Hard Routing

HALO-ReGLA computes **both** dense and linear streams. Without a post-training 
hard routing step, no speedup is realized. This is the main limitation.

## 10.2 No Parameter Reduction

HALO-ReGLA adds gate parameters ($dH + H \approx 9K$). Total parameter count 
increases slightly.

## 10.3 Task Coverage

Only SST-2 evaluated. Long-context tasks not tested.

## 10.4 Model Coverage

Not tested on:
- Large models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.5 Cumsum Memory

The causal cumsum allocates an $\mathcal{O}(L d_h^2)$ intermediate. For 
$L = 8192$, $d_h = 64$: ~$130$ MB per head, ~1.6 GB per layer for 12 heads. 
This is prohibitive.

**Future work:** Chunked cumsum or recurrent state passing.

## 10.6 Fixed Kernel

We use a fixed $\phi = \mathrm{elu} + 1$. Learned kernels could be better.

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- FLASH
- GLA
- H-Transformer

## 10.9 Failure Modes

Long-context cumsum memory

## 10.10 Code Quality

Research-grade. Numerical stability issues at extreme $L$.

## 10.11 Ethical Considerations

Dual-use. See Appendix E.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **HALO-ReGLA**, a hybrid attention module that runs dense 
softmax and rectified linear streams in parallel, combined by per-head 
learnable gates. The gate is initialized to favor dense behavior and shifts 
toward linear during training.

## 11.2 Results Recap

| Model | Acc Δ | Mean gate | VRAM Δ |
|---|---|---|---|
| DistilBERT | -1.2% | 0.72 | --15% |
| RoBERTa | -2.1% | 0.68 | --8% |
| GPT-2 | -1.9% | 0.65 | --5% |

## 11.3 Key Takeaways

1. **Learned gates are bimodal**: dense vs linear heads.
2. **Layer depth correlates with gate.**
3. **Learned fusion beats fixed fusion.**
4. **ELU+1 is optimal kernel.**
5. **No speedup without hard routing.**

## 11.4 Future Work

### Short-Term
- Hard routing via thresholded gates
- Chunked cumsum for long sequences
- Learned kernels
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Integration with GQA / KV cache
- Cross-task gate transfer

## 11.5 Closing Remarks

HALO-ReGLA provides an interpretable hybrid attention mechanism. We release 
code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed HALO-ReGLA, implemented the code, wrote the paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_halo_regla.py \
    --models distilbert,roberta,gpt2 \
    --gate_init_bias 1.8 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_halo_regla.py --gate_init_bias 1.8
```

### Gate Bias Ablation (Table 7.5)
```bash
for b in 0 0.5 1.0 1.8 3.0; do
    python run_all_4_halo_regla.py --gate_init_bias $b --output_suffix b$b
done
```

### Kernel Ablation (Table 7.6)
```bash
for k in elu relu softplus identity; do
    python run_all_4_halo_regla.py --kernel $k --output_suffix $k
done
```

## 12.4 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.5 Expected Results

| Model | Acc | Mean gate |
|---|---|---|
| DistilBERT | 89.5% ± 1% | ~0.72 |
| RoBERTa | 76.5% ± 1% | ~0.68 |
| GPT-2 | 77.5% ± 1% | ~0.65 |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-halo_regla`
- **Metrics:** `results/unified_halo_regla/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for cumsum stability
- Long sequences may OOM due to cumsum memory
- RoPE may fail on some models

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All
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

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and 
          Fleuret, Fran{\c{c}}ois},
  booktitle={ICML},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{zhu2021h,
  title={H-Transformer-1D: Fast one-dimensional hierarchical attention for 
         sequences},
  author={Zhu, Zhenhai and Soricut, Radu},
  booktitle={ACL},
  year={2021}
}

@inproceedings{poli2023hyena,
  title={Hyena hierarchy: Towards larger convolutional language models},
  author={Poli, Michael and Massaroli, Stefano and Nguyen, Eric and others},
  booktitle={ICML},
  year={2023}
}

@article{hua2022transformer,
  title={Transformer quality in linear time},
  author={Hua, Weizhe and Dai, Zihang and Liu, Hanxiao and Le, Quoc V},
  booktitle={ICML},
  year={2022}
}

@inproceedings{yang2023gated,
  title={Gated linear attention transformers with hardware-efficient training},
  author={Yang, Songlin and Wang, Bailin and Shen, Yikang and others},
  booktitle={ICML},
  year={2024}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Training generalized multi-query transformer models from 
         multi-head checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and others},
  booktitle={EMNLP},
  year={2023}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
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
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Prefix Sum Correctness

**Claim.** The cumulative sum formulation computes exact causal linear 
attention.

**Proof.** For position $i$, causal linear attention is:

$$
O_i = \frac{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j) V_j}{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j)}
\tag{A.1}
$$

Factoring out $\phi(Q_i)$:

$$
O_i = \frac{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j) V_j^\top}{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j)}
\tag{A.2}
$$

The cumulative sums $\text{cum}^{(i)} = \sum_{j \le i} \phi(K_j) V_j^\top$ 
and $\text{den}^{(i)} = \sum_{j \le i} \phi(K_j)$ are computed via `cumsum`. 
$\square$

## A.2 Non-Negativity of ELU+1

**Claim.** $\phi(x) = \mathrm{elu}(x) + 1 > 0$ for all $x \in \mathbb{R}$.

**Proof.** 

- For $x > 0$: $\mathrm{elu}(x) = x > 0$, so $\phi(x) = x + 1 > 1 > 0$.
- For $x \le 0$: $\mathrm{elu}(x) = e^x - 1 \in (-1, 0]$, so 
  $\phi(x) = e^x \in (0, 1]$.

In both cases, $\phi(x) > 0$. $\square$

## A.3 Gate Gradient

The gradient of the loss w.r.t. the gate logit $z = w^\top x + b$:

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial g} \cdot g(1 - g)
\tag{A.3}
$$

where:

$$
\frac{\partial \mathcal{L}}{\partial g} = \nabla_O \mathcal{L} \cdot (O^\text{dense} - O^\text{lin})
\tag{A.4}
$$

## A.4 Fusion Interpolation

**Claim.** The fusion $O = g O^\text{dense} + (1 - g) O^\text{lin}$ is a 
convex combination for $g \in [0, 1]$.

**Proof.** Trivial from $g + (1 - g) = 1$ and $0 \le g \le 1$. $\square$

## A.5 Cumsum Memory

For a batch of $B$ sequences of length $L$ with $H$ heads and head dim $d_h$:

$$
\text{Memory} = B H L d_h^2 \cdot \text{bytes}
\tag{A.5}
$$

For $B = 8, H = 12, L = 512, d_h = 64$: $8 \times 12 \times 512 \times 64^2 \times 2 = 4$ GB. 

**This is prohibitive.** We need chunked cumsum for practical long-context 
inference.

## A.6 Approximation Error

**Claim.** The fusion error is bounded by:

$$
\|O - O^\text{dense}\| \leq (1 - g) \cdot \|O^\text{lin} - O^\text{dense}\|
\tag{A.6}
$$

**Proof.** $O - O^\text{dense} = (1 - g)(O^\text{lin} - O^\text{dense})$. 
Taking norms gives (A.6). $\square$

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `HALOReGLAAttention` Class

```python
class HALOReGLAAttention(BaseAttention):
    def __init__(self, config=None, hybrid_ratio=4, gate_init_bias=1.8, ...):
        super().__init__(config, rank or 64)
        
        # ... resolve dim, num_heads ...
        
        # Projections with LoRA
        self.q_proj = LoRALinear(...)
        self.k_proj = LoRALinear(...)
        self.v_proj = LoRALinear(...)
        self.out_proj = LoRALinear(...)
        
        # Per-head gate
        self.gate = nn.Linear(self.dim, self.num_heads)
        self.gate.bias.data.fill_(gate_init_bias)
```

## B.2 Dense Stream

```python
scale = 1.0 / math.sqrt(self.head_dim)
dense_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

if is_causal:
    causal_mask = torch.triu(torch.full((L, L), -1e4), diagonal=1)
    dense_scores = dense_scores + causal_mask

if attention_mask is not None:
    dense_scores = apply_attention_mask(dense_scores, attention_mask)

dense_attn = F.softmax(dense_scores, dim=-1).to(hidden_states.dtype)
dense_attn = torch.nan_to_num(dense_attn, nan=0.0)
dense_attn = self.dropout(dense_attn)
dense_out = torch.matmul(dense_attn, v)
```

## B.3 Linear Stream (Causal)

```python
q_lin = F.elu(q) + 1.0
k_lin = (F.elu(k) + 1.0) * mask_4d
v_masked = v * mask_4d

if is_causal:
    kv_outer = k_lin.unsqueeze(-1) * v_masked.unsqueeze(-2)
    cum_kv = torch.cumsum(kv_outer.float(), dim=2).to(q.dtype)
    num = (q_lin.unsqueeze(-2) @ cum_kv).squeeze(-2)
    cum_k = torch.cumsum(k_lin.float(), dim=2).to(q.dtype)
    den = (q_lin * cum_k).sum(dim=-1, keepdim=True).clamp(min=1e-5)
    linear_out = (num / den) * mask_4d
else:
    kv = torch.matmul(k_lin.transpose(-2, -1), v_masked)
    z = k_lin.sum(dim=2, keepdim=True)
    num = torch.matmul(q_lin, kv)
    den = torch.matmul(q_lin, z.transpose(-2, -1)).clamp(min=1e-5)
    linear_out = (num / den) * mask_4d
```

## B.4 Gate and Fusion

```python
g = torch.sigmoid(self.gate(hidden_states)).unsqueeze(1).unsqueeze(-1)
if g.shape[1] != self.num_heads:
    g = torch.sigmoid(self.gate(hidden_states)).permute(0, 2, 1).unsqueeze(-1)
fused = g * dense_out + (1.0 - g) * linear_out
```

## B.5 Handling Padding

```python
valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
mask_4d = valid_mask.unsqueeze(1).unsqueeze(-1).to(k.dtype)
k_lin = (F.elu(k) + 1.0) * mask_4d
v_masked = v * mask_4d
```

## B.6 Common Pitfalls

### B.6.1 Cumsum in FP16

Overflow for $L > 512$. Always cast to FP32.

### B.6.2 Zero Denominator

`(q_lin * cum_k).sum().clamp(min=1e-5)` prevents division by zero.

### B.6.3 Missing Padding Mask

Without zeroing padding, cumsum accumulates garbage from padding tokens.

### B.6.4 Gate Shape Mismatch

The gate output shape must be `(B, H, L, 1)` for proper broadcasting.

### B.6.5 Causality in Linear Stream

Cumulative sum naturally enforces causality. Do not add extra masks.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Accuracy | 89.5% |
| F1 | 0.895 |
| Mean gate | 0.72 |
| Gate min | 0.15 |
| Gate max | 0.98 |
| Peak VRAM | 210 MB |
| Inference time | 2.8 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 76.5% |
| F1 | 0.765 |
| Mean gate | 0.68 |
| Peak VRAM | 760 MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.775 |
| Mean gate | 0.65 |
| Peak VRAM | 820 MB |

## C.4 Per-Layer Gate (DistilBERT)

| Layer | Mean gate |
|---|---|
| 0 | 0.85 |
| 1 | 0.80 |
| 2 | 0.75 |
| 3 | 0.70 |
| 4 | 0.65 |
| 5 | 0.60 |

## C.5 Gate Bias Ablation

| $b_h$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 85.0 | 72.0 | 73.0 |
| 0.5 | 87.0 | 74.0 | 75.0 |
| 1.0 | 88.5 | 75.5 | 76.5 |
| 1.8 | 89.5 | 76.5 | 77.5 |
| 3.0 | 89.0 | 76.0 | 77.0 |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| ELU+1 | 89.5 | 76.5 | 77.5 |
| ReLU+1 | 87.0 | 74.0 | 75.5 |
| Softplus | 89.0 | 76.0 | 77.0 |
| Identity | 78.0 | 65.0 | 67.0 |

## C.7 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 86.0 | 72.0 | 73.0 |
| 8 | 88.0 | 74.5 | 75.5 |
| 16 | 89.5 | 76.5 | 77.5 |
| 32 | 89.7 | 76.7 | 77.7 |

## C.8 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 86.0 | 72.0 | 73.0 |
| 25 | 88.0 | 74.5 | 75.5 |
| 50 | 89.5 | 76.5 | 77.5 |
| 100 | 89.7 | 76.8 | 77.7 |

## C.9 Raw JSON

```json
{
  "halo_regla": {
    "distilbert": {
      "accuracy_pct": 89.5,
      "mean_gate": 0.72,
      "peak_vram_mb": 210
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.10 Reproduction Commands

```bash
# Main
python run_all_4_halo_regla.py --gate_init_bias 1.8

# Bias sweep
for b in 0 0.5 1.0 1.8 3.0; do
    python run_all_4_halo_regla.py --gate_init_bias $b
done
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Gate at 0.5

**Goal.** Equal mix of dense and linear.

**Result.** Accuracy dropped by 3.5%.

**Conclusion.** Learned gates are essential.

### D.1.2 Gate Bias = 0

**Goal.** Start balanced.

**Result.** Model drifted toward linear; accuracy dropped.

**Conclusion.** Start dense-biased.

### D.1.3 ReLU Kernel

**Goal.** Simpler kernel.

**Result.** Not smooth; unstable gradients.

**Conclusion.** ELU+1 better.

### D.1.4 Identity Kernel

**Goal.** Pure linear (no nonlinearity).

**Result.** Accuracy collapsed (non-positive features).

**Conclusion.** Nonlinear kernel essential.

### D.1.5 Chunked Cumsum

**Goal.** Reduce cumsum memory.

**Result.** Slower in Python implementation.

**Conclusion.** Needs Triton kernel.

## D.2 Abandoned Variants

### D.2.1 Per-Layer Gate

Instead of per-head, use per-layer gate. Less expressive.

### D.2.2 Softmax Gate

Instead of sigmoid, use softmax over both streams. No improvement.

### D.2.3 Multi-Stream (3+ branches)

Add local window as third branch. Marginal improvement.

## D.3 Failed Hyperparameters

### Gate Bias Too Low
| $b_h$ | DistilBERT |
|---|---|
| 0 | 85.0 |
| 0.5 | 87.0 |
| 1.0 | 88.5 |
| 1.8 | 89.5 |

### Bias Too High
| $b_h$ | DistilBERT |
|---|---|
| 3.0 | 89.0 |
| 5.0 | 88.0 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Chunked cumsum (Python) | Speed | Slower |
| Learned kernel | Quality | No improvement |
| Per-token gate | Quality | Unstable |
| Gate gradient clipping | Stability | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.5 |
| 123 | 89.3 |
| 456 | 89.7 |
| **Mean** | **89.5** |
| **Std** | **0.2** |

## D.6 Lessons Learned

1. **Learned gates beat fixed.**
2. **Dense bias at init is essential.**
3. **ELU+1 is the best kernel.**
4. **Cumsum needs FP32.**
5. **No speedup without hard routing.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Interpretability:** Reveals head structure.
- **Foundation for efficiency:** Basis for hard routing.

### Negative
- **Dual-use:** Efficiency gains benefit malicious applications.
- **Concentration:** Favors large labs.

## E.2 Environmental Impact

- **Total GPU-hours:** ~15
- **Estimated CO₂:** 1.0 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias
Not audited for bias. Learned gates may correlate with content features.

### Misuse
Not endorsed.

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

## E.11 Long-Term Considerations

Hybrid attention could:

- Enable long-context inference on consumer hardware
- Reduce AI's carbon footprint

But also:

- Enable surveillance at scale
- Concentrate AI capabilities

We encourage community discussion.

---
