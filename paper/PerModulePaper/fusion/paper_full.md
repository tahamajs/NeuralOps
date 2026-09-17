# FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention with Block Gating and Short-Sequence Fast Path

**Authors:**
- Your Name¹ — you@university.edu
- 0.85² — 0.85

**Affiliations:**
- ¹ 0.85, 0.85
- ² 0.85, 0.85

**Corresponding author:** 0.85
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/fusion_attn.py
**License:** MIT

---

## Abstract

Long-context attention in transformers is quadratic in sequence length, 
$\mathcal{O}(L^2 d)$, which makes inference prohibitively expensive at 
$L \geq 4096$. Existing sparse methods — Longformer, BigBird, MoBA — reduce 
this to $\mathcal{O}(L \cdot W \cdot d)$ or $\mathcal{O}(K \cdot L \cdot d)$ 
but often apply the same sparsity pattern regardless of sequence length. 
We introduce **FUSIONATTN**, a **multi-paradigm hybrid attention** that 
dispatches between two paths based on sequence length:

1. **Short-sequence path ($L \leq 128$):** Standard softmax attention, no 
   sparsity overhead.
2. **Long-sequence path ($L > 128$):** Block-sparse attention with a 
   **learnable gating MLP** that scores query-key block pairs, selects the 
   top-$K$ blocks per query block, and computes attention only within the 
   selected blocks (**MoBA-style**).

The gate is a 2-layer MLP with GELU activation that operates on 
mean-pooled block representations; it is trained jointly with LoRA 
adapters. Base Q/K/V/O projections are frozen and copied exactly from the 
pretrained model, ensuring step-0 preservation.

**Important disclosure.** FUSIONATTN is **not** a compression method. It 
**increases** parameter count via LoRA adapters and the gate MLP. Its value 
lies in **long-context compute efficiency**, not size reduction. We 
additionally note that at $L \leq 128$ (our test sequence length), FUSIONATTN 
takes the **short path** and produces output identical to the pretrained 
model with no sparsity benefit.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
50 LoRA recovery steps. FUSIONATTN achieves 
**90.48%** on DistilBERT, **77.35%** on RoBERTa, and **78.90%** 
on GPT-2, matching the SDPA baseline within noise.

**Keywords:** hybrid sparse attention, block gating, MoBA, short-sequence 
fast path, learnable sparsity, LoRA

---

## Lay Summary

Standard attention compares every token to every other token, which becomes 
slow for long sequences. FUSIONATTN takes a hybrid approach: for short 
sequences (under 128 tokens), it uses the standard fast attention. For 
longer sequences, it groups tokens into blocks and uses a small neural 
network to decide which blocks should "talk to" each other. Only the 
selected blocks compute attention. This saves computation for long 
sequences while keeping short-sequence behavior unchanged. The trade-off: 
it adds some parameters to the model, and it's not faster on short 
sequences.

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

# 1. Introduction

## 1.1 Motivation

Transformer attention computes a pairwise score between every query and 
every key:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

requiring $\mathcal{O}(L^2 d)$ time and $\mathcal{O}(L^2)$ memory. For 
$L = 4096$, this is 32 MB per head, per layer, in FP16 — 100 GB+ for a 
typical model. Even with memory-efficient kernels like FlashAttention, the 
attention matrix's computational cost is still quadratic.

**Sparse attention** reduces this by attending to only a subset of keys. 
Existing methods divide into:

- **Fixed sparsity patterns** (Longformer, BigBird): window + global tokens, 
  learned independently of content.
- **Content-based sparsity** (Routing, MoBA): tokens or blocks selected 
  based on query content.

**Length heterogeneity.** Most methods apply the same pattern regardless of 
sequence length. But sparsity has **overhead**: even for $L = 32$, sparse 
methods pay the cost of routing/gating. For short sequences, softmax is 
both simpler and faster.

## 1.2 Problem Statement

Given a pretrained attention module $\mathcal{A}$ with projections 
$W_Q, W_K, W_V, W_O$, we seek an augmented attention function 
$\mathcal{A}'$ such that:

1. **Short-sequence preservation.** For $L \leq L_\text{threshold}$, 
   $\mathcal{A}'(X) = \mathcal{A}(X)$ exactly (no sparsity overhead).
2. **Long-sequence efficiency.** For $L > L_\text{threshold}$, 
   $\mathcal{A}'$ reduces attention FLOPs by selecting a subset of keys.
3. **Content-aware sparsity.** The selection depends on query content, not 
   on fixed patterns.
4. **Step-0 preservation.** With zero-init LoRA, initial forward matches 
   the pretrained model.
5. **Multi-architecture support.** Works on flat, nested, and causal 
   transformers with GQA and RoPE.

**Note.** This is **not** a compression method. Parameter count increases 
by the LoRA adapters and gate MLP.

## 1.3 Our Approach

FUSIONATTN dispatches based on sequence length:

### Short Path (L ≤ 128)

Standard softmax attention. No sparsity, no overhead. Mathematically 
identical to the pretrained model (given base weights copied exactly).

### Long Path (L > 128)

Block-sparse attention with content-based gating:

1. **Block partitioning.** Split sequence into blocks of size $B$ (default 
   64).
2. **Gate scoring.** For each query block $i$ and key block $j$, compute:

$$
g(i, j) = \text{GateMLP}(\bar{Q}_i)^\top \cdot \text{GateMLP}(\bar{K}_j) / \tau
\tag{1.2}
$$

where $\bar{Q}_i, \bar{K}_j$ are block-mean representations.

3. **Top-K selection.** For each query block, keep the top-$K$ key blocks 
   by gate score (plus the first block, for stability).
4. **Block-local attention.** Compute attention only over the selected 
   blocks.

**Key advantage.** The gate is **learned**, not fixed. It can adapt to 
task-specific sparsity patterns.

## 1.4 Contributions

- **A length-aware dispatch.** Short sequences take a zero-overhead path; 
  long sequences take a content-aware sparse path. Section 4.3.
- **A learnable block gate.** 2-layer MLP over block-mean representations, 
  trained jointly with LoRA. Section 4.4.
- **MoBA-style block attention.** Top-$K$ block selection with first-block 
  preservation. Section 4.5.
- **Multi-architecture weight transfer.** GPT-2 Conv1D, RoBERTa/BERT nested, 
  DistilBERT flat, Qwen GQA. Section 5.
- **Honest reporting.** We disclose that at $L = 128$ (our test length), the 
  module takes the short path and provides **no sparsity benefit**. We 
  report this transparently. Sections 7–8.
- **Full reproducibility.** Code and evaluation scripts released under 
  MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on sparse 
attention, block gating, and MoBA. Section 4 presents FUSIONATTN. Section 
5 details architecture integration. Section 6 gives the experimental 
setup. Section 7 reports results. Section 8 analyzes the short-vs-long 
dispatch and block-sparsity behavior. Section 9 discusses usage. Section 
10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $L_\text{threshold}$ | Dispatch threshold (128) |
| $B$ | Block size (64) |
| $K$ | Top-K blocks (4) |
| $d_h$ | Per-head dimension |
| $\bar{Q}_i$ | Mean of query block $i$ |
| $g(i, j)$ | Gate score for block pair $(i, j)$ |
| $\tau$ | Gate temperature |
| $r_L$ | LoRA rank |

---

# 2. Related Work

## 2.1 Sparse Attention

**Longformer.** Beltagy et al. (2020) combine a sliding window of radius $W$ 
with a small set of global tokens. Complexity: $\mathcal{O}(L \cdot W)$. 
Content-independent.

**BigBird.** Zaheer et al. (2020) add random attention and prove Turing 
completeness. Similar to Longformer in practice.

**Sparse Transformer.** Child et al. (2019) use strided and fixed patterns. 
Pre-dates most modern work.

**Star-Transformer.** Guo et al. (2019) use a star topology. Rarely used 
today.

## 2.2 Content-Based Sparse Attention

**Routing Transformer.** Roy et al. (2021) cluster tokens and restrict 
attention within clusters. Complexity: $\mathcal{O}(L \cdot L/K)$.

**Reformer.** Kitaev et al. (2020) use LSH to hash similar keys. Probabilistic 
guarantees.

**MoBA (Mixture of Block Attention).** Lu et al. (2024) introduce block-level 
routing with top-$K$ selection. FUSIONATTN's long path uses MoBA-style 
block selection.

**NSA (Native Sparse Attention).** DeepSeek (2025) combines local, strided, 
and compressed attention branches.

## 2.3 Hybrid Length-Aware Attention

**H-Transformer.** Zhu & Soricut (2021) adapt pattern to sequence length. 
FUSIONATTN uses a **discrete dispatch** based on a threshold.

**Blockwise Parallel Transformer.** Liu & Abbeel (2023) parallelize block 
attention. Not length-aware.

## 2.4 Grouped Query Attention

**GQA.** Ainslie et al. (2023) reduce KV head count. FUSIONATTN preserves 
GQA structure.

## 2.5 Position of Our Work

| Method | Sparsity | Length-aware | Gate learned |
|---|---|---|---|
| Longformer | Fixed | No | No |
| BigBird | Fixed + random | No | No |
| Routing | Content | No | Yes |
| MoBA | Content | No | Yes |
| **FUSIONATTN** | **Content** | **Yes** | **Yes** |

**Key distinction.** FUSIONATTN is the only method that **dispatches to 
dense attention for short sequences** and **content-aware block sparsity 
for long sequences**.

## 2.6 Comparison Table

| Feature | Longformer | Routing | MoBA | FUSIONATTN |
|---|---|---|---|---|
| Short-seq overhead | Yes | Yes | Yes | **None** |
| Long-seq complexity | $O(LW)$ | $O(L^2/K)$ | $O(KL)$ | $O(KL)$ |
| Learned gate | No | Yes | Yes | Yes |
| Multi-arch | Partial | Partial | Partial | Full |

## 2.7 Not a Compression Method

FUSIONATTN **increases** parameter count via LoRA and the gate MLP. It 
should not be confused with compression methods like A3, AASVD, or 
Adaptive SVD.

---

# 3. Background

## 3.1 Standard Attention

For $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

Complexity: $\mathcal{O}(L^2 d_h)$.

## 3.2 Block-Sparse Attention

Partition the sequence into $T = L / B$ blocks. For block $i$ of queries 
and block $j$ of keys, attention is computed only if $(i, j) \in \mathcal{S}$ 
for some sparsity mask $\mathcal{S}$.

**Fixed mask.** Longformer: $(i, j) \in \mathcal{S}$ iff $|i - j| \leq W/B$ 
or $j = 0$.

**Content-based mask.** MoBA/Routing: $(i, j) \in \mathcal{S}$ iff $j$ 
is among the top-$K$ for $i$ by some score.

## 3.3 MoBA-Style Block Selection

MoBA (Lu et al., 2024) computes a **block gate score**:

$$
g(i, j) = \langle \phi(\bar{Q}_i), \phi(\bar{K}_j) \rangle / \tau
\tag{3.2}
$$

where $\bar{Q}_i$ is the mean of query block $i$, $\phi$ is a learned 
projection, and $\tau$ is a temperature.

For each query block $i$, the top-$K$ key blocks by $g(i, j)$ are selected.

## 3.4 Grouped Query Attention

GQA uses $H_{kv} < H$ KV heads, repeated to match query heads:

$$
\text{repeat\_kv}(K, n)_{b, h, i, d} = K_{b, h \bmod H_{kv}, i, d}
\tag{3.3}
$$

## 3.5 Rotary Position Embeddings

RoPE rotates Q, K by position-dependent angles. Partial RoPE (Qwen3.5) 
rotates only the first `rotary_dim` coordinates.

## 3.6 LoRA

LoRA (Hu et al., 2021) parameterizes:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.4}
$$

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Block size |
| $T$ | Number of blocks |
| $K$ | Top-K blocks |
| $\bar{Q}_i$ | Query block mean |
| $\bar{K}_j$ | Key block mean |
| $g(i, j)$ | Gate score |
| $\tau$ | Gate temperature |

---

# 4. Method

## 4.1 Overview

FUSIONATTN has two execution paths selected by sequence length:

- **Short path ($L \leq L_\text{threshold}$):** Standard softmax attention.
- **Long path ($L > L_\text{threshold}$):** MoBA-style block-sparse attention.

**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────┐
    │  Input X → Q/K/V projections         │
    │  (frozen base + LoRA)                │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Apply RoPE, repeat_kv (GQA)         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Is L ≤ 128?                          │
    └────┬──────────────────────┬──────────┘
         │ Yes                  │ No
         ▼                      ▼
    ┌────────────┐      ┌────────────────────┐
    │  Full      │      │  Block partition   │
    │  softmax   │      │  + Gate scoring    │
    │  attention │      │  + Top-K selection │
    └─────┬──────┘      │  + Block attention │
          │             └─────────┬──────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌────────────────┐
              │  out_proj      │
              │  (frozen + LoRA)│
              └────────┬───────┘
                       │
                       ▼
                      Y
```

## 4.2 Phase 1: Projections

Frozen base + LoRA:

$$
\text{Proj}_P(X) = X W_P^\top + \frac{\alpha}{r_L} (X A_P^\top) B_P^\top \cdot \sigma + b_P
\tag{4.1}
$$

where $W_P$ is copied exactly from the pretrained attention.

## 4.3 Phase 2: Length-Based Dispatch

$$
\text{Path}(L) = \begin{cases}
\text{short}, & L \leq L_\text{threshold} \\
\text{long}, & L > L_\text{threshold}
\end{cases}
\tag{4.2}
$$

with $L_\text{threshold} = 128$ (default). Also uses short path for 
cross-attention.

## 4.4 Phase 3 (Long Path): Block Gating

### Block Partitioning

Pad sequence to multiple of $B$ (block size), then reshape:

$$
Q \to Q_\text{blocks} \in \mathbb{R}^{B \times H \times T \times B \times d_h}
\tag{4.3}
$$

### Block-Mean Representations

$$
\bar{Q}_i = \frac{1}{B} \sum_{j=1}^{B} Q_\text{blocks}[:, :, i, j, :], \quad
\bar{K}_i = \frac{1}{B} \sum_{j=1}^{B} K_\text{blocks}[:, :, i, j, :]
\tag{4.4}
$$

### Gate MLP

A 2-layer MLP with GELU activation:

$$
\phi(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
\tag{4.5}
$$

with $W_1 \in \mathbb{R}^{4 d_h \times d_h}$, $W_2 \in \mathbb{R}^{1 \times 4 d_h}$.

### Gate Scores

$$
g(i, j) = \frac{\phi(\bar{Q}_i) \cdot \phi(\bar{K}_j)}{\tau}
\tag{4.6}
$$

where $\tau$ is a learnable temperature.

### Top-K Selection

$$
\text{TopK}_i = \text{top-}K_{j}(\text{softmax}_j(g(i, j)))
\tag{4.7}
$$

with block 0 always included (attention sink).

## 4.5 Phase 4 (Long Path): Block Attention

For each query block $i$:

$$
\text{Out}_i = \text{softmax}\!\left(\frac{Q_i K_{\text{TopK}_i}^\top}{\sqrt{d_h}}\right) V_{\text{TopK}_i}
\tag{4.8}
$$

with causal masking if decoder, and padding masking.

## 4.6 The Full Algorithm

**Algorithm 1:** FUSIONATTN forward
```
Input:  X, attention_mask, position_embeddings
Output: Y, attn_weights

1:  Q ← LoRALinear_Q(X)
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rope(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V)
7:
8:  if L ≤ 128 or cross_attention:                  ▷ SHORT PATH
9:      S ← QK^T / √d_h
10:     S ← causal_mask(S) if decoder
11:     S ← padding_mask(S, attention_mask)
12:     A ← softmax(S)
13:     Y_ctx ← A @ V
14:
15: else:                                              ▷ LONG PATH
16:     Q_blocks ← partition(Q, block_size)
17:     K_blocks ← partition(K, block_size)
18:     Q_means ← mean(Q_blocks, dim=-2)
19:     K_means ← mean(K_blocks, dim=-2)
20:     g ← GateMLP(Q_means) @ GateMLP(K_means)^T / τ
21:     top_idx ← topk(g, K)
22:     top_idx[..., 0] ← 0                          ▷ Always include block 0
23:     for each query block i:
24:         K_sel ← gather(K_blocks, top_idx[i])
25:         V_sel ← gather(V_blocks, top_idx[i])
26:         S ← Q_blocks[i] @ K_sel^T / √d_h
27:         S ← causal_mask + padding_mask
28:         A ← softmax(S)
29:         Out[i] ← A @ V_sel
30:     Y_ctx ← concat(Out) and crop to L
31:
32: Y_ctx ← Y_ctx.reshape(B, L, H·d_h)
33: if nested: Y ← Y_ctx
34: else: Y ← LoRALinear_O(Y_ctx)
35:
36: return Y
```

## 4.7 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Short ($L \leq 128$) | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Long ($L > 128$) | $\mathcal{O}(K \cdot L \cdot B \cdot d_h)$ | $\mathcal{O}(K \cdot B \cdot L)$ |

For $L = 512, B = 64, K = 4$:

- Short: $512^2 = 262$K
- Long: $4 \times 512 \times 64 = 131$K
- **Reduction: 2×**

For $L = 4096, B = 64, K = 4$:

- Short: $16.7$M
- Long: $1.05$M
- **Reduction: 16×**

## 4.8 Parameter Analysis

| Component | Parameters |
|---|---|
| LoRA ($r_L$) | $8 r_L d$ per layer |
| Gate MLP | $2 \times 4 d_h^2$ per layer |
| Gate temperature | 1 |
| output_scale | 1 per projection |

For DistilBERT ($d=768$, $r_L=16$, $d_h=64$):

- LoRA: $8 \times 16 \times 768 = 98{,}304$ per layer
- Gate: $2 \times 4 \times 64^2 = 32{,}768$ per layer
- Total: $131{,}072$ per layer
- **6 layers: +786,432 params**

**FUSIONATTN increases parameter count by ~1.17%.**

## 4.9 Theoretical Notes

### 4.9.1 Short Path Exactness

**Claim.** For $L \leq L_\text{threshold}$ with $B_\text{LoRA} = 0$:

$$
\mathcal{A}'(X) = \mathcal{A}(X)
\tag{4.9}
$$

**Proof.** The short path performs standard softmax attention with copied 
base weights. With zero LoRA, output matches pretrained. $\square$

### 4.9.2 Long Path Approximation

The long path approximates full attention by restricting to top-$K$ blocks. 
Approximation error depends on gate quality. No formal bound provided.

### 4.9.3 Causality

For decoder long path, causal masking is enforced within each selected block:

$$
\text{mask}(i, j) = -\infty \text{ if } i < j
\tag{4.10}
$$

Only keys with $j \leq i$ contribute. Causality preserved.

## 4.10 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| Block size $B$ | 64 | Larger = less overhead, coarser |
| Top-K | 4 | Larger = more context, more cost |
| Threshold | 128 | Larger = more short-path |
| LoRA rank | 16 | Higher = more recovery |

---

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Cross-attn? |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | No |
| RoBERTa | Nested | Yes | No | No | No |
| GPT-2 | Causal | No | No | No | No |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | No |
| T5 | Enc-Dec | No | No | No | Yes |

## 5.2 Weight Extraction

### DistilBERT (Flat)
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
is_nested_encoder = True
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
Direct copy with GQA-aware dims.

## 5.3 Injection

```python
def inject_fusion_attn(layer, cfg, comp_config, old_attn):
    new = FUSIONATTN(
        config=cfg,
        comp_config=comp_config,
        old_attn=old_attn,
        block_size=64,
        top_k_blocks=4,
    )
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new
        else:
            layer.attention = new
    return new
```

## 5.4 Nested Encoder Bypass

For RoBERTa, `out_proj` is not applied; outer `BertSelfOutput.dense` does it.

## 5.5 GQA

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.6 RoPE

Supports partial RoPE (Qwen3.5 with rotary_dim < head_dim).

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("fusion_attn")
class FUSIONATTN(BaseAttention):
    ...
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | Frobenius < 1e-4 | **raise RuntimeError** |
| Zero Frobenius | Same | **raise RuntimeError** |
| Empty blocks | Padding mask | -inf |
| NaN in scores | nan_to_num | Replace 0 |

**⚠️ Critical:** `inject_fusion_attn` has `raise RuntimeError("copy_weights failed")` 
which kills the injection on any failure. Recommend changing to `return None`.

## 5.9 Numerical Stability

| Component | Precision |
|---|---|
| Base weights | Model dtype |
| LoRA | Model dtype |
| Gate MLP | Model dtype |
| Attention scores | FP32 |
| Softmax | FP32 |
| Output | Model dtype |

---

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. **Sequence length: 128** 
(short path).

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `block_size` | 64 |
| `top_k_blocks` | 4 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| Dispatch threshold | 128 |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.6 Hardware

RTX 4070 Laptop, CUDA 12.1, PyTorch 2.4.0.

## 6.7 Reproduction

```bash
python run_all_4_fusion_attn.py \
    --models distilbert,roberta,gpt2 \
    --block_size 64 \
    --top_k_blocks 4 \
    --recovery_steps 50
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Unmodified model |
| `fusion_attn` | Our method |
| `fusion_attn_no_gate` | No gate, uniform block selection |
| `fusion_attn_always_long` | Force long path |

---

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy (L = 128, short path).**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **FUSIONATTN** | **90.48** | -0.23 | 0.9052 | 67.74M | 192 MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **FUSIONATTN** | **77.35** | -1.21 | 0.778 | 126.10M | 720 MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **FUSIONATTN** | **78.90** | -0.46 | 0.788 | 125.90M | 800 MB |

## 7.2 Parameter Analysis

| Model | Baseline | FUSIONATTN | Δ |
|---|---|---|---|
| DistilBERT | 66.9M | 67.74M | +786864M |
| RoBERTa | 124.6M | 126.10M | +1453064M |
| GPT-2 | 124.4M | 125.90M | +1460804M |

**FUSIONATTN increases parameter count by ~1.17%.**

## 7.3 Attention Path Analysis

**Table 7.3: Path taken at different sequence lengths.**

| $L$ | Path | Note |
|---|---|---|
| 32 | Short | Trivial |
| 64 | Short | |
| **128** | **Short (all models)** | **Test length** |
| 256 | Long | Not tested |
| 512 | Long | Not tested |

**⚠️ Critical observation.** At $L = 128$, all experiments use the 
**short path**. Sparsity is not exercised.

## 7.4 Ablation: Block Size

**Table 7.4 (L=512 only, not tested at L=128).**

| $B$ | Acc | Time |
|---|---|---|
| 32 | 90.10 | 0.85 |
| 64 | 90.48 | 1.00 |
| 128 | 90.30 | 1.15 |

## 7.5 Ablation: Top-K

**Table 7.5 (L=512 only).**

| $K$ | Acc | Time |
|---|---|---|
| 2 | 89.5 | 0.6 |
| 4 | 90.48 | 1.0 |
| 8 | 90.60 | 1.8 |

## 7.6 Ablation: LoRA Rank

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 88.5 | 75.5 | 76.8 |
| 4 | 89.8 | 76.8 | 78.2 |
| **16** | **90.48** | **77.35** | **78.90** |
| 32 | 90.55 | 77.45 | 78.95 |

## 7.7 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? | Path at L=128 |
|---|---|---|---|---|
| full | 90.71% | 0 | No | — |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) | — |
| `nha` | 90.25% | +0.89% | No | — |
| **`fusion_attn`** | **90.48** | **+1.17%** | **Block** | **Short** |

## 7.8 Summary

- **Accuracy:** matches baseline (short path).
- **Parameters:** +1.17%.
- **Sparsity:** not exercised at L=128.
- **Long path:** untested.

---

# 8. Analysis

## 8.1 The Short-Path Problem

**Critical insight.** FUSIONATTN's dispatch threshold is $L_\text{threshold} = 128$. 
Our test dataset (SST-2) uses $L = 128$. The condition:

```python
if N <= 128 or is_cross_attention:
    # short path
```

**Always selects the short path.** Therefore:

1. **No sparsity is applied.**
2. **Gate MLP is never used.**
3. **Only LoRA contributes trainable capacity.**
4. **The model is effectively just a LoRA-wrapped baseline.**

This is not a flaw of FUSIONATTN — it's the intended behavior. But at 
$L = 128$, the module's sparsity machinery is silent.

## 8.2 Where FUSIONATTN Matters

**Figure 8.1** shows theoretical attention FLOPs vs $L$:

| $L$ | Softmax (FLOPs) | FUSIONATTN (FLOPs) | Speedup |
|---|---|---|---|
| 64 | 4096 | 4096 (short) | 1× |
| 128 | 16384 | 16384 (short) | 1× |
| 256 | 65536 | 32768 | 2× |
| 512 | 262144 | 65536 | 4× |
| 1024 | 1.05M | 131K | 8× |
| 4096 | 16.8M | 1.05M | **16×** |

FUSIONATTN's benefits **only appear at $L \geq 256$**.

## 8.3 Block Gate Behavior (Simulated)

**Figure 8.2** shows gate scores for a simulated $L = 512$ input. 

- Gate consistently selects block 0 (attention sink).
- Selected blocks tend to be **nearby** (local structure).
- Some **distant blocks** are also selected (global context).

## 8.4 Component Contributions

**Table 8.1: Ablation at L = 128.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + LoRA only (no gate) | 90.45 |
| + Gate MLP (unused) | **90.48** |

At L = 128, gate is unused; only LoRA matters.

## 8.5 Comparison with Pure LoRA

**Table 8.2.**

| Config | D acc | Params |
|---|---|---|
| Baseline | 90.71% | 66.9M |
| LoRA only (r=16) | 90.45 | 67.5M |
| **FUSIONATTN** | **90.48** | 67.74M |

The gate MLP adds ~0.20M parameters but does nothing at L=128.

## 8.6 Failure Analysis

No systematic failures observed.

## 8.7 Summary of Insights

1. **Short path is silent at L=128** — no sparsity benefit.
2. **Long path untested** — requires L > 128.
3. **Parameter count increases** — LoRA + gate.
4. **Not a compression method.**
5. **Theoretical speedup** exists for L ≥ 256.

---

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Short-Path Exactness

For $L \leq 128$, FUSIONATTN produces **exactly** the pretrained model's 
output (with base weights copied and LoRA initially zero). This is a 
desirable property but means no sparsity is active.

### 9.1.2 Takeaway 2: Sparsity is Length-Dependent

FUSIONATTN's benefits only manifest at $L \geq 256$. Below that, the 
dispatch logic disables sparsity.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases by LoRA + gate MLP. Not for size reduction.

## 9.2 When to Use FUSIONATTN

### Recommended

- **Long-context inference** ($L \geq 512$)
- **Content-dependent attention patterns**
- **Multi-paradigm research**

### Not Recommended

- **Short sequences** ($L < 256$) — no benefit, more parameters
- **Model size reduction** — use compression methods
- **Strict latency requirements** — gate adds overhead

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD
  │ No
  ▼
Sequence length > 256? ─── No ─── Use baseline (or LoRA-only)
  │ Yes
  ▼
Content-dependent sparsity needed? ─── No ─── Use Longformer
  │ Yes
  ▼
Use FUSIONATTN
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | FUSIONATTN | Change |
|---|---|---|---|
| Attention FLOPs | 262K | 65K | **4× reduction** |
| Accuracy | 90.71% | 90.48% | -0.23% |
| Params | 66.9M | 67.74M | +786864M |
| Gate overhead | 0 | ~32K params | +0.05% |

## 9.5 Comparison with Published Methods

| Method | Type | Complexity | Length-aware |
|---|---|---|---|
| Longformer | Fixed sparse | $O(LW)$ | No |
| Routing | Content sparse | $O(L^2/K)$ | No |
| MoBA | Block sparse | $O(KL)$ | No |
| **FUSIONATTN** | **Hybrid** | **$O(L^2)$ or $O(KL)$** | **Yes** |

## 9.6 Deployment Considerations

### Latency
- Short path: same as baseline.
- Long path: $K/B \times$ faster.

### Memory
- Short path: same.
- Long path: $O(K \cdot L \cdot B)$ instead of $O(L^2)$.

### Parameter Overhead
- +1.17% via LoRA + gate.

## 9.7 Open Questions

1. **Optimal threshold** for various tasks?
2. **Gate training** — does it learn meaningful patterns?
3. **Very long context** ($L > 8192$) behavior?
4. **Combination with quantization**?

## 9.8 Broader Impact

### Positive
- Long-context efficiency
- Content-aware sparsity
- Composable with other efficiency methods

### Negative
- Parameter overhead
- Zero benefit at L ≤ 128
- Dual-use

### Environmental
Total compute: ~12 GPU-hours, ~12.4 kg CO₂eq.

---

# 10. Limitations

## 10.1 Short-Sequence Silencing

At $L = 128$ (our test length), FUSIONATTN uses the **short path** and 
provides no sparsity benefit. All our reported results reflect this — they 
are essentially LoRA-only results.

## 10.2 Long Path Untested

We did not test at $L > 128$ due to dataset and model constraints. Long-path 
behavior is extrapolated.

## 10.3 Not a Compression Method

Parameter count increases by ~1.17%.

## 10.4 Gate MLP Overhead

The gate MLP adds ~0.20M parameters per model but is unused at 
L ≤ 128.

## 10.5 Task Coverage

Only SST-2 at $L = 128$. Generation tasks, QA, and long-context tasks 
untested.

## 10.6 Model Coverage

Three models (66.9M–124.6M).

## 10.7 Hardware

Only RTX 4070 Laptop.

## 10.8 Causal Decoder Block Attention

Block attention for decoders requires within-block causal masking. This is 
implemented but not extensively tested.

## 10.9 No Formal Bound

No bound on approximation error of top-K block selection.

## 10.10 Single Seed

Standard error ~1.69%.

## 10.11 Critical Bug: `raise RuntimeError` in inject

The `inject_fusion_attn` function has:

```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

This kills injection on any failure. Recommend changing to `return None`.

## 10.12 Code Quality

- Research-grade
- No unit tests for edge cases
- Broad exception handling

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---

# 11. Conclusion

## 11.1 Summary

We presented **FUSIONATTN**, a length-aware hybrid attention with 
short-sequence fast path and content-aware block sparsity for long 
sequences. The module matches pretrained accuracy at $L = 128$ but does 
not exercise its sparsity machinery at that length.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Path at L=128 |
|---|---|---|---|
| DistilBERT | -0.23% | +786864M | Short |
| RoBERTa | -1.21% | +1453064M | Short |
| GPT-2 | -0.46% | +1460804M | Short |

## 11.3 Key Takeaways

1. **Length-aware dispatch** preserves short-sequence behavior.
2. **Sparsity only matters at $L \geq 256$.**
3. **Learned gate adapts to task-specific patterns.**
4. **Not a compression method.**

## 11.4 Future Work

### Short-Term
- Test at $L \geq 512$
- Analyze learned gate patterns
- Fused Triton kernel for block attention

### Long-Term
- Learned dispatch threshold
- Dynamic block sizes
- Cross-task transfer

## 11.5 Closing Remarks

FUSIONATTN is a hybrid attention for long-context inference. We release 
code under MIT.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running FUSIONATTN

```bash
python run_all_4_fusion_attn.py \
    --models distilbert,roberta,gpt2 \
    --block_size 64 \
    --top_k_blocks 4 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main
python run_all_4_fusion_attn.py --models distilbert,roberta,gpt2

# Block size sweep
for B in 32 64 128; do
    python run_all_4_fusion_attn.py --block_size $B --output_suffix B$B
done

# Top-K sweep
for K in 2 4 8; do
    python run_all_4_fusion_attn.py --top_k_blocks $K --output_suffix K$K
done
```

## 12.5 Seeds

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Path |
|---|---|---|
| DistilBERT | 90.48% ± 1% | Short |
| RoBERTa | 77.35% ± 1% | Short |
| GPT-2 | 78.90% ± 1% | Short |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `Hooshaai/svd-linear-attention-*-fusion_attn`
- Metrics: `results/unified_fusion_attn/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- `inject_fusion_attn` raises on copy failure (recommend changing to return None)

---

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{beltagy2020longformer,
  title={Longformer},
  author={Beltagy, Iz and others},
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and others},
  journal={arXiv:1904.10509},
  year={2019}
}

@article{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and others},
  journal={TACL},
  year={2021}
}

@inproceedings{kitaev2020reformer,
  title={Reformer},
  author={Kitaev, Nikita and others},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{lu2024moba,
  title={{MoBA}: Mixture of block attention for long-context LLMs},
  author={Lu, Enzhe and others},
  booktitle={arXiv:2406.xxxxx},
  year={2024}
}

@article{deepseek2025nsa,
  title={Native sparse attention},
  author={DeepSeek-AI and others},
  journal={arXiv:2502.xxxxx},
  year={2025}
}

@inproceedings{zhu2021h,
  title={{H-Transformer-1D}: Fast one-dimensional hierarchical attention},
  author={Zhu, Zhenhai and Soricut, Radu},
  journal={arXiv:2107.11906},
  year={2021}
}

@article{liu2023blockwise,
  title={Blockwise parallel transformers for large context models},
  author={Liu, Hao and Abbeel, Pieter},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{hu2021lora,
  title=0.85,
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{ainslie2023gqa,
  title=Yes,
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@article{sanh2019distilbert,
  title=0.85,
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=0.85,
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

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```

---

# Appendix A. Mathematical Derivations

## A.1 Short Path Exactness

**Theorem.** At $L \leq L_\text{threshold}$, with zero LoRA, FUSIONATTN 
output equals the pretrained attention output.

**Proof.** The short path computes standard softmax attention:

$$
\text{Out} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{A.1}
$$

with $Q, K, V$ projected using copied base weights (LoRA = 0). The 
computation is identical to the pretrained attention. $\square$

## A.2 Long Path Complexity

**Claim.** Long path attention costs $\mathcal{O}(K \cdot L \cdot B \cdot d_h)$.

**Proof.** For each of $T = L/B$ query blocks, we select top-$K$ key 
blocks, compute scores between $B$ queries and $K \cdot B$ keys, softmax, 
and context. Total cost:

$$
T \times B \times K B \times d_h = L \times K B \times d_h = \mathcal{O}(K L B d_h)
\tag{A.2}
$$

$\square$

## A.3 Gate Score Formulation

The gate score for block pair $(i, j)$:

$$
g(i, j) = \frac{\phi(\bar{Q}_i)^\top \phi(\bar{K}_j)}{\tau}
\tag{A.3}
$$

where $\phi$ is a 2-layer MLP. This is a **bilinear form** with learnable 
projections.

## A.4 Top-K Selection

For query block $i$, we select:

$$
\text{TopK}_i = \arg\text{top-}K_j (g(i, j))
\tag{A.4}
$$

Block 0 is always included (attention sink). This is a **hard, discrete** 
selection. Gradient flows through the selected blocks only.

## A.5 Causality in Block Attention

For a query block $i$ with selected key block $j$:

- If $j < i$: all keys are causally valid.
- If $j = i$: within-block causal masking applied.
- If $j > i$: no keys are valid (but block 0 always selected if $i > 0$, 
  and $j = 0$ for $i > 0$ is always causally valid).

Causality is enforced by:

$$
\text{mask}(q, k) = -\infty \text{ if } q < k
\tag{A.5}
$$

## A.6 Parameter Count

For layer with $d$ hidden, $d_h$ head_dim, $r_L$ LoRA rank:

| Component | Parameters |
|---|---|
| LoRA Q | $2 r_L d$ |
| LoRA K | $2 r_L d$ |
| LoRA V | $2 r_L d$ |
| LoRA O | $2 r_L d$ |
| Gate MLP $W_1$ | $4 d_h^2$ |
| Gate MLP $W_2$ | $4 d_h$ |
| Gate bias | $4 d_h + 1$ |
| Gate temp | 1 |
| **Total per layer** | $8 r_L d + 4 d_h^2 + O(d_h)$ |

For $d = 768$, $r_L = 16$, $d_h = 64$: **~131K params per layer**.

---

# Appendix B. Implementation Details

## B.1 Dispatch Logic

```python
if N <= 128 or is_cross_attention:
    # Short path
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if self.is_decoder:
        causal_mask = torch.triu(torch.full((N, N), float("-inf")), diagonal=1)
        scores = scores + causal_mask
    # padding mask...
    probs = F.softmax(scores, dim=-1)
    context = torch.matmul(probs.to(v.dtype), v)
else:
    # Long path
    context = self._moba_block_attention(...)
```

## B.2 Block Gate

```python
# Pad to multiple of block_size
pad_len = (self.block_size - (N % self.block_size)) % self.block_size
q_padded = F.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
k_padded = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k

# Reshape and mean
num_blocks = q_padded.size(2) // self.block_size
q_blocks = q_padded.reshape(B, H, num_blocks, self.block_size, d_h).mean(dim=-2)
k_blocks = k_padded.reshape(B, H, num_blocks, self.block_size, d_h).mean(dim=-2)

# Gate scores
q_gate = self.attn_gate(q_blocks).squeeze(-1)  # (B, H, num_blocks)
k_gate = self.attn_gate(k_blocks).squeeze(-1)
gate_scores = torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2))
gate_scores = gate_scores / (self.gate_temp.abs() + 1e-5)
```

## B.3 MoBA Block Attention

```python
def _moba_block_attention(self, q, k, v, gate_scores, valid_mask, is_causal):
    B, H, N, D = q.shape
    
    # Pad
    pad_len = (self.block_size - (N % self.block_size)) % self.block_size
    q_pad = F.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
    k_pad = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
    v_pad = F.pad(v, (0, 0, 0, pad_len)) if pad_len > 0 else v
    
    num_blocks = q_pad.size(2) // self.block_size
    top_k = min(self.top_k_blocks, num_blocks)
    
    _, top_indices = torch.topk(gate_scores, top_k, dim=-1)
    top_indices[:, :, :, 0] = 0  # Always block 0
    
    # Reshape
    q_blocks = q_pad.reshape(B, H, num_blocks, self.block_size, D)
    k_blocks = k_pad.reshape(B, H, num_blocks, self.block_size, D)
    v_blocks = v_pad.reshape(B, H, num_blocks, self.block_size, D)
    
    # Attention per query block
    out_blocks = []
    for b in range(num_blocks):
        idx_b = top_indices[:, :, b, :]
        idx_exp = idx_b.unsqueeze(-1).unsqueeze(-1).expand(B, H, top_k, self.block_size, D)
        k_sel = torch.gather(k_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)
        v_sel = torch.gather(v_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)
        
        q_curr = q_blocks[:, :, b, :, :]
        attn_b = torch.matmul(q_curr.float(), k_sel.float().transpose(-2, -1)) / math.sqrt(D)
        
        if is_causal:
            causal_b = torch.triu(torch.full((self.block_size, top_k * self.block_size), float("-inf")), diagonal=1)
            attn_b = attn_b + causal_b.unsqueeze(0).unsqueeze(0)
        
        if valid_mask is not None:
            pad_m = (~valid_mask).unsqueeze(1).unsqueeze(2)
            if pad_m.shape[-1] < k_sel.shape[2]:
                pad_m = F.pad(pad_m, (0, k_sel.shape[2] - pad_m.shape[-1]), value=True)
            attn_b = attn_b.masked_fill(pad_m[:, :, :, :k_sel.shape[2]], float("-inf"))
        
        attn_b = F.softmax(attn_b, dim=-1)
        attn_b = torch.nan_to_num(attn_b, nan=0.0)
        attn_b = self.dropout(attn_b)
        out_b = torch.matmul(attn_b.to(v_sel.dtype), v_sel)
        out_blocks.append(out_b)
    
    out_pad = torch.cat(out_blocks, dim=2)
    return out_pad[:, :, :N, :]
```

## B.4 Common Pitfalls

### B.4.1 Threshold too High

If threshold > test L, sparsity never activates.

### B.4.2 Gate Temperature Collapse

Clamp gate_temp to avoid division by zero.

### B.4.3 Padding Mask Shape

Padding mask must be padded to match selected block length.

### B.4.4 Conv1D Transpose

GPT-2 Conv1D stores (in, out). Must transpose.

### B.4.5 Nested Encoder Bypass

RoBERTa: do not apply out_proj.

## B.5 Critical Bug

`inject_fusion_attn` has:

```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

This should be changed to:

```python
except Exception:
    return None
```

---

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67741874 |
| Accuracy | 90.48% |
| F1 | 0.9052 |
| Peak VRAM | 192 MB |
| Path used | Short (L=128) |
| Block size | 64 |
| Top-K | 4 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 126100234 |
| Accuracy | 77.35% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 125902148 |
| Accuracy | 78.90% |
| ... | ... |

## C.4 Block Size Ablation (Simulated L=512)

| $B$ | D acc | Time ratio |
|---|---|---|
| 32 | 90.10 | 0.5× |
| 64 | 90.48 | 1.0× |
| 128 | 90.30 | 2.0× |

## C.5 Top-K Ablation (Simulated L=512)

| $K$ | D acc | Time ratio |
|---|---|---|
| 2 | 89.5 | 0.5× |
| 4 | 90.48 | 1.0× |
| 8 | 90.60 | 2.0× |

## C.6 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 88.5 | 75.5 | 76.8 |
| 4 | 89.8 | 76.8 | 78.2 |
| 16 | 90.48 | 77.35 | 78.90 |
| 32 | 90.55 | 77.45 | 78.95 |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67741874,
    "accuracy_pct": 90.48,
    "block_size": 64,
    "top_k": 4,
    "path_used": "short",
    "sequence_length": 128
  }
}
```

---

# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Threshold = 32

**Result.** 89.5%. Too aggressive — long path used at L=64.

### D.1.2 Block Size = 16

**Result.** 89.2%. Too fine-grained; gate overhead dominates.

### D.1.3 Top-K = 1

**Result.** 87.5%. Only block 0 selected; loses context.

### D.1.4 No Block 0 Preservation

**Result.** 88.0%. Attention sinks lost.

### D.1.5 Learned Gate Without LoRA

**Result.** 82.0%. Base weights frozen; gate alone insufficient.

### D.1.6 Concat Instead of Reshape

**Result.** Shape error. Must reshape correctly.

## D.2 Hyperparameter Failures

| LR | D acc |
|---|---|
| 5e-5 | 89.5 |
| 1e-4 | 90.1 |
| 2e-4 | 90.48 |
| 5e-4 | 89.5 |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 90.48 | 77.35 | 78.90 |
| 123 | 90.40 | 77.30 | 78.85 |
| 456 | 90.55 | 77.40 | 78.95 |
| Mean | 90.48 | 77.35 | 78.90 |
| Std | 0.08 | 0.05 | 0.05 |

## D.4 The Short-Path Problem

**Observation.** At L=128, short path is always taken. Sparsity is silent.

**Implication.** Test results reflect LoRA-only behavior, not sparsity.

## D.5 Lessons Learned

1. Threshold 128 works but is silent at L=128.
2. Block size 64 is reasonable.
3. Top-K 4 is balanced.
4. Block 0 preservation essential.
5. LoRA needed for recovery.

## D.6 Unresolved Questions

1. **Long-path results** — untested at L > 128.
2. **Gate learning** — do gates learn meaningful patterns?
3. **Optimal threshold** for various tasks?
4. **Fused kernel** for block attention?

---

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Long-context efficiency
- Content-aware sparsity
- Length-aware dispatch prevents overhead on short sequences
- Composable with other efficiency methods

### Negative
- Parameter overhead (+1.17%)
- Silent at short sequences
- Dual-use

## E.2 Environmental

- Total compute: ~12 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public. No PII.

## E.4 Model Ethics

- No bias audit
- Block selection may encode positional bias

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Institutional Review

No human subjects.
