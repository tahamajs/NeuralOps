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
  Apache 2.0.

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