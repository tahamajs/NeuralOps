<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Multi-head self-attention is the core operation of every modern transformer, 
but it is also the largest consumer of parameters and compute. A standard 
12-layer BERT-base model has 12 attention heads per layer, each with its own 
$W_Q, W_K, W_V, W_O$ projections totaling 4.7M parameters per layer.

**Head redundancy is well documented.** Voita et al. (2019) find that over 
half of the heads in BERT can be pruned without significant accuracy loss. 
Michel et al. (2019) show that in many layers, a **single head** is 
sufficient. Clark et al. (2019) characterize attention patterns as 
"positional", "syntactic", or "rare-word" — with the majority falling into 
redundant categories.

**The naive approach — prune heads — loses information.** Simply removing 
a head discards whatever information it carried. If the head was doing 
something useful, accuracy drops.

**The alternative — fine-tune the full model — is expensive.** Retraining a 
pruned model requires substantial compute and full-parameter optimization.

**The opportunity.** What if we could **wrap** the pretrained attention with 
a small trainable adapter, and let the adapter learn to compensate for 
compressed head capacity? This is the idea behind **Squeezing Heads**.

## 1.2 Problem Statement

Given a pretrained transformer with $H$ attention heads per layer, we seek 
a modified module $\mathcal{A}'$ such that:

1. **Compressible:** Effective head dimension can be reduced by a factor $s$.
2. **Distillable:** Trainable adapters can recover lost capacity.
3. **Drop-in:** Pretrained weights are copied exactly; no architecture change.
4. **Portable:** Works across flat, nested, and causal architectures.
5. **Quality:** $\Delta\text{Acc} \leq 5\%$ on SST-2.

## 1.3 Our Approach: Squeezing Heads

**Squeezing Heads** replaces each attention projection with a 
**LoRA-wrapped linear layer**:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{1.1}
$$

where:
- $W_\text{frozen}$ is the pretrained projection (unchanged).
- $B A$ is a low-rank adapter with rank $r$ (trainable).
- $\alpha$ is the LoRA scaling factor.

**Figure 1.1** compares three head-compression strategies:

| Strategy | Prunes heads? | Trains full model? | Recovers info? |
|---|---|---|---|
| Head pruning | Yes | No | ❌ |
| Full fine-tune | Yes | Yes | ✅ (expensive) |
| **Squeezing Heads** | Optional | **LoRA only** | ✅ |

**Key design choices:**

1. **Wrapper, not replacement.** The pretrained attention is preserved 
   exactly. Only the LoRA parameters differ from initialization.
2. **Per-projection LoRA.** Each of $W_Q, W_K, W_V, W_O$ has its own LoRA 
   adapter, allowing fine-grained capacity allocation.
3. **GQA-aware pooling.** When the source model uses GQA (Qwen, LLaMA), 
   we pool the source KV heads into the target head count.
4. **Principal head preservation.** When the head dimension is reduced, we 
   preserve the leading head dimensions (the principal components) rather 
   than random slices.

## 1.4 Contributions

- **A LoRA-wrapped attention for head distillation.** Squeezing Heads 
  preserves pretrained weights exactly while enabling low-rank adaptation.
- **A GQA-aware copy procedure.** We handle the Grouped Query Attention 
  case by pooling source KV heads.
- **Multi-architecture support.** DistilBERT, RoBERTa, GPT-2, and Qwen with 
  RoPE, GQA, and causal masking preserved.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over LoRA rank, head-dim reduction, and GQA 
  pooling.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts 
  released under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
LoRA, and head pruning. Section 4 presents Squeezing Heads. Section 5 
describes integration with four architectures. Section 6 details the 
experimental setup. Section 7 reports results. Section 8 analyzes the 
contribution of LoRA and head-dim reduction. Section 9 discusses practical 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Per-head dimension |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $s$ | Head-dim reduction factor |
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $\mathcal{A}$ | Standard attention |
| $\mathcal{A}'$ | Squeezing Heads attention |

---
