<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer inference cost has two dominant components: the quadratic 
attention matrix ($\mathcal{O}(L^2)$ in sequence length) and the linear 
projection and FFN operations ($\mathcal{O}(L d^2)$). For long sequences, 
the second component dominates once $L \gg d$. Every token that survives 
through the attention stack contributes to both.

**Observation.** Real text contains many tokens with low information content: 
padding, repeated function words, subword fragments that add little to the 
residual stream. For a trained sentiment classifier, most of the final 
prediction depends on a small subset of high-signal tokens.

**Token pruning** exploits this by removing unimportant tokens as the model 
runs, reducing downstream compute. Prior work has explored this for vision 
transformers (DynamicViT, EViT) and for text encoders (LTP, Kim et al., 2022). 
EAT extends this to a multi-architecture setting, combined with weight 
transfer from the pretrained attention.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ layers and target 
downstream task $\mathcal{T}$, we seek a modified attention block $\mathcal{A}'$ 
such that:

1. $\mathcal{A}'$ matches $\mathcal{A}$ at initialization (via weight 
   transfer, no retraining of base weights).
2. $\mathcal{A}'$ enables trainable adaptation through a small number of 
   additional parameters (LoRA).
3. $\mathcal{A}'$ optionally prunes a fraction of tokens at inference time.
4. $\mathcal{A}'$ works across four attention architectures.

We do **not** aim to reduce model size. EAT is designed for **inference-time 
efficiency**, not for model compression. We state this explicitly because 
prior work has conflated the two.

## 1.3 Our Approach: EAT

EAT replaces the standard MHSA block with:

1. **Frozen base projections** $W_Q, W_K, W_V, W_O$ transferred from the 
   pretrained attention.
2. **Trainable LoRA adapters** added to each projection, providing a 
   parameter-efficient recovery path.
3. **Optional progressive token pruning** at inference time, controlled by 
   a lightweight scorer.

The forward pass computes:

$$
\text{EAT}(X) = W_O \cdot \text{Concat}(\text{head}_1, \dots, \text{head}_H) + \text{LoRA}(X)
\tag{1.1}
$$

where each head applies standard scaled dot-product attention to 
LoRA-adapted projections.

If `prune_layer` is provided and the model is not training, the output of 
each attention block passes through a pruning module that reduces the active 
sequence length.

## 1.4 Contributions

- **A multi-architecture attention replacement.** EAT works on flat 
  (DistilBERT), nested (RoBERTa/BERT), causal (GPT-2), and GQA+RoPE 
  (Qwen/LLaMA) attention blocks without code changes.
- **Weight transfer with defensive fallback.** A single `copy_weights` 
  routine handles fused Conv1D, nested encoder, and standard linear 
  projections, with Frobenius validation.
- **Optional progressive pruning.** An attachable pruning layer that reduces 
  inference token count without affecting training.
- **Honest evaluation.** We report parameter count changes, accuracy drops, 
  and inference overhead transparently. EAT is not a compressor.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MHSA, GQA, 
RoPE, LoRA, and token pruning. Section 4 presents EAT. Section 5 details 
multi-architecture integration. Section 6 gives the experimental setup. 
Section 7 reports results. Section 8 analyzes contributions and failure 
modes. Section 9 provides practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden size |
| $L$ | Sequence length |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Per-head dimension |
| $N_\ell$ | Layer count |
| $r$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| $W_Q, W_K, W_V, W_O$ | Attention projection matrices |
| $\pi$ | Token pruning operator |

---