<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer architectures are typically described in terms of **blocks**, 
where each block comprises:

$$
\text{Block}(X) = X + \text{Attention}(\text{LayerNorm}(X))
\tag{1.1}
$$

(pre-LayerNorm) or:

$$
\text{Block}(X) = \text{LayerNorm}(X + \text{Attention}(X))
\tag{1.2}
$$

(post-LayerNorm). The attention mechanism is the focus of most efficiency 
research, but the **block structure itself** is often left untouched.

We ask: what happens if we replace the entire block, including its 
normalization and residual structure, rather than just the attention?

0.85

<!-- 
EXAMPLE continuation:
"This is not just an academic question. Many transformer efficiency methods 
— e.g., RetNet, Mamba, GLA — are described at the block level, not the 
attention level. Adapting these methods to existing pretrained models 
requires replacing the block, which means introducing new LayerNorm 
parameters and changing the residual structure."
-->

## 1.2 Problem Statement

Given a pretrained transformer block:

$$
\text{Block}_\text{pre}(X) = \text{LN}_\text{pre}(X + \text{Attn}_\text{pre}(X))
\tag{1.3}
$$

we seek a replacement block:

$$
\text{Block}_\text{new}(X) = \text{LN}_\text{new}(X + \text{Attn}_\text{new}(X))
\tag{1.4}
$$

such that $\text{Block}_\text{new} \approx \text{Block}_\text{pre}$, where 
$\text{LN}_\text{new}$ may be fresh (not pretrained) and $\text{Attn}_\text{new}$ 
may be a different attention mechanism.

## 1.3 Our Approach

**GatedDeltaNetBlock** offers two variants:

1. **Full-attention variant** (`use_full_attention=True`): uses 
   `nn.MultiheadAttention` with pretrained Q/K/V/O copied into its fused 
   `in_proj_weight` and `out_proj`.

2. **GatedDeltaNet variant** (`use_full_attention=False`): uses GatedDeltaNet 
   (see companion paper).

Both variants wrap the attention in a **residual + LayerNorm** structure.

## 1.4 Contributions

- **A block-level wrapper** for attention replacement that gives explicit 
  control over normalization and residual.
- **An honest analysis** of two structural failure modes:
  - LayerNorm initialization mismatch
  - Silent failure in the MHA weight-transfer path
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **A negative result:** the block adds overhead without accuracy benefit.
- **Full reproducibility**, including code paths and error handling.

## 1.5 Paper Organization

Section 2 reviews block-level attention replacement, LayerNorm 
initialization, and hybrid approaches. Section 3 provides background on 
transformer blocks, LayerNorm, and `nn.MultiheadAttention`. Section 4 
presents GatedDeltaNetBlock. Section 5 covers integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes failure 
modes. Section 9 discusses. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $\text{Block}(X)$ | Transformer block | §1.1 |
| $\text{LN}$ | LayerNorm | §1.1 |
| $\text{Attn}$ | Attention mechanism | §1.1 |
| $\gamma, \beta$ | LayerNorm parameters | §3.2 |
| $d$ | Hidden size | §1.3 |
| $H$ | Number of heads | §1.3 |
| $r$ | LoRA rank | §4.4 |
</parameter>

---