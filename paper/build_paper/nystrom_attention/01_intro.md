<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Standard scaled dot-product attention (Vaswani et al., 2017) computes:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

The intermediate matrix $Q K^\top \in \mathbb{R}^{N \times N}$ has 
$\mathcal{O}(N^2)$ time and memory. For long sequences ($N > 4096$), this 
becomes the dominant cost.

**Low-rank attention approximation** exploits the observation that softmax 
attention matrices are often approximately low-rank. Nyströmformer (Xiong 
et al., 2021) applies the **Nyström method** from numerical linear algebra: 
instead of computing the full attention matrix, use $m \ll N$ landmark 
rows and columns to reconstruct it.

The reconstruction takes the form:

$$
\tilde{A} = \text{softmax}(Q K_m^\top) \cdot \text{pinv}(\text{softmax}(Q_m K_m^\top)) \cdot \text{softmax}(Q_m K^\top)
\tag{1.2}
$$

where $Q_m, K_m$ are landmark queries and keys. This gives the output:

$$
\text{Output} = \tilde{A} V = \underbrace{\text{softmax}(Q K_m^\top)}_{N \times m} \cdot \underbrace{\text{pinv}(A_{mm})}_{m \times m} \cdot \underbrace{\text{softmax}(Q_m K^\top) V}_{m \times d_h}
\tag{1.3}
$$

Complexity: $\mathcal{O}(N m d)$ instead of $\mathcal{O}(N^2 d)$.

## 1.2 Why Reproduce Nyströmformer?

Nyströmformer was designed for BERT-like architectures. We ask: **can it 
be ported to DistilBERT, RoBERTa, GPT-2, and Qwen3.5 without losing 
accuracy?**

Practical question because:

1. **Causal decoders.** Nyström approximation assumes non-causal attention. 
   For causal decoders, we must fall back to dense attention or adapt the 
   algorithm.
2. **Nested encoders.** RoBERTa's attention output projection lives outside 
   the inner attention, requiring nested encoder handling.
3. **Fused Conv1D.** GPT-2's `c_attn` fuses Q, K, V, requiring split-and-transpose.
4. **GQA + RoPE.** Qwen uses GQA and rotary position embeddings, requiring 
   both `repeat_kv` and RoPE handling.

A successful reproduction would enable Nyström approximation on any 
transformer.

## 1.3 Our Contributions

We make the following contributions:

- **A faithful reproduction of Nyströmformer** with iterative 
  Newton-Schulz pinv, adaptive pooling landmarks, and associative 
  multiplication.
- **A multi-architecture portability study** covering four families.
- **Engineering hardening** including short-sequence dense fallback, 
  causal decoder path, and safe injection returning `None` on failure.
- **An honest empirical evaluation** on SST-2 (short sequence, dense 
  fallback) and analysis of when the Nyström branch activates.
- **Full reproducibility.** Code, weights, and scripts released under 
  Apache 2.0.

We emphasize: **this paper is a reproduction, not a novel architecture.** 
We do not claim to improve Nyströmformer.

## 1.4 What Nyströmformer Does and Does Not Do

**Does:**
- Reduces **attention compute** for long sequences ($N \gg m$).
- Achieves $\mathcal{O}(N m d)$ complexity.
- Preserves softmax attention for short sequences (fallback).

**Does not:**
- Reduce parameter count.
- Reduce KV-cache size.
- Help for short sequences ($N \leq 2m$).

This is important context: Nyströmformer is **orthogonal** to low-rank 
weight compression (MinimalSVD, SVD-LLM V2) and KV-cache compression (MLA).

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on the 
Nyström method. Section 4 formalizes the algorithm. Section 5 describes 
multi-architecture integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes when the Nyström branch activates. 
Section 9 discusses guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $m$ | Number of landmarks |
| $d_h$ | Per-head dimension |
| $H$ | Number of heads |
| $H_{kv}$ | Number of KV heads (GQA) |
| $Q, K, V$ | Query, Key, Value ($N \times d_h$) |
| $Q_m, K_m$ | Landmark queries, keys ($m \times d_h$) |
| $A_{mm}$ | Landmark attention matrix ($m \times m$) |
| $\text{pinv}(\cdot)$ | Moore-Penrose pseudoinverse |
| $A$ | Full attention matrix ($N \times N$) |
| $\tilde{A}$ | Nyström approximation ($N \times N$) |

---