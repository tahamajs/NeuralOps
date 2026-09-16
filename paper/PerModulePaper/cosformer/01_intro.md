<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer self-attention computes pairwise interactions between all tokens 
in a sequence:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{1.1}
$$

The intermediate attention matrix $A = \text{softmax}(QK^\top / \sqrt{d})$ 
requires $\mathcal{O}(L^2)$ time and memory. For $L = 4096$ and 16-bit 
precision, this is 32 MB per head — the dominant cost for long-context 
inference.

**Linear attention** replaces softmax with a kernel feature map 
$\phi(\cdot)$ such that the attention can be decomposed:

$$
\text{Attn}(Q_i, K, V) \approx \frac{\phi(Q_i)^\top \sum_j \phi(K_j) V_j^\top}{\phi(Q_i)^\top \sum_j \phi(K_j)}
\tag{1.2}
$$

The sums $\sum_j \phi(K_j) V_j^\top$ and $\sum_j \phi(K_j)$ can be computed 
once with $\mathcal{O}(L)$ complexity, then queried per-token.

**Problem.** Linear attention loses the **locality** of softmax attention. 
Softmax with a scale factor tends to concentrate mass on nearby tokens 
(especially with positional encodings); kernel-based linear attention 
spreads attention uniformly.

## 1.2 Cosformer's Solution

Qin et al. (2022) propose Cosformer, which recovers locality by 
**reweighting** the key and query features with cosine/sine of position:

$$
\tilde{q}_i = \phi(q_i) \cdot \cos\!\left(\frac{\pi i}{2 L}\right), \quad
\tilde{q}'_i = \phi(q_i) \cdot \sin\!\left(\frac{\pi i}{2 L}\right)
\tag{1.3}
$$

The attention output becomes:

$$
\text{Attn}(Q_i, K, V) = \frac{\sum_j \tilde{q}_i^\top \tilde{k}_j V_j + \sum_j \tilde{q}_i'^\top \tilde{k}_j' V_j}{\sum_j \tilde{q}_i^\top \tilde{k}_j + \sum_j \tilde{q}_i'^\top \tilde{k}_j'}
\tag{1.4}
$$

The cosine/sine weights decay with position, giving nearby tokens higher 
weights — mimicking softmax's locality.

## 1.3 Our Approach

We present a **production-hardened port** of Cosformer to pretrained 
transformers. Key features:

1. **Exact weight transfer** via `copy_weights` for four architecture families.
2. **Two computational paths:**
   - **Encoder:** associative accumulation via `KV = K^T V` matmul.
   - **Decoder:** chunked prefix-sum with `chunk_size = 32`.
3. **RoPE support** for modern decoders (Qwen, LLaMA).
4. **GQA support** via `repeat_kv`.
5. **LoRA recovery** at rank $r$.

## 1.4 Contributions

- **A complete port of Cosformer** to four pretrained transformer families.
- **Two implementations of the attention path:** bidirectional associative 
  and causal chunked prefix-sum.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **An honest assessment** of what works (encoder path) and what doesn't 
  (RoBERTa's training instability).
- **Full reproducibility**, including code, checkpoints, and evaluation scripts.

## 1.5 Paper Organization

Section 2 reviews linear attention, kernel methods, and positional 
reweighting. Section 3 provides background on softmax, kernel attention, 
and Cosformer. Section 4 presents our implementation. Section 5 covers 
architecture integration. Section 6 details the experimental setup. 
Section 7 reports results. Section 8 analyzes the failure modes on 
RoBERTa. Section 9 discusses. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of query heads | §1.1 |
| $H_{kv}$ | Number of KV heads | §3.2 |
| $\phi$ | Kernel feature map | §1.2 |
| $\cos, \sin$ | Positional weights | §1.2 |
| $r$ | LoRA rank | §4.4 |
| $C$ | Chunk size | §4.4 |
</parameter>

---