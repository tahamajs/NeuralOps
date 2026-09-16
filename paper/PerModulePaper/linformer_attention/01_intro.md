# 1. Introduction

## 1.1 Motivation

Transformer self-attention has quadratic complexity in sequence length $L$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V, 
\quad QK^\top \in \mathbb{R}^{L \times L}
\tag{1.1}
$$

The intermediate attention matrix costs $\mathcal{O}(L^2)$ time and memory. 
At $L = 4096$ with $H = 12$ heads, this is ~200M entries per batch item, 
becoming prohibitive for deployment.

**Linformer** (Wang et al., 2020) observes that the attention matrix 
$P = \text{softmax}(QK^\top/\sqrt{d_h})$ is **empirically low-rank**. They 
exploit this by projecting $K$ and $V$ to a fixed dimension $k \ll L$ via 
learned matrices $E, F \in \mathbb{R}^{k \times L}$:

$$
\tilde{K} = E K, \quad \tilde{V} = F V, \quad \tilde{K}, \tilde{V} \in \mathbb{R}^{k \times d_h}
\tag{1.2}
$$

Attention then becomes $\mathcal{O}(L \cdot k)$:

$$
\text{Attn}_\text{Lin}(Q, K, V) = \text{softmax}\!\left(\frac{Q \tilde{K}^\top}{\sqrt{d_h}}\right) \tilde{V}
\tag{1.3}
$$

For fixed $k$, this is linear in $L$.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ and a target low-rank dimension 
$k$, we seek an implementation of Linformer such that:

$$
\text{Cost}(\mathcal{A}_\text{Lin}) = \mathcal{O}(L \cdot k), \qquad
\mathcal{A}_\text{Lin} \approx \mathcal{A}
\tag{1.4}
$$

with the following production constraints:

1. **Multi-architecture:** Support DistilBERT, RoBERTa (nested), GPT-2 
   (Conv1D), and Qwen (GQA + RoPE).
2. **Causal correctness:** For decoders, ensure causality is preserved.
3. **Variable sequence length:** Handle $L$ ranging from small (≤$k$) to 
   larger than the pre-allocated $E, F$ matrices.
4. **No base retraining:** Only LoRA may be trained.
5. **Bounded parameter overhead:** Report honestly.

## 1.3 Our Implementation

We present a production-hardened Linformer with three key contributions:

### 1.3.1 Diagonal-Init E, F Matrices

We initialize $E$ and $F$ as identity-in-first-block:

$$
E_{ij} = \begin{cases} 1 & i = j, i < \min(k, L_\text{max}) \\ \sim \mathcal{N}(0, 0.02^2) & \text{otherwise} \end{cases}
\tag{1.5}
$$

This ensures the initial projection is (approximately) the identity, so 
Linformer starts as standard attention and deviates only as training 
progresses.

### 1.3.2 Adaptive Slicing

We handle three cases for $E, F$ usage based on $L$:

- **$L \leq k$:** Use the top-left $L \times L$ block (no compression).
- **$L \leq L_\text{max}$:** Use $E[:k, :L]$ (standard Linformer).
- **$L > L_\text{max}$:** Interpolate $E$ to length $L$ (graceful extension).

### 1.3.3 Causal Fallback

For decoders, low-rank projection of $K$ and $V$ mixes future and past 
information, breaking causality. We detect this case and fall back to 
**exact dense causal attention** when $L > k$:

$$
\text{if } \text{is\_decoder} \text{ and } L > k: \text{ use dense causal}
\tag{1.6}
$$

This preserves correctness at the cost of quadratic complexity for 
long-sequence decoders.

## 1.4 Contributions

- **A production-hardened Linformer.** Handles four architectures with 
  GQA, RoPE, nested encoders, and Conv1D.
- **Adaptive E/F slicing.** Graceful behavior across all sequence lengths.
- **Causal-safe fallback.** Preserves correctness for decoders.
- **Honest parameter accounting.** We state the parameter increase 
  explicitly.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on Linformer, 
attention, and low-rank projections. Section 4 presents our implementation. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes trade-offs. Section 9 discusses 
deployment. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $k$ | Low-rank projection dimension |
| $L_\text{max}$ | Pre-allocated max sequence length |
| $H, H_{kv}$ | Query, KV heads |
| $d_h$ | Head dimension |
| $E, F$ | Projection matrices |
| $\tilde{K}, \tilde{V}$ | Projected keys, values |

---