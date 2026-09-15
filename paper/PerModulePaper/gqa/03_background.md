<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Attention

Standard MHA computes:

$$
\begin{aligned}
Q_h &= X W_Q^{(h)}, \quad K_h = X W_K^{(h)}, \quad V_h = X W_V^{(h)} \\
\text{head}_h &= \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h \\
\text{MHA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.1}
$$

where $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_h}$.

**KV cache.** During autoregressive generation, $K_h$ and $V_h$ are stored 
per head per token. The total cache size is:

$$
\text{KV}_\text{MHA} = 2 H d_h L B \cdot \text{bytes}
\tag{3.2}
$$

## 3.2 Multi-Query Attention

MQA shares a single $K, V$ pair across all query heads:

$$
K = X W_K, \quad V = X W_V, \quad W_K, W_V \in \mathbb{R}^{d \times d_h}
\tag{3.3}
$$

Query heads still project separately. During generation:

$$
\text{KV}_\text{MQA} = 2 d_h L B \cdot \text{bytes}
\tag{3.4}
$$

Reduction factor: $H \times$.

## 3.3 Grouped-Query Attention

GQA partitions $H$ query heads into $G$ groups, each with its own $K, V$:

$$
G = H_{KV} \le H
\tag{3.5}
$$

The $g$-th group produces:

$$
K_g = X W_K^{(g)}, \quad V_g = X W_V^{(g)}, \quad W_K^{(g)}, W_V^{(g)} \in \mathbb{R}^{d \times d_h}
\tag{3.6}
$$

Query heads in group $g$ share $K_g, V_g$. The KV cache size is:

$$
\text{KV}_\text{GQA} = 2 G d_h L B \cdot \text{bytes}
\tag{3.7}
$$

Reduction factor: $H / G$. The MHA and MQA cases are special cases at 
$G = H$ and $G = 1$.

## 3.4 KV Head Repetition

During the forward pass, each KV head must be repeated to match the 
corresponding query heads. If $H / G = r$ (integer), each KV head is 
replicated $r$ times:

$$
\tilde{K} = \text{repeat\_kv}(K, r) \in \mathbb{R}^{B \times H \times L \times d_h}
\tag{3.8}
$$

This is implemented efficiently via `reshape` + `expand` (no memory copy 
in most cases).

## 3.5 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.9}
$$

GQA applies RoPE to the shared $K$ heads before repetition. This is the 
standard configuration in LLaMA, Mistral, and Qwen.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. In GQA, we freeze the KV projection weights 
and train LoRA adapters for recovery.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $H$ | Query head count |
| $G$ | KV head count (= group count) |
| $d_h$ | Per-head dimension |
| $L$ | Sequence length |
| $B$ | Batch size |
| $r$ | KV repeat factor ($H / G$) |
| $r_L$ | LoRA rank |
| $\theta_i$ | RoPE angle at position $i$ |
