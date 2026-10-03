# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B 	imes L 	imes d}$, multi-head attention projects keys, queries, and values via learned matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

The attention output is computed as

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(rac{QK^	op}{\sqrt{d_h}}ight)V
$$

This produces the standard quadratic attention bottleneck in sequence length.

## 3.2 Grouped-Query Attention

Grouped-query attention reduces the number of KV heads relative to Q heads, lowering cache size and memory cost while preserving a large query head count. This design is common in modern LLMs and matters for efficient inference.

## 3.3 Rotary Position Embeddings

RoPE injects position-dependent rotations into query and key vectors to encode relative order. This matters for causal generation and for any module that must preserve positional behavior under long-context usage.

## 3.4 LoRA Recovery

LoRA learns a low-rank residual update of the form

$$
\Delta W = rac{lpha}{r}BA
$$

where $B$ and $A$ are trainable low-rank matrices and $W$ is the frozen base weight. In the recovery phase, the model can adapt to a new attention structure while preserving the original pretrained weights.

## 3.5 Attention Sinks and Heavy Hitters

Recent work shows that a small number of tokens can dominate attention concentration. This motivates guardrails for padding, causal masking, and stable initialization when replacing attention paths.
