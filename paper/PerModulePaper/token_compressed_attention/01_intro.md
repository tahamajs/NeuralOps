# 1. Introduction

## 1.1 Motivation

Transformer inference on long sequences is dominated by two costs:

1. **Attention compute:** $\mathcal{O}(L^2)$ in sequence length $L$.
2. **KV cache memory:** $\mathcal{O}(L)$ per layer per head.

For a 12-layer model with 12 heads and head dimension 64, the KV cache 
requires $12 \times 12 \times 2 \times L \times 64$ floats — about **6 MB 
per 1K tokens** at FP16. At 32K tokens, this grows to ~200 MB, exceeding 
the footprint of the model itself.

**KV cache compression** addresses this bottleneck. Existing approaches fall 
into three categories:

1. **Quantization** (KIVI, KVQuant): reduce bit-width per element.
2. **Eviction** (H2O, StreamingLLM): drop unimportant tokens.
3. **Low-rank projection** (MLA, MQA): project to smaller subspaces.

Our work belongs to the second category. We ask: **can we learn which tokens 
to keep, rather than using a fixed heuristic?**

## 1.2 Problem Statement

Given a sequence of $L$ tokens, we seek a compressed KV sequence of length 
$k = \lceil \rho L \rceil$ such that:

$$
\text{Attn}(q_i, K_c, V_c) \approx \text{Attn}(q_i, K, V)
\tag{1.1}
$$

for all query positions $i$, where $K_c, V_c$ are the compressed KV tensors.

We impose the following constraints:

1. **Preserve `[CLS]`:** The first token must always be retained.
2. **Respect causality:** For decoders, query at position $i$ must not attend 
   to compressed keys whose original position is $> i$.
3. **Learn the criterion:** The selection must be trainable, not a fixed rule.
4. **Cross-architecture:** Must work for flat, nested, and causal attention.
5. **No base retraining:** Only the scoring head and LoRA may be trained.

## 1.3 Our Approach

We introduce **TokenCompressedAttention** (TCA). Given projected $Q, K, V$, we:

1. Compute a per-token importance score combining:
   - The $\ell_2$ norm of the token's hidden state (a proxy for activation energy).
   - A learnable score predicted by a small MLP.
2. Always retain `[CLS]` (position 0).
3. Select the top-$k$ tokens by score.
4. Gather the corresponding rows of $K$ and $V$.
5. Apply standard attention with strict causal masking against the **original** positions.

**Key innovation.** The learnable score is trained end-to-end with the recovery 
LoRA. This lets the model discover which tokens matter, rather than relying 
on a fixed norm-based heuristic.

## 1.4 Contributions

- **A learnable token-selection module.** We propose a lightweight 
  score_net (66.4M parameters) that learns to identify 
  important tokens.
- **Causal-safe selection.** We derive a strict causal mask over compressed 
  keys, ensuring decoder correctness.
- **`[CLS]` guarantee.** The first token is always retained — critical for 
  classification tasks where `[CLS]` aggregates the sequence.
- **A production-hardened implementation.** Handles DistilBERT, RoBERTa 
  (nested), GPT-2 (Conv1D), and Qwen (GQA + RoPE).
- **Honest framing.** We do **not** claim parameter reduction; we reduce 
  inference compute and memory only.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
KV cache, and token selection. Section 4 presents TCA. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the contribution of each component. Section 9 discusses 
trade-offs. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Full sequence length |
| $k$ | Compressed KV length |
| $\rho$ | Compression ratio ($k = \lceil \rho L \rceil$) |
| $H$ | Query heads |
| $H_{kv}$ | KV heads |
| $d_h$ | Head dimension |
| $s_i$ | Importance score for token $i$ |
| $\mathcal{S}$ | Set of selected token indices |

---