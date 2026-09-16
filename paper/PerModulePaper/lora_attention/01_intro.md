# 1. Introduction

## 1.1 Motivation

Full fine-tuning of a 7B-parameter language model requires:
- **~28 GB** of optimizer state (AdamW with FP32)
- **~14 GB** of gradients (FP16)
- **Multiple hours** of GPU time
- **Backprop through every weight**

LoRA (Hu et al., 2021) reduces this dramatically by freezing the base model 
and training only rank-$r_L$ adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad
B \in \mathbb{R}^{d \times r_L}, \quad A \in \mathbb{R}^{r_L \times k}
\tag{1.1}
$$

with $r_L \ll \min(d, k)$. For $r_L = 16$ and $d = 768$, this reduces 
trainable parameters per projection by **~48×**.

**Why attention?** Most existing LoRA implementations focus on the FFN 
because it contains most parameters. However, attention projections contain 
the **task-specific** information and benefit disproportionately from 
fine-tuning (He et al., 2022; Zhang et al., 2023).

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with attention projections 
$\{W_Q, W_K, W_V, W_O\}$, we seek a parameter-efficient adaptation mechanism 
$\mathcal{A}'$ such that:

$$
\text{Params}_\text{trainable}(\mathcal{A}') \ll \text{Params}(\mathcal{M})
\tag{1.2}
$$

subject to:

1. **Multi-architecture support:** DistilBERT (flat), RoBERTa (nested), 
   GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).
2. **Zero inference overhead:** Adapters must be mergeable into base weights.
3. **Compatibility with compression:** Must work on top of SVD/pruned/quantized 
   models.
4. **Weight transfer:** Must copy base weights exactly before adding adapters.

## 1.3 Our Contributions

We present **LoRA Attention**, a production-hardened implementation with:

### 1.3.1 Four-Architecture Weight Transfer

`copy_weights` handles:
- **GPT-2 Conv1D:** Transpose and split fused `c_attn`.
- **RoBERTa nested:** Extract from `self.query/key/value` and `output.dense`.
- **Qwen GQA:** Handle $H_{kv} < H$.
- **DistilBERT flat:** Direct copy.

### 1.3.2 Safe Nested Detection

We check for `self.query` **before** class-name heuristics, correctly 
classifying DistilBERT (which ends with "SelfAttention" but is flat).

### 1.3.3 Explicit Compression Compatibility

Unlike generic LoRA, our module is designed to attach to **compressed** 
attention modules (SVDLinear, A3FactorizedLinear, etc.) as a recovery 
mechanism.

## 1.4 Paper Organization

Section 2 reviews related work. Section 3 provides background on LoRA and 
attention. Section 4 presents LoRA Attention. Section 5 describes 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes trade-offs. Section 9 discusses usage patterns. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Query heads |
| $H_{kv}$ | KV heads |
| $r_L$ | LoRA rank |
| $\alpha$ | LoRA scaling constant |
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $A, B$ | LoRA adapter matrices |

---