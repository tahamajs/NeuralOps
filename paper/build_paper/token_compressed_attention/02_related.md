# 2. Related Work

## 2.1 KV Cache Compression

### 2.1.1 Quantization

KIVI (Liu et al., 2024) applies asymmetric 2-bit quantization per channel. 
KVQuant (Hooper et al., 2024) uses non-uniform quantization with outlier 
handling. These methods are **orthogonal** to token selection: our 
compression can be applied first, then quantized.

### 2.1.2 Token Eviction

H2O (Zhang et al., 2023) tracks accumulated attention scores and evicts 
low-scoring tokens. StreamingLLM (Xiao et al., 2024) preserves initial 
tokens as "attention sinks" plus a sliding window. Scissorhands (Liu et al., 
2023) uses a similar paradigm.

**Our distinction.** H2O and StreamingLLM use **fixed** criteria (attention 
mass or position). TCA uses a **learnable** score, trained end-to-end with 
the recovery fine-tuning.

### 2.1.3 Low-Rank Projection

Multi-Query Attention (Shazeer, 2019) and Grouped-Query Attention (Ainslie 
et al., 2023) share KV heads across query heads. Multi-Head Latent Attention 
(DeepSeek-V2, 2024) projects KV to a low-rank latent. These reduce KV size 
per token but do not reduce the token count.

### 2.1.4 Hybrid Methods

SnapKV (Li et al., 2024) combines observation windows with token selection. 
PyramidKV (Cai et al., 2024) applies different budgets per layer. Our method 
is a single-layer module that can be stacked with these.

## 2.2 Learnable Sparsity

**Adaptive attention.** Adaptive Attention Span (Sukhbaatar et al., 2019) 
learns per-head attention spans. Routing Transformer (Roy et al., 2021) 
learns content-based token clusters. **TCA learns per-token importance**, 
a finer granularity than per-head spans.

**Dynamic token pruning.** Progressive Token Pruning (EAT) prunes at 
intermediate layers. TCA prunes inside each attention block.

## 2.3 Compressed Attention for Vision

DynamicViT (Rao et al., 2021), A-ViT (Yin et al., 2022), and EViT (Liang 
et al., 2022) prune tokens in vision transformers. These operate on the 
residual stream, while TCA operates on K/V only.

## 2.4 Position of Our Work

| Method | Criterion | Learnable? | Preserves [CLS]? | Causal-safe? |
|---|---|---|---|---|
| H2O | Accumulated attn | No | Partial | Partial |
| StreamingLLM | Position | No | Yes (sinks) | Yes |
| SnapKV | Observation window | No | Partial | Partial |
| **TCA** | **Score net + norm** | **Yes** | **Yes** | **Yes** |

**Key distinction.** TCA's selection criterion is learned. This is the first 
KV token selection method to jointly learn the scoring function with LoRA 
recovery.

---

## Comparison Table

| Method | Type | Params | KV bytes | Learnable |
|---|---|---|---|---|
| Baseline | — | $P$ | $2 H_{kv} L d_h$ | — |
| H2O | Eviction | $P$ | $2 H_{kv} k d_h$ | No |
| StreamingLLM | Sink + window | $P$ | $2 H_{kv} (s + w) d_h$ | No |
| KIVI (2-bit) | Quantization | $P$ | $\frac{1}{4} \cdot 2 H_{kv} L d_h$ | No |
| **TCA** | **Learnable selection** | $P + \epsilon$ | $2 H_{kv} k d_h$ | **Yes** |

---