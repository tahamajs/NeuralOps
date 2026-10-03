# 2. Related Work

## 2.1 Low-Rank Factorization

**Classical SVD.** Eckart-Young theorem gives the optimal rank-$r$ 
approximation in Frobenius norm. We apply this to attention projections.

**Linformer** (Wang et al., 2020) projects K, V to a fixed sequence dimension 
using learned matrices E, F. This is a different axis of compression 
(sequence, not feature).

**ALBERT** (Lan et al., 2020) shares parameters across layers, exploiting 
cross-layer redundancy. Our method exploits **intra-layer** Q/K/V correlation.

**Nyströmformer** (Xiong et al., 2021) uses Nyström landmarks for attention 
approximation, not projection compression.

## 2.2 Attention Compression

**Multi-Query Attention (MQA)** (Shazeer, 2019) shares K, V across all query 
heads, reducing KV parameters by $H/H_{kv}$. **Grouped-Query Attention (GQA)** 
(Ainslie et al., 2023) is a generalization.

**Multi-Head Latent Attention (MLA)** (DeepSeek-V2, 2024) projects KV to a 
low-rank latent. This is a **sequence-level** projection.

**Our distinction.** We compress the **weight matrices** directly, sharing 
the input basis across Q, K, V via joint SVD. This is orthogonal to 
sequence-level compression methods.

## 2.3 Structured Pruning

**Structured pruning** removes entire channels. **SVD-based methods** 
(like ours) factorize instead.

**Sliding window + SVD** hybrids exist (SVD-LLM, etc.), but we do not 
combine these.

## 2.4 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) freezes base weights and trains low-rank adapters. 
**AdaLoRA** (Zhang et al., 2023) allocates rank adaptively. **DoRA** (Liu 
et al., 2024) decomposes magnitude and direction.

**Our use.** We freeze SVD factors and apply LoRA on top, allowing 
recovery fine-tuning without undoing the compression.

## 2.5 Position of Our Work

| Method | Axis | Joint? | Params reduction |
|---|---|---|---|
| Linformer | Sequence | No | No (adds params) |
| MQA/GQA | KV heads | No | Yes |
| MLA | Sequence latent | No | Yes |
| **Compressed MHA** | **Feature (weights)** | **Yes (QKV)** | **Yes** |

**Key distinction.** Compressed MHA is the first to jointly factorize Q, K, V 
with a shared input basis, exploiting intra-layer correlation. This yields 
smaller compressed matrices than independent SVD.

---

## Comparison Table

| Method | Compresses | Reduction | Joint | Multi-arch |
|---|---|---|---|---|
| SVD (independent) | Q, K, V, O | $4d^2 \to 8rd$ | No | Yes |
| **Compressed MHA** | Q, K, V, O | $4d^2 \to 2r(d+d_h)$ | **Yes** | **Yes** |
| MQA | K, V | $2d^2 \to 2 d d_{kv}$ | No | Yes |
| MLA | K, V | Sequence-level | No | Specific |

---