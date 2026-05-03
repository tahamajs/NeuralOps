
# 📘 Grouped Query Attention Family

> A collection of efficient attention variants that share keys and values across query heads to reduce memory and compute while preserving quality.

```bash
grouped_query/
├── __init__.py            # Exports all attention classes
├── gqa_attention.py       # Standard Grouped-Query Attention
├── mla_attention.py       # Multi-Head Latent Attention
├── qcqa_attention.py      # Quality & Capacity-aware GQA
└── wgqa_attention.py      # Weighted Grouped-Query Attention
```

---

## 🧠 Variants at a Glance

| Attention | Key Idea | KV Heads | Dynamic? |
|-----------|----------|----------|----------|
| **GQA** | Fixed grouping of query heads → KV heads | Fixed `num_kv_heads` | No |
| **MLA** | Low‑rank KV bottleneck (DeepSeek‑V2) | `rank` latent dim | No |
| **QCQA** | Learns per‑token quality → soft grouping | Adaptive | Yes |
| **WGQA** | Per‑head learnable weights on KV projections | Fixed | No |

---

## 🚀 Integration

All variants are automatically registered in `ATTENTION_REGISTRY`:

```python
from framework.attentions.grouped_query import (
    GroupedQueryAttention,
    MLAAttention,
    QCQAAttention,
    WeightedGroupedQueryAttention
)
```

You can replace a model’s attention layer using your framework’s injection system:

```python
# Example: replace with GQA (4 KV heads)
ATTENTION_REGISTRY["gqa"](layer, config, ...)
```

---

## 1. Grouped Query Attention (`GQA`)

Standard fixed grouping: **H query heads share K/V heads**.  
Reduces KV‑cache size by **H / K**×.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, config, num_kv_heads=4):
        # num_heads = total query heads
        # num_kv_heads = shared KV heads (must divide num_heads)
```

**Forward flow**  
1. Project Q with full heads, K/V with fewer heads.  
2. `repeat_interleave` K/V to match Q count.  
3. Standard scaled‑dot‑product attention.

---

## 2. Multi‑Head Latent Attention (`MLA`)

Inspired by **DeepSeek‑V2**.  
Compress the input into a low‑rank **latent vector** then expand to K and V **separately**.

```python
class MLAAttention(nn.Module):
    def __init__(self, config, rank=128):
        # rank = dimension of the compressed latent space
```

- **kv_down**: `hidden → rank`  
- **k_up**, **v_up**: `rank → hidden`  
- Greatly reduces KV cache size and FLOPs during inference.

---

## 3. Quality & Capacity‑Aware GQA (`QCQA`)

Dynamically adjusts how query heads are assigned to KV heads based on a **learned quality score** per token.  
The assignment is done per head using a soft grouping mechanism (gumbel‑softmax for training, hard assignment at inference).

```python
class QCQAAttention(nn.Module):
    def __init__(self, config, num_kv_heads=4):
        # group_quality: Linear(head_dim → 1) computes a score for each query token
```

**Forward outline**  
1. Compute quality scores for each query head position.  
2. Use softmax across KV heads to obtain assignment weights.  
3. Recompute K/V projections once and multiply by assignment weights.  
4. Standard attention with the dynamically weighted KV buffer.

*This variant adapts to token importance on‑the‑fly.*

---

## 4. Weighted Grouped Query Attention (`WGQA`)

Similar to GQA but with **learnable channel‑wise weights** (`k_weights`, `v_weights`) applied to the KV heads before repeating.  
These weights are learned during training and can emphasise certain KV heads over others.

```python
class WeightedGroupedQueryAttention(nn.Module):
    def __init__(self, config, num_kv_heads=4):
        self.k_weights = nn.Parameter(torch.ones(num_kv_heads))
        self.v_weights = nn.Parameter(torch.ones(num_kv_heads))
```

**Flow**  
- Project K/V normally.  
- Multiply element‑wise: `k = k * self.k_weights[..., None]`.  
- Then repeat and proceed.

---

## 🔧 Customisation Tips

- All classes accept a HuggingFace‑style `config` object (e.g., `DistilBertConfig`).
- Use `num_kv_heads` to control compression ratio; must divide `num_attention_heads`.
- For MLA, `rank` is a hyperparameter; typical values 64–256 depending on model size.
- Registration via `ATTENTION_REGISTRY` makes them directly swappable in your pipeline.

---

## 📜 References

- **GQA**: Ainslie, et al. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.” 2023.
- **MLA**: DeepSeek‑V2. “DeepSeek‑V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.” 2024.
- **QCQA** / **WGQA**: Custom extensions for adaptive KV sharing.

All implementations are provided as building blocks for research and production.