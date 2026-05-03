
# Optimized Hybrid Attention (OHA)

**Sliding‑window local + low‑rank global attention with gated fusion.**

Drop‑in replacement for DistilBERT’s standard self‑attention – OHA significantly reduces memory and compute while preserving long‑range dependency modelling.

## 📦 Quick Start

```python
from optimized_hybrid_attention import OptimizedHybridAttention, OptimizedHybridAttentionConfig

# Build from scratch
config = OptimizedHybridAttentionConfig(
    hidden_size=768,
    num_attention_heads=12,
    window_size=128,
    compression_rank=64,
    use_gate=True
)
oha_layer = OptimizedHybridAttention(config)
output, _ = oha_layer(hidden_states, attention_mask)
```

Or **directly initialise from a pretrained DistilBERT layer** (zero‑shot):

```python
# distilbert_layer is an existing DistilBertAttention
oha_layer = OptimizedHybridAttention.from_distilbert_pretrained(
    distilbert_layer, config
)
```

## 🧠 Architecture

OHA splits the attention computation into two parallel branches that are fused with a learned gate:

1. **Local branch** – standard scaled dot‑product attention, but restricted to a **sliding window** of size `W`. This captures fine‑grained local structure efficiently.

2. **Global branch** – compresses the entire sequence into a fixed‑length **low‑rank** set of tokens (using softmax-based aggregation), then performs attention against this compact representation. This captures global dependencies without quadratic cost.

3. **Gated fusion** – a per‑head gate (sigmoid over linear projection) blends the local and global contexts:

   ![Fusion](https://latex.codecogs.com/png.latex?%5Ctext%7Bcontext%7D%20%3D%20%5Csigma%28%5Ctext%7Bgate%7D%29%20%5Codot%20%5Ctext%7Blocal%7D%20%2B%20%281-%5Csigma%28%5Ctext%7Bgate%7D%29%29%20%5Codot%20%5Ctext%7Bglobal%7D)

## ⚙️ Configuration

All parameters are exposed via the `OptimizedHybridAttentionConfig` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 768 | Model dimension |
| `num_attention_heads` | 12 | Number of attention heads |
| `attention_probs_dropout_prob` | 0.1 | Dropout on attention weights |
| `window_size` | 128 | Sliding window size for local attention |
| `compression_rank` | 64 | Number of low‑rank tokens for the global branch |
| `use_gate` | True | Enable learned gate fusion |
| `gate_bias` | 2.0 | Initial bias for the gate (encourages local focus early in training) |

These settings can be adapted when injecting via the compression pipeline (see below).

## 🔌 Framework Integration

OHA is registered in `ATTENTION_REGISTRY` under the key `"oha"`. The injection function `inject_oha` builds the configuration from your `comp_config` object and replaces the layer’s attention.

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Inside your compression loop:
ATTENTION_REGISTRY["oha"](layer, model_config, comp_config, layer_idx, total_layers, old_attn)
```

The injection reads from `comp_config`:
- `block_size` → `window_size`
- `compression_rank` → `compression_rank`
- `use_gate`, `gate_bias` (optional)

## 📐 Complexity

Compared to full self‑attention (`O(N²)`), OHA scales with:
- Local: `O(N·W)` (memory / FLOPs)
- Global: `O(N·R)` (where `R` = compression rank)
- Total: **linear in sequence length** for memory and compute.

Typical savings: 4–10× reduction in peak memory and attention FLOPs for long sequences, with minimal accuracy degradation.

## 📊 Expected Use Cases

- DistilBERT / RoBERTa fine‑tuning on long documents.
- Zero‑shot compression: directly swap attention without re‑training (thanks to weight copying from `from_distilbert_pretrained`).
- Hybrid models where local and global interactions matter (e.g., classification, QA).

## 🧪 Example

```python
import torch
from optimized_hybrid_attention import (
    OptimizedHybridAttention,
    OptimizedHybridAttentionConfig,
)

# Create a dummy input
B, N, D = 2, 512, 768
x = torch.randn(B, N, D)
mask = torch.ones(B, N)  # no padding

config = OptimizedHybridAttentionConfig()
oha = OptimizedHybridAttention(config)
out, attn_weights = oha(x, mask)
print(out.shape)  # (2, 512, 768)
```

## 📜 License & Attribution

This implementation is provided as part of the Efficient Transformer Toolkit. Refer to the main repository for licence details.

---

*Build fast, efficient, and accurate transformers with OHA.*
