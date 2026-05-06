
# SVD Hybrid Layer & Adaptive Compressor

Two complementary tools that further reduce model complexity by replacing standard attention with SVD‑based operations.

## Modules

| Class | Registry key | Description |
|-------|--------------|-------------|
| `SVDSingleHead` | – | Single‑head SVD approximation with positional encoding |
| `DirectSVDEncoderLayer` | – | Complete encoder layer using `SVDSingleHead` + FFN |
| `DirectSVDCompressor` | `direct_svd` | Placeholder compressor (model‑specific replacement) |
| `AdaptiveTransformerCompressor` | `adaptive_svd` | Compresses all `nn.MultiheadAttention` layers using energy‑threshold SVD |

## Usage

### 1. Use `SVDSingleHead` as an attention replacement

```python
from framework.compression.hybrid_svd_layer import SVDSingleHead

svd_head = SVDSingleHead(hidden_size=768, svd_rank=32)
output = svd_head(hidden_states)
```

### 2. Build a model from `DirectSVDEncoderLayer`

```python
from framework.compression.hybrid_svd_layer import DirectSVDEncoderLayer
from transformers import BertConfig

config = BertConfig()
config.svd_rank = 32
layer = DirectSVDEncoderLayer(config)
output, = layer(hidden_states)
```

### 3. Compress an existing model with `AdaptiveTransformerCompressor`

```python
from framework.compression import AdaptiveTransformerCompressor

compressor = AdaptiveTransformerCompressor(model, energy_threshold=0.95, min_rank=16)
compressed_model, stats = compressor.compress()
```

This compressor automatically compresses all `nn.MultiheadAttention` layers in the model (e.g., from a standard Transformer encoder) using the `CompressedAttentionLayer`.

---

*When you need extreme simplicity – SVD can replace entire attention heads.*
