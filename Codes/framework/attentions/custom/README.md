# 🧰 Custom Compressor & Attention Templates

This folder provides **ready‑to‑extend** templates for creating new compression techniques and attention mechanisms within the Efficient Transformer Toolkit.  
You can copy, rename, and modify them to implement your own methods without reinventing the infrastructure.

## 📁 Contents

```
custom/
├── template_compressor.py    # Base compressor with low‑rank, pruning, quantization, hybrid
├── template_attention.py     # Custom attention with optional rank modulation
├── __init__.py               # Auto‑imports both modules
└── README.md                 # This file
```

---

## 🚀 Quick Start

### 1. Template Compressor

```python
from framework.compression.custom.template_compressor import YourCustomCompressor

compressor = YourCustomCompressor(
    model=your_model,
    target_sparsity=0.5,       # only used for pruning/hybrid
    target_rank=32,            # rank for low‑rank / hybrid
    method="low_rank",         # "low_rank", "pruning", "quantization", or "hybrid"
    custom_param=0.1           # extra parameter (e.g., bits for quantization)
)

compressed_model, stats = compressor.compress()
print(f"Parameters saved: {stats['parameters_saved']}")
df = compressor.get_layer_wise_report()
compressor.visualize_compression(save_path="compression.png")
```

**Available methods:**

| Method | Description |
|--------|-------------|
| `"low_rank"` | Replaces every `nn.Linear` with an SVD‑based `SVDLinearLayer` (U @ diag(S) @ V). |
| `"pruning"` | Keeps top rows/columns by L2 norm; reduces dimensions proportionally to `target_sparsity`. |
| `"quantization"` | Quantises weights uniformly to `custom_param` bits (default 8). |
| `"hybrid"` | Prunes first, then applies low‑rank factorization on the smaller matrix. |

### 2. Template Attention

```python
from framework.attentions.custom.template_attention import YourCustomAttention

config = model.config   # HuggingFace config
attn = YourCustomAttention(config, rank=32, custom_param=0.1)
output, attn_weights = attn(hidden_states, attention_mask)
```

When `rank` is set (and > 0), the attention scores are modulated by a global pooled vector passed through a low‑rank projection → sigmoid. This serves as a simple example of injecting custom logic.

---

## 🔧 Customization Guide

### Step 1: Copy and rename

```bash
cp template_compressor.py my_svd_pruner.py
cp template_attention.py my_sparse_attention.py
```

### Step 2: Change class names and registry keys

In your new file:

```python
@COMPRESSOR_REGISTRY.register("my_svd_pruner")
class MySVDPruner(YourCustomCompressor):   # or inherit from BaseCompressor
    ...
```

Same for attention:

```python
@ATTENTION_REGISTRY.register("my_sparse_attn")
class MySparseAttention(YourCustomAttention):
    ...
```

### Step 3: Implement your logic

**For a compressor**, override any of these methods:

- `compress()` – main entry point.
- `_compress_embeddings()`, `_compress_attention_layers()`, `_compress_ffn_layers()`, `_compress_classifier()` – layer‑wise compression.
- `_create_low_rank_layer()`, `_create_pruned_layer()`, etc. – your new compression primitives.

**For an attention module**, override `forward()` and replace the marked section with your mechanism. You have access to:
- `self.query`, `self.key`, `self.value`, `self.out_proj`
- `self.rank_proj` (optional low‑rank module)
- `attention_mask` and any extra kwargs.

---

## 📦 Integration with the Registry

Both templates are already registered. After you create your own subclass and register it under a new key, you can use it throughout the framework:

```python
# Replace attention in a layer
ATTENTION_REGISTRY["my_sparse_attn"](layer, model_config, comp_config, ...)

# Compress a full model
COMPRESSOR_REGISTRY["my_svd_pruner"](model, target_rank=64).compress()
```

---

## 📊 Example: Custom Low‑Rank + Sparsity Attention

Here’s a sketch of combining the two templates:

```python
@ATTENTION_REGISTRY.register("hybrid_attn")
class HybridAttention(YourCustomAttention):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # 1. Project to Q, K, V
        Q = self._reshape(self.query(hidden_states), hidden_states.size(0))
        K = self._reshape(self.key(hidden_states), hidden_states.size(0))
        V = self._reshape(self.value(hidden_states), hidden_states.size(0))
        
        # 2. Apply sparse SVD to K, V using the compressor primitives
        from framework.compression.custom.template_compressor import SVDLinearLayer
        K_svd = SVDLinearLayer(K, rank=16)
        V_svd = SVDLinearLayer(V, rank=16)
        
        # 3. Standard attention on compressed K, V
        scores = torch.matmul(Q, K_svd.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V_svd)
        ...
```

---

## 🧪 Testing Your Custom Module

Use the provided helper to quickly check your attention:

```python
import torch
from your_new_attention import YourAttention

config = type('cfg', (), {
    'hidden_size': 768,
    'num_attention_heads': 12,
    'attention_probs_dropout_prob': 0.1
})()
attn = YourAttention(config, rank=32)
x = torch.randn(2, 128, 768)
mask = torch.ones(2, 128)
out, w = attn(x, mask)
print(out.shape)   # expected (2, 128, 768)
```

---

## 📚 Reference

- BaseCompressor API: `framework.compression.core.base`
- BaseAttention API: `framework.attentions.core.base`
- Registry documentation: `framework.core.registry`

---

*Start building your own efficient transformer components!*
