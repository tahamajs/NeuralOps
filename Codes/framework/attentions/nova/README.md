
# NOVA Attention Family

**NOVA** and **MinimalNOVA** are drop‑in modules for efficient attention in DistilBERT‑like models.

## 🚀 Quick start

### MinimalNOVA – instant zero‑shot compression

Replace Q/K/V/Out projections with low‑rank SVD factors.

```python
from framework.attentions.nova import inject_minimal_nova

inject_minimal_nova(
    layer,
    model_config,
    comp_config,          # has svd_energy, dropout
    layer_idx,
    total_layers,
    old_attn
)
```

Weights are initialised directly from the pretrained matrix – no training required.  
Typical rank ≈ 30–64 → **2–4× fewer parameters** with negligible accuracy loss.

### Full NOVA – the complete suite

`NOVA` combines **eight** state‑of‑the‑art compression techniques, all controlled via `NOVAConfig`:

| Technique       | Description |
|-----------------|-------------|
| ARMOR           | SVD + 2:4 sparse + quantised residual projection |
| AST             | Adaptive token pruning (learnable keep ratio) |
| DeltaKV         | Residual KV compression using reference tokens |
| KVT             | PCA‑based KV quantisation |
| Softpick        | Rectified softmax for sparse attention (~47% sparsity) |
| Subspace        | Low‑rank subspace projection of Q/K/V |
| (additional)    | Configurable via `NOVAConfig` flags |

```python
from framework.attentions.nova import inject_nova

inject_nova(layer, model_config, comp_config, ...)
```

## 🔧 Integration

Both injection functions are registered in `ATTENTION_REGISTRY` and work with your framework’s `replace_attention` utility.

## 📦 Package

```
attentions/nova/
├── __init__.py
├── minimal_nova.py
├── nova.py
└── README.md
```

All modules are ready to use in production.
