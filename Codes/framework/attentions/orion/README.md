
# ORION — Optimized Rectified Input‑aware Orthogonal Network

**Hybrid attention with SVD‑factorised projections, kernelised linear attention, and Softpick.**

ORION replaces DistilBERT’s standard self‑attention with a zero‑shot compressed variant that combines parameter efficiency and sparse attention.

## 🚀 Quick Start

```python
from orion import ORIONAttention, ORIONConfig
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Inject via registry
ATTENTION_REGISTRY["orion"](layer, model_config, comp_config, layer_idx, total_layers, old_attn)
```

Or instantiate directly:

```python
cfg = ORIONConfig(svd_energy=0.95, num_kernel_heads=8, num_full_heads=4)
attn = ORIONAttention(model_config, cfg, pretrained_attn=old_attn)
```

## 🧠 Architecture

ORION splits the attention heads into two groups:

1. **Kernel heads** – use **Random Fourier Features** to perform linearised (causal) attention, scaling as O(N·F).
2. **Full heads** – use standard scaled dot‑product attention with **Softpick**, a sparse‑friendly rectified softmax.

All Q/K/V/Out projections are **SVD‑factorised** for a 2‑4× reduction in parameter count.

### Head Split
- `num_kernel_heads` heads compute efficiently with RBF‑based linear attention.
- `num_full_heads` heads attend over the full sequence with Softpick for high‑quality alignment.

## ⚙️ Configuration

All settings live in `ORIONConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy` | 0.95 | Energy retained in SVD truncation |
| `svd_min_rank` / `max_rank` | 4 / 64 | Rank bounds for factorised projections |
| `num_kernel_heads` | 8 | Number of linear‑attention heads |
| `num_full_heads` | 4 | Number of full‑attention heads |
| `rff_features` | 256 | Dimension of random Fourier features |
| `softpick_temperature` | 1.0 | Temperature for Softpick activation |
| `dropout` | 0.1 | Dropout on attention weights |

## 📦 Integration

ORION is registered in `ATTENTION_REGISTRY` under `"orion"`. The injection function reads `comp_config` for `svd_energy`, `rff_features`, and `dropout`.

## 🧪 Example Usage

```python
import torch
from orion import ORIONAttention, ORIONConfig

dummy_config = type('config', (), {
    'hidden_size': 768,
    'num_attention_heads': 12
})()
old_attn = DistilBertAttention(dummy_config)  # from HuggingFace

cfg = ORIONConfig(num_kernel_heads=8, num_full_heads=4)
orion = ORIONAttention(dummy_config, cfg, pretrained_attn=old_attn)
x = torch.randn(2, 128, 768)
out, _ = orion(x)
```

---

*Part of the Efficient Transformer Toolkit – drop‑in acceleration for your NLP models.*
