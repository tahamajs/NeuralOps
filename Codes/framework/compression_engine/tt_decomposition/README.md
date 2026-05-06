
# 📦 Tensor Train (TT) & Kronecker (SoKA) Compression

Parameter‑efficient replacements for `nn.Linear` using **Tensor Train decomposition** and **Kronecker product (SoKA) approximation**.  
Both methods dramatically reduce model size while preserving accuracy and are fully integrated into the framework’s compression registry.

## 🧠 Overview

| Method | Key Idea | Parameter Reduction | Registry Key |
|--------|----------|--------------------|---------------|
| **Tensor Train (TT)** | Decomposes a weight matrix into a chain of small 3‑D tensors (cores). | 10–100× | `"tt"` |
| **SoKA (Kronecker)** | Approximates weight as sum of Kronecker products, realized via SVD. | 5–50× | `"soka"` |

Both are accessible as **standalone layers** (`TTLinearLayer`) and as **full‑model compressors** (`TTCompressor`, `SoKACompressor`).  
The compressors inherit from `BaseCompressor`, support pretrained weight transfer, and return detailed compression statistics.

## 📂 File Structure

```
framework/compression/
├── core/
│   ├── base.py                # BaseCompressor
│   └── registry.py            # COMPRESSOR_REGISTRY
├── tt_linear.py               # TTLinearLayer (with SVD fallback)
├── tt_compressor.py           # TTCompressor (registered as "tt")
├── soka_compressor.py         # SoKACompressor (registered as "soka")
└── __init__.py
```

## 🚀 Quick Start

### 1. Replace a single linear layer

```python
from framework.compression import TTLinearLayer
import torch.nn as nn

original = nn.Linear(768, 768)
compressed = TTLinearLayer.from_linear(original, tt_rank=16)
```

### 2. Compress an entire model

```python
from framework.compression import TTCompressor
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")
compressor = TTCompressor(model, tt_rank=16)
compressed_model, stats = compressor.compress()

print(f"Params saved: {stats['parameters_saved']}")
print(f"Compression ratio: {stats['compression_ratio']:.2f}")
```

### 3. Using the registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# TT
TTCompressorClass = COMPRESSOR_REGISTRY.get("tt")
compressor = TTCompressorClass(model, tt_rank=32)

# SoKA
SoKACompressorClass = COMPRESSOR_REGISTRY.get("soka")
compressor = SoKACompressorClass(model, rank=8)
```

## ⚙️ Installation

For true Tensor Train decomposition (higher quality), install `torchtt`:

```bash
pip install torchtt
```

If `torchtt` is not installed, `TTLinearLayer` automatically falls back to a rapid SVD decomposition, providing identical parameter savings with a slight accuracy trade‑off.

## 📊 Parameter Savings

### Example: 768 × 768 linear layer (590k parameters)

| Method | Rank | Parameters (approx) | Reduction |
|--------|------|--------------------|-----------|
| Original | – | 590,000 | 1× |
| TT (torchtt) | 16 | ~24,000 | 24× |
| TT (SVD fallback) | 16 | ~24,000 | 24× |
| SoKA | 8 | ~12,000 | 48× |

*Note: TT with `torchtt` can go even lower if the matrix has strong low‑rank structure across multiple dimensions.*

## 🧪 Using the CLI

Both compressors are available via the command‑line interface:

```bash
# Compress with Tensor Train
python cli.py compress --method tt --target-rank 16 --model distilbert-base-uncased -o tt_compressed

# Compress with SoKA
python cli.py compress --method soka --target-rank 8 --model bert-base-uncased -o soka_compressed
```

## 📈 Comparison

| Feature | TT (torchtt) | TT (SVD fallback) | SoKA |
|---------|-------------|-------------------|------|
| Decomposition | True TT (multi‑linear) | Low‑rank matrix | Low‑rank matrix |
| Parameter reduction | Highest | High | High |
| Accuracy preservation | Excellent | Very good | Good |
| Speed (forward) | Fast | Fast | Fastest |
| External dependency | `torchtt` | None | None |
| Multi‑dimensional structure | Yes | No | No |

- **TT with torchtt** is recommended when you need maximal compression and can install the package.
- **TT SVD fallback** is a robust zero‑dependency alternative.
- **SoKA** is extremely lightweight and ideal for rapid experiments.

## 🛠️ Customization

You can create your own compressor by following the same pattern:

1. Inherit from `BaseCompressor`.
2. Implement `compress()` returning `(model, stats)`.
3. Decorate the class with `@COMPRESSOR_REGISTRY.register("your_key")`.

The new compressor will automatically appear in the CLI and any script that uses the registry.

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.