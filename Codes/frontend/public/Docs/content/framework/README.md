# 🧠 SVD Linear Attention Framework Core

The core PyTorch library providing **118 attention architectures**, **80 compression engines**, **22 benchmark datasets**, and **5 visualization studio engines**.

---

## 🏛️ Directory Layout

```
Codes/framework/
├── attentions/             # 118 Attention module PyTorch nn.Modules & docs
├── compression_engine/     # 80 Compression engines (SVD, CUR, TT, SliceGPT, DeltaKV)
├── core/                   # ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, MultiModeBenchmark
├── data/                   # Benchmark datasets (WikiText-2, C4, PTB, PG-19, GLUE)
├── visualization/          # 5 Studio backend engines (Attention Canvas, Studios)
├── tests/                  # 69 Unit tests & end-to-end integration tests
├── run_experiments.py      # Experiment runner wrapper
└── test_all_framework.py   # 6-Stage master validation test suite
```

---

## 🚀 Key Framework APIs

### 1. Registering & Instantiating Attention Modules
```python
from framework.core.registry import ATTENTION_REGISTRY
import framework.attentions  # triggers auto-registration

# List all 118 registered attention keys
print(list(ATTENTION_REGISTRY.keys()))

# Instantiate SVD Linear Attention
svd_attn_cls = ATTENTION_REGISTRY.get("svd_linear")
attn_module = svd_attn_cls(d_model=768, n_heads=12, rank=32)

# Forward pass
import torch
x = torch.randn(2, 512, 768)
out = attn_module(x)  # Shape: [2, 512, 768]
```

### 2. Loading Benchmark Datasets
```python
from framework.data.benchmark_datasets import load_benchmark_dataset

# Load WikiText-2 validation dataloader
val_loader = load_benchmark_dataset(
    "wikitext2",
    seq_len=2048,
    batch_size=8,
    split="validation",
    max_samples=1000
)
```

### 3. Multi-Mode Benchmark Execution
```python
from framework.core.multi_mode_benchmark import MultiModeBenchmarkRunner

runner = MultiModeBenchmarkRunner(device="cuda" if torch.cuda.is_available() else "cpu")
result = runner.benchmark_attention("svd_linear", seq_len=4096, batch_size=4)
print(f"Latency: {result['latency_ms']:.2f} ms | VRAM: {result['peak_vram_mb']:.1f} MB")
```

---

## 🧪 Testing & Verification
```bash
# Run 69 unit tests
PYTHONPATH=Codes/framework:Codes python3 -m unittest discover -s Codes/framework/tests

# Run 6-stage master integration test
PYTHONPATH=Codes python3 Codes/framework/test_all_framework.py
```