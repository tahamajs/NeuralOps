# Quick Start

## Installation

```bash
pip install neuralops[cuda]  # or [cpu], [mps], [rocm]
```

## 1. Replace Attention in a Model

```python
from transformers import AutoModel
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention_in_model

model = AutoModel.from_pretrained("distilbert-base-uncased")
attn_cls = ATTENTION_REGISTRY.get("svd_linear_attention")
model = replace_attention_in_model(model, attn_cls)
```

## 2. Compress a Model

```python
from transformers import AutoModelForSequenceClassification
from neuralops.compression_engine.svd import ASVDCompressor

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
compressor = ASVDCompressor(model)
compressed = compressor.compress(target_rank=64)
```

## 3. Run a Benchmark

```bash
python -m neuralops.benchmarks run \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64
```

## 4. Inspect Results

```bash
python -m neuralops.benchmarks results --latest --summary
```
