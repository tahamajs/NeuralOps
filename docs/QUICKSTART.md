# Quick Start (5 minutes)

## Step 1: Install

```bash
pip install neuralops
```

## Step 2: Verify

```bash
python -m neuralops.benchmarks verify
```

## Step 3: Run your first benchmark

```bash
python -m neuralops.benchmarks run \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64
```

## Step 4: Replace attention in your model

```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.utils.replace_attention import replace_attention

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = replace_attention(model, "svd_linear_attention", svd_rank=128)

inputs = torch.randint(0, 1000, (2, 64))
outputs = model(inputs)
print("Logits:", outputs.logits.shape)
```

## Step 5: Explore

- [API Reference](API_REFERENCE.md)
- [Examples](EXAMPLES.md)
- [Troubleshooting](TROUBLESHOOTING.md)
