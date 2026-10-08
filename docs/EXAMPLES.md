# NeuralOps Usage Examples

End-to-end practical walkthroughs for benchmarking, attention surgery, compression, and custom module registration.

---

## 1. Unified Benchmark Run

Execute a comprehensive sweep across all registered attention mechanisms on SST-2 with DistilBERT:

```bash
python -m neuralops.benchmarks run \
    --all-attention \
    --model distilbert \
    --dataset sst2 \
    --device cuda \
    --batch-size 32 \
    --epochs 3
```

---

## 2. In-Memory Attention Surgery

Replace vanilla quadratic self-attention in a pretrained transformer with subquadratic attention:

```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.evaluation.eval_all_models_datasets import apply_module_to_model

# 1. Load HuggingFace model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Inject target linear attention module
model, stats = apply_module_to_model(
    model,
    module_name="cosformer",
    module_type="attention",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

print(f"Surgery complete: {stats}")
```

---

## 3. Post-Training Model Compression

Apply low-rank SVD compression to linear projections:

```python
import torch
from transformers import AutoModelForCausalLM
from neuralops.evaluation.eval_all_models_datasets import apply_module_to_model

model = AutoModelForCausalLM.from_pretrained("gpt2")

compressed_model, stats = apply_module_to_model(
    model,
    module_name="svd_linear",
    module_type="compressor",
    rank=64,
)
print("Compression statistics:", stats)
```

---

## 4. Registering a Custom Module

Define and register your own attention layer into NeuralOps:

```python
import torch
import torch.nn as nn
from neuralops.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("custom_fast_attention", aliases=["cfa"])
class CustomFastAttention(nn.Module):
    supports_causal = True
    supports_gqa = False
    supports_rope = False

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Custom subquadratic kernel or linear formulation
        return self.out_proj(x)
```
