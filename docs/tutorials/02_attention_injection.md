# Tutorial 2: Attention Injection into Pretrained Transformers

Dynamically swap standard quadratic self-attention layers with subquadratic variants.

```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.core.injector import AttentionInjector

# 1. Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Inject Subquadratic Galerkin Attention
injector = AttentionInjector(target_attn="galerkin_attention")
injected_model = injector.inject(model)

# 3. Verify forward pass
dummy_input = torch.randint(0, 1000, (1, 128))
output = injected_model(dummy_input)
print("Logits shape:", output.logits.shape)
```
