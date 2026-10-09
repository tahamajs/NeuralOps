# Tutorial 5: Registering Custom Attention & Compression Modules

Extend the NeuralOps registry with your own custom architecture.

```python
import torch
import torch.nn as nn
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("my_custom_linear_attention")
class MyCustomAttention(BaseAttention):
    def __init__(self, config=None):
        super().__init__(config)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # O(N) linear transformation
        return self.proj(hidden_states), None

# Test retrieval from registry
custom_attn_cls = ATTENTION_REGISTRY["my_custom_linear_attention"]
instance = custom_attn_cls()
out, _ = instance(torch.randn(2, 32, instance.hidden_size))
print("Custom attention verified:", out.shape)
```
