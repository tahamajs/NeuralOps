# Custom Attention Tutorial

Register your custom attention module with `@ATTENTION_REGISTRY.register`:

```python
import torch
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("my_attention")
class MyAttention(BaseAttention):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None
```
