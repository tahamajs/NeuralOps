# Attention Families (`neuralops.attentions`)

> **Note (v1.6.0 Tier Migration)**: Canonical attention implementations have been physically migrated to [`neuralops.layers`](../layers/). This package (`neuralops.attentions`) is fully maintained as a backward-compatibility facade using PEP 562 lazy attribute routing.

Attention implementations are organized by family. Importing a family package does not execute its implementation modules; the requested class or module is imported only when that attribute is accessed:

```python
# Direct import from new tier architecture
from neuralops.layers.linear.cosformer import CosformerAttention

# Or via backward-compatible shim
from neuralops.attentions.linear import CosformerAttention
```

The top-level attention package retains historical module aliases and the registry remains the source of runtime registration metadata. Keep implementation imports out of family `__init__.py` files to preserve the lazy boundary and avoid cross-family import cycles.
