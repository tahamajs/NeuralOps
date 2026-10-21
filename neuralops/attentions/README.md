# Attention Families

Attention implementations are organized by family. Importing a family package
does not execute its implementation modules; the requested class or module is
imported only when that attribute is accessed.

```python
from neuralops.attentions import cosformer
from neuralops.attentions.linear import CosformerAttention
```

The top-level attention package retains the historical module aliases and the
registry remains the source of runtime registration metadata. Family exports
are discovered from Python syntax trees, so adding an implementation does not
require duplicating import statements in a family initializer. Keep
implementation imports out of family `__init__.py` files to preserve the lazy
boundary and avoid cross-family import cycles.
