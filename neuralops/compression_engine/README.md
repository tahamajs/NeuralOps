# Compression Engine Families (`neuralops.compression_engine`)

> **Note (v1.6.0 Tier Migration)**: Canonical compression implementations have been physically migrated to [`neuralops.compression`](../compression/). This package (`neuralops.compression_engine`) is fully maintained as a backward-compatibility facade using PEP 562 lazy attribute routing.

Compression implementations are grouped by method family. Importing a family package does not execute every compressor in that family. The requested class or module is loaded on attribute access:

```python
# Direct import from new tier architecture
from neuralops.compression.svd.res_svd import ResSVDCompressor

# Or via backward-compatible shim
from neuralops.compression_engine.svd import ResSVDCompressor
```

The top-level aliases and `COMPRESSOR_REGISTRY` remain available. Keep implementation imports out of family `__init__.py` files to preserve the lazy boundary and avoid cross-family import cycles.
