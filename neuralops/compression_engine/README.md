# Compression Engine Families

Compression implementations are grouped by method family. Importing a family
package does not execute every compressor in that family. The requested class
or module is loaded on attribute access:

```python
from neuralops.compression_engine.svd import ResSVDCompressor
from neuralops.compression_engine import residual_svd_compressor
```

The existing top-level aliases and `COMPRESSOR_REGISTRY` remain available.
Family exports are discovered from source syntax trees without importing
implementation code. Do not add implementation imports to family
`__init__.py` files; doing so would defeat lazy loading and can reintroduce
cross-family import cycles.
