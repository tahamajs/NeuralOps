# Compression Engines API Reference

NeuralOps registers **72 compression engines** unified under `neuralops.core.registry.COMPRESSOR_REGISTRY`.

## Core Categories
- **Singular Value Decomposition**: `fwsvd`, `adasvd_compressor`, `asvd_compressor`, `direct_svd`, `whitened_svd`, `fisher_svd`, `dobi_svd`.
- **Pruning**: `sparsegpt`, `wanda_real`, `flap`, `slicegpt`, `admm_pruner`.
- **Quantization**: `gptq_real`, `smoothquant_real`, `quarot`, `w4a16`, `rotational_slicegpt_int4`.
- **Tensor Decompositions**: `kronecker_compressor`, `tt_compressor`, `rpca_compressor`, `tucker_compressor`.

## Usage

```python
import neuralops

# Run model compression with rank budget
compressed = neuralops.compress(model, method="adasvd_compressor", rank=32)
```
