# NeuralOps Troubleshooting Guide

Frequently encountered issues, root cause explanations, and verified resolutions.

---

## 1. CUDA Out of Memory (OOM) During Long Sequence Benchmarks

### Symptom
`torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate...`

### Resolution
1. **Reduce Batch Size / Gradient Accumulation**:
   Add `--batch-size 4` or use gradient checkpointing.
2. **Enable Low-Rank Bottlenecks**:
   Specify `--rank 64` or `--rank 32` when injecting low-rank or SVD linear projections.
3. **Empty CUDA Cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## 2. Triton Kernel Compilation Failure on Headless or CPU Environments

### Symptom
`ModuleNotFoundError: No module named 'triton'` or kernel JIT compiler failure.

### Resolution
NeuralOps imports shared Triton definitions safely from `neuralops.core.triton_kernels._types.py` without triggering hardware kernel compilation on CPU nodes.
When deploying Triton fused kernels on GPU:
```bash
pip install triton
```

---

## 3. Deprecation Warnings on Registry Access

### Symptom
`DeprecationWarning: 'adaptive_svd' is deprecated. Use 'adaptive_svd_attention' instead.`

### Resolution
NeuralOps v1.0.0 maps all 641 historical aliases to canonical names. Update your experiment configurations to use snake_case canonical identifiers (`adaptive_svd_attention`, `cosformer`, `svd_linear`).

---

## 4. Corrupted Cached Tensors

### Symptom
`RuntimeError: PytorchStreamReader failed reading zip archive...`

### Resolution
Purge corrupted tensor cache files and rebuild:
```bash
rm -rf ~/.cache/neuralops/
# or pass custom clean directory
python -m neuralops.benchmarks run --cache-dir /tmp/clean_cache
```
