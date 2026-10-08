# Troubleshooting Guide

Common issues and resolutions when deploying NeuralOps.

---

## 1. Triton Not Available on macOS / CPU

**Issue:** `ModuleNotFoundError: No module named 'triton'` or `HAS_TRITON == False`.  
**Resolution:** This is completely normal on macOS, Windows, or CPU-only Linux machines. NeuralOps provides **100% pure PyTorch fallbacks** for all 19 Triton kernels. No action is needed; your code will execute transparently using standard PyTorch autograd.

---

## 2. CUDA Out of Memory (OOM)

**Issue:** `torch.cuda.OutOfMemoryError` during evaluation sweeps.  
**Resolution:**
- Decrease the batch size: `--batch_size 16` or `--batch_size 8`.
- Enable activation checkpointing or 8-bit inference.
- Use subquadratic attention layers (`svd_linear_attention`, `cosformer`, `sliding_window_attention`).

---

## 3. Deprecation Warnings

**Issue:** `DeprecationWarning: Module alias ... is deprecated.`  
**Resolution:** Check [`docs/RENAME_MAP.md`](RENAME_MAP.md) for the canonical name and update your imports or registry keys. Legacy names remain functional throughout the v1.x series.

---

## 4. Package Installation Extras

**Issue:** Missing optional dependencies for evaluation or benchmarking.  
**Resolution:** Install the matching extras:
- For evaluation: `pip install neuralops[eval]`
- For benchmarking: `pip install neuralops[benchmark]`
- For complete setup: `pip install neuralops[all]`
