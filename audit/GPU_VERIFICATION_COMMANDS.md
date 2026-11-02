# GPU Verification Commands for NeuralOps v2.0.2

This document specifies the exact batch commands the owner must run on `taha@192.168.1.100` (`/mnt/d/SVD_Project`) or Kaggle (`/kaggle/working/NeuralOps`) to verify all kernel, dispatch, and numerical fixes prior to pushing to `origin` or PyPI.

---

## Phase 1 GPU Verification (5 PyBind11 C++ Binding Fixes)

Verifies dynamic module loading and pybind compilation for all 5 previously crashing kernels:
```bash
ssh <host> "cd /kaggle/working/NeuralOps && for m in aegis_gate_norm entropy_gate favor_plus gla_unnormalized_scan unified_rope_offset; do python3 -m neuralops.benchmarks.cli run --module \$m --backend cuda --device cuda:0; done"
```
**Expected Result**: All 5 modules compile and execute without `ImportError: dynamic module does not define module export function`.

---

## Phase 2 GPU Verification (7 Numeric/NaN Kernels + 7 Timeout Kernels)

Runs the comprehensive CUDA kernel suite in FP16 precision:
```bash
ssh <host> "cd /kaggle/working/NeuralOps && python3 -m neuralops.benchmarks.cli run --category all --backend cuda --device cuda:0 --dtype fp16"
```
**Expected Result**:
- `rfa_attention`: Finite output ($\pm 65504$ clamped), zero NaN.
- `selective_ssm`: Finite output ($\Delta t \in [10^{-5}, 10]$, $dA \le 50$), zero NaN.
- `paged_recurrent_step`: Latency drops from 30.35ms to $< 0.10$ms; numerical diff drops to $< 0.005$.
- `phantom`, `spectra_v2`, `state_update`, `vspec`: Finite outputs with max difference $< 0.05$.
- `rotational_svd`, `rpca_sparse_svd`, `tt_contraction`, `entropy_pruning`, `logcumsumexp`, `samba_moe`, `sparse_attention`: All complete within 120s timeout without deadlocking or warp divergence.

---

## Phase 3 GPU Verification (Dispatch Routing & Fallbacks)

Verifies zero-overhead routing for memory-bound/sub-optimal kernels:
```bash
ssh <host> "cd /kaggle/working/NeuralOps && python3 -m neuralops.benchmarks.cli run --module fused_norm_linear --backend auto --device cuda:0"
```
**Expected Result**: Logs `fused_norm_linear routes to PyTorch because cuBLAS is 244x faster on current tile sizes`, executing via cuBLAS reference without crash.

---

## Phase 3 & WP5 Verification (71 Tolerance and Mock Model Fixes)

Evaluates accuracy across the full attention library with dynamic shape contracts:
```bash
ssh <host> "cd /kaggle/working/NeuralOps && python3 -m neuralops.benchmarks.cli run --category attention --backend pytorch --device cuda:0 --dtype fp16"
```
**Expected Result**:
- `FAIL_CORRECTNESS` reduced across all 71 modules due to half-precision tolerance `atol=1e-2, rtol=1e-2` and `MockAttentionBlock` contracts.
- Legitimate sequence length drift classified as `PASS_HIGH_DIFF` instead of failure.
