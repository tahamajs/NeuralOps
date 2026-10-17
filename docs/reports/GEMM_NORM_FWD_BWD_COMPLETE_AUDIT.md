# NeuralOps v1.4.0: GEMM & Elementwise Forward/Backward Audit Report

## Target SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM)

| Module | Fwd Status | Bwd Status | Grad Diff (FP16/FP32) | NaNs / Infs | Parity Pass |
|---|---|---|---|---|---|
| low_rank_svd | Dispatched / Analytical | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| rpca_sparse_svd | Dispatched / Analytical | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| rotational_svd | Dispatched / Analytical | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| act_scale_gemm | Dispatched / Analytical | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| hadamard_gemm | Dispatched / Analytical | Analytical Adjoint (W dequant) | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| quantized_gemm (fp8) | Dispatched / CUDA | CUDA / Analytical | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| vspec | Dispatched / CUDA | Adjoint Recurrence Scan | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| phantom | Dispatched / CUDA | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| samba_moe | Dispatched / CUDA | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| fused_ce | Dispatched / CUDA | CUDA Analytical | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| entmax_bisect | Dispatched / CUDA | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| bitonic_topk | Dispatched / CUDA | Discrete / Autograd | Exact Top-K | 0 / 0 | PASS |
| fused_norm_linear | Dispatched / CUDA | CUDA dX + Matmul dW | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| fused_dropout_add | Dispatched / CUDA | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| rms_variance_match | Dispatched / CUDA | Analytical Adjoint | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| softplus | Dispatched / CUDA | Dispatched / Autograd | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| spectra_v2 | Dispatched / CUDA | Dispatched / Autograd | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| spectral_shrinkage | Dispatched / CUDA | Dispatched / Autograd | < 1e-4 / < 0.01 | 0 / 0 | PASS |
| tt_contraction | Dispatched / CUDA | Dispatched / Autograd | < 1e-4 / < 0.01 | 0 / 0 | PASS |

### Test Suite Verification Summary
- tests/cuda_kernels/test_gemm_elementwise_autograd_parity.py: 15/15 PASSED
- tests/cuda_kernels/: 147/147 PASSED
- Zero NaNs, Zero Infs across all test shapes and layouts.
