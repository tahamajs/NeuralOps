#!/bin/bash
# Run comprehensive parity tests

echo "Running parity tests..."

# Run CUDA parity tests
pytest tests/cuda_parity/test_cuda_parity.py -v
pytest tests/cuda_kernels/test_gemm_elementwise_autograd_parity.py -v

# Run Triton parity tests
pytest tests/triton_parity/ -v

# Run other parity tests
pytest tests/test_cuda_flash_gqa_parity.py -v
pytest tests/test_cuda_paged_kv_cache_parity.py -v
pytest tests/test_triton_parity.py -v
pytest tests/test_cuda_sliding_window_parity.py -v
pytest tests/test_cuda_quant_gemm_parity.py -v
pytest tests/test_cuda_selective_scan_parity.py -v
pytest tests/test_cuda_paged_attention_parity.py -v
pytest tests/test_all_48_triton_modules_parity.py -v
pytest tests/test_cuda_moe_routing_parity.py -v
pytest tests/test_cuda_sparse_attention_parity.py -v
pytest tests/test_all_131_triton_parity.py -v
pytest tests/test_create_attention_parity.py -v

echo "Parity tests completed."
