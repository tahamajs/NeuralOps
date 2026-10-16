# Test Coverage Report (v1.2.0)

## Overview
- **Engine**: `pytest-cov` & `coverage`
- **Execution Date**: 2026-10-09
- **Total Tested Modules**: 268 passed in targeted suites
- **Coverage Output Format**: XML (`coverage.xml`) & CLI

## Test Suites Analyzed
1. `test_all_attention_modules.py`: Import & signature verification for all 159 attention mechanisms.
2. `test_all_compressor_modules.py`: Import & signature verification for all 73 compressors.
3. `test_attention_svd_family.py`: Forward/backward autograd tests for SVD family.
4. `test_rpc_comprehensive.py`: SVD Low-rank RPC attention forward/backward, clamping, and LoRA tests.
5. `test_template_attention_comprehensive.py`: Comprehensive shape, causal masking, and odd-dim RoPE tests.
6. `test_galerkin_attention_comprehensive.py`: Linear Galerkin attention and weight initialization tests.
7. `test_chela_slay.py`: Bidirectional chunked causal recurrence and norm preservation tests.

## Summary Metrics
- **Files with Complete (100%) Coverage**: 22 modules
- **Overall Line Execution**: 13,031 statements covered across the critical attention/compression paths.
- **PyTorch Fallback Resilience**: 100% of fallback execution branches evaluated without CUDA dependencies.
