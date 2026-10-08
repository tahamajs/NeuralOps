# Exhaustive Triton Kernel Audit
**Total Kernel Modules Audited:** 19

### Kernel: `autotuner`
- **File:** `neuralops/core/triton_kernels/autotuner.py`
- **Lines:** 58
- **@triton.jit functions:** None (Wrapper / Dispatcher)
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** NO
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_autotuner.py']

### Kernel: `block_sparse`
- **File:** `neuralops/core/triton_kernels/block_sparse.py`
- **Lines:** 44
- **@triton.jit functions:** None (Wrapper / Dispatcher)
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** NO
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_block_sparse.py']

### Kernel: `elementwise`
- **File:** `neuralops/core/triton_kernels/elementwise.py`
- **Lines:** 52
- **@triton.jit functions:** ['swiglu_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_elementwise_kernels.py']

### Kernel: `flash_decoding`
- **File:** `neuralops/core/triton_kernels/flash_decoding.py`
- **Lines:** 91
- **@triton.jit functions:** ['flash_decoding_stage1_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_flash_decoding.py']

### Kernel: `fms_scan`
- **File:** `neuralops/core/triton_kernels/fms_scan.py`
- **Lines:** 545
- **@triton.jit functions:** ['_fms_fwd_kernel', '_fms_bwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_fms_scan.py']

### Kernel: `fused_rope`
- **File:** `neuralops/core/triton_kernels/fused_rope.py`
- **Lines:** 60
- **@triton.jit functions:** ['fused_rope_fwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_fused_rope.py']

### Kernel: `gdn_scan`
- **File:** `neuralops/core/triton_kernels/gdn_scan.py`
- **Lines:** 503
- **@triton.jit functions:** ['_gdn_fwd_kernel', '_gdn_bwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_gdn_scan.py']

### Kernel: `gla_scan`
- **File:** `neuralops/core/triton_kernels/gla_scan.py`
- **Lines:** 482
- **@triton.jit functions:** ['_gla_fwd_kernel', '_gla_bwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_gla_scan.py']

### Kernel: `gqa`
- **File:** `neuralops/core/triton_kernels/gqa.py`
- **Lines:** 166
- **@triton.jit functions:** ['_triton_gqa_expand_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** NO
- **Associated Tests:** ['test_gqa_utils.py', 'test_gqa_comprehensive.py', 'test_triton_gqa.py', 'test_wgqa_comprehensive.py']

### Kernel: `hgrn2_scan`
- **File:** `neuralops/core/triton_kernels/hgrn2_scan.py`
- **Lines:** 254
- **@triton.jit functions:** ['_hgrn2_fwd_kernel', '_hgrn2_bwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_hgrn2_scan.py']

### Kernel: `linear_attn`
- **File:** `neuralops/core/triton_kernels/linear_attn.py`
- **Lines:** 451
- **@triton.jit functions:** ['_triton_chunked_causal_linear_attn_fwd_kernel', '_triton_bidirectional_linear_attn_fwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** NO
- **Docstring:** NO
- **Associated Tests:** ['test_triton_linear_attn.py']

### Kernel: `low_rank`
- **File:** `neuralops/core/triton_kernels/low_rank.py`
- **Lines:** 261
- **@triton.jit functions:** ['_triton_gemm_bias_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** NO
- **Associated Tests:** ['test_triton_low_rank.py']

### Kernel: `paged_attention`
- **File:** `neuralops/core/triton_kernels/paged_attention.py`
- **Lines:** 96
- **@triton.jit functions:** ['paged_attention_fwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_paged_attention_kernel.py']

### Kernel: `selective_scan`
- **File:** `neuralops/core/triton_kernels/selective_scan.py`
- **Lines:** 437
- **@triton.jit functions:** ['_selective_scan_fwd_kernel', '_selective_scan_bwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_selective_scan.py']

### Kernel: `sliding_window`
- **File:** `neuralops/core/triton_kernels/sliding_window.py`
- **Lines:** 120
- **@triton.jit functions:** ['sliding_window_attn_fwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_sliding_window_triton.py', 'test_sliding_window.py']

### Kernel: `softplus`
- **File:** `neuralops/core/triton_kernels/softplus.py`
- **Lines:** 55
- **@triton.jit functions:** ['softplus', 'softplus']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_triton_softplus.py']

### Kernel: `sparse_attn`
- **File:** `neuralops/core/triton_kernels/sparse_attn.py`
- **Lines:** 291
- **@triton.jit functions:** ['_triton_block_sparse_fwd_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** NO
- **__all__ defined:** NO
- **Docstring:** NO
- **Associated Tests:** ['test_triton_sparse_attn.py']

### Kernel: `state_update`
- **File:** `neuralops/core/triton_kernels/state_update.py`
- **Lines:** 309
- **@triton.jit functions:** ['_triton_ssm_recurrence_kernel']
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** YES
- **CUDA check:** YES
- **__all__ defined:** YES
- **Docstring:** NO
- **Associated Tests:** ['test_triton_state_update.py']

### Kernel: `varlen_attention`
- **File:** `neuralops/core/triton_kernels/varlen_attention.py`
- **Lines:** 51
- **@triton.jit functions:** None (Wrapper / Dispatcher)
- **PyTorch fallbacks:** Native PyTorch fallback implemented
- **HAS_TRITON check:** NO
- **CUDA check:** NO
- **__all__ defined:** YES
- **Docstring:** YES
- **Associated Tests:** ['test_varlen_attention.py']

