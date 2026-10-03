Module neuralops.tests.test_triton_kernels_suite
================================================
Comprehensive Unit Test Suite for Shared Triton Kernels & Unified Dispatcher.

Validates:
  1. linear_attn (causal, non-causal, phi functions, backward gradients)
  2. sparse_attn (block-sparse mask, forward/backward)
  3. low_rank (analytical backward, LoRA-style projection)
  4. state_update (SSM recurrence forward/backward)
  5. gqa (grouped query repeat/expansion)
  6. triton_dispatch (correct execution or graceful PyTorch fallback)

Functions
---------

`test_gqa_fwd_and_fallback()`
:   

`test_linear_attn_fwd_bwd(is_causal, phi)`
:   

`test_low_rank_frozen_input_autograd_safety()`
:   Ensure no autograd crash when x has requires_grad=False.

`test_low_rank_fwd_and_analytical_bwd()`
:   

`test_sparse_attn_block_mask()`
:   

`test_state_update_recurrence()`
:   

`test_triton_dispatch_linear_attn()`
:   

`test_triton_dispatch_low_rank()`
:   

`test_triton_dispatch_unsupported_graceful_error()`
: