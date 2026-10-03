Module neuralops.tests.test_triton_svd_linear_attention_comprehensive
=====================================================================

Functions
---------

`test_svd_linear_projection_identity_init()`
:   Verify BUG 5 FIX: U is initialized with identity-like structure so initial output is non-zero.

`test_triton_svd_linear_attention_masking()`
:   Verify BUG 2 FIX: Padding tokens are zeroed on K and V without gradient issues.

`test_triton_svd_linear_bidirectional_and_causal_shapes()`
:   Verify forward pass shapes for causal and bidirectional modes.

`test_triton_svd_linear_clean_autograd()`
:   Verify BUG 1 FIX: Clean backward without double-backward warnings.

`test_triton_svd_linear_config_handling()`
:   Verify safe config handling and dimension inference.

`test_triton_svd_linear_injection_pipeline()`
:   Verify inject_triton_svd_linear_attention installs cleanly.

`test_triton_svd_linear_rope_and_gqa()`
:   Verify BUG 3 & BUG 4 FIX: RoPE and GQA support.

`test_triton_svd_linear_weight_copy_multi_arch()`
:   Verify copy_weights correctly factorizes weights from pretrained modules via SVD.