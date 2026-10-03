Module neuralops.tests.test_adaptive_ranks_jg
=============================================
Unit tests for Adaptive Ranks Compressor with Jacobian-Gated SVD (JG-SVD).
Tests all 9 core properties:
1. GELU derivative squared at y=0 is 0.25 (phi=0.5, pdf=0, prime=0.5 -> prime^2=0.25)
2. GELU derivative squared at y=5 is ~1.0 (saturation in active region)
3. GELU derivative squared at y=-5 is ~0.0 (saturation in inactive region)
4. SiLU derivative squared at y=0 is 0.25 (sig=0.5, sig*(1+0)=0.5 -> 0.25)
5. Activation detection: DistilBERT ('ffn.lin1') -> 'gelu'
6. Activation detection: Qwen ('gate_proj', 'up_proj') -> 'silu'
7. Attention Q/K detection: 'q_lin', 'k_lin', 'q_proj', 'k_proj' detected, 'out_lin'/'o_proj' not detected
8. Backward compatibility: use_jg_svd=False runs standard adaptive ranks compression identically
9. Opt-in JG-SVD: use_jg_svd=True initializes, accepts flags, and performs JG-weighted compression

Functions
---------

`test_1_gelu_derivative_at_zero()`
:   GELU derivative squared at y=0 should equal 0.25.

`test_2_gelu_derivative_at_positive()`
:   GELU derivative squared at y=5 should saturate near 1.0.

`test_3_gelu_derivative_at_negative()`
:   GELU derivative squared at y=-5 should saturate near 0.0.

`test_4_silu_derivative_at_zero()`
:   SiLU derivative squared at y=0 should equal 0.25.

`test_5_activation_detection_distilbert()`
:   Detect activation type maps DistilBERT ffn.lin1 and intermediate.dense to gelu.

`test_6_activation_detection_qwen()`
:   Detect activation type maps Qwen/LLaMA SwiGLU projections to silu.

`test_7_attention_qk_detection()`
:   Detect attention Q and K layers for softmax Jacobian scalar weighting.

`test_8_backward_compatibility_jg_off()`
:   Default compressor with use_jg_svd=False runs without error and leaves used_jg_svd False.

`test_9_opt_in_jg_svd_on()`
:   Opt-in compressor with use_jg_svd=True factorizes with JG weighting.