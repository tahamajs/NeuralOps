Module neuralops.tests.test_fused_svd_kernel
============================================
Unit tests for Fused SVD-Linear Layer with in-SRAM RMSNorm, Autograd backward pass,
and integration into SVDLinear and SVDLinearAttention modules.

Functions
---------

`test_analytical_vs_reference_gradients() ‑> None`
:   RESEARCH FIX: Verify analytical backward gradients are 100% numerically identical
    to PyTorch autograd graph backpropagation.

`test_fused_svd_linear_forward_backward() ‑> None`
:   Verify forward and backward gradients of FusedSVDLinear with RMSNorm and bias.

`test_fused_svd_linear_gradcheck_full() ‑> None`
:   Verify analytical gradients match numerical gradients via torch.autograd.gradcheck.

`test_fused_svd_linear_gradcheck_no_norm_weight() ‑> None`
:   Verify gradcheck without norm_weight and bias.

`test_fused_svd_linear_gradcheck_no_rms_norm() ‑> None`
:   Verify gradcheck when RMSNorm is disabled.

`test_fused_svd_linear_module_methods() ‑> None`
:   Verify FusedSVDLinear helper methods: set_low_rank_factors, get_orthogonality_loss, extra_repr.

`test_fused_svd_linear_nd_shapes() ‑> None`
:   Verify arbitrary multi-dimensional input tensor shapes (2D, 4D).

`test_fused_svd_linear_no_bias() ‑> None`
:   Verify forward and backward pass without bias.

`test_fused_svd_linear_no_rms_norm() ‑> None`
:   Verify forward and backward pass when RMSNorm is disabled.

`test_svd_linear_attention_integration() ‑> None`
:   Verify SVDLinearAttention executes with use_fused_kernel=True across exact and linear modes.

`test_svd_linear_integration() ‑> None`
:   Verify SVDLinear module with use_fused_kernel=True and False.

`test_svd_linear_layer_compression_integration() ‑> None`
:   Verify SVDLinearLayer in compression engine with use_fused_kernel=True.