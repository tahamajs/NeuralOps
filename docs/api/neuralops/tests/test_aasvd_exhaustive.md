Module neuralops.tests.test_aasvd_exhaustive
============================================
Exhaustive Verification Suite for AASVD (Activation-Aware SVD) Compressor.
Validates structural factorizations, streaming covariance estimation,
multi-architecture compression, and gradient backpropagation.

Classes
-------

`TestAASVDCompressorEndToEnd()`
:   

    ### Methods

    `test_compress_distilbert_unweighted(self)`
    :

    `test_compress_qwen_decoder(self)`
    :

    `test_compress_with_simulated_covariance(self)`
    :

    `test_registry_resolution(self)`
    :

`TestSVDLinearLayer()`
:   

    ### Methods

    `test_svd_linear_backward_gradients(self)`
    :

    `test_svd_linear_factor_assignment(self)`
    :

    `test_svd_linear_initialization_and_forward(self)`
    :

    `test_svd_linear_parameter_reduction(self)`
    :