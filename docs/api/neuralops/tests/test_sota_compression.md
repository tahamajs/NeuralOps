Module neuralops.tests.test_sota_compression
============================================
Unit Tests for State-of-the-Art (SOTA) Compression Engines:
1. Dynamic Entropy SVD Compressor & Dynamic Rank Allocator (dynamic_rank.py).
2. Spectral Residual LoRA / Res-SVD Compressor (res_svd.py).
3. Central Compressor Registry & PEFT Aliases.

Classes
-------

`TestCompressorRegistrations(methodName='runTest')`
:   Test suite verifying framework registry integration for all compressors.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `test_registry_lookups(self) ‑> None`
    :   Verify dynamic_entropy_svd, entropy_rank_allocator, res_svd, and spectral_residual_lora.

`TestDynamicEntropySVD(methodName='runTest')`
:   Test suite for Dynamic Entropy SVD and rank allocation.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self) ‑> None`
    :   Hook method for setting up the test fixture before exercising it.

    `test_dynamic_entropy_svd_compressor_end_to_end(self) ‑> None`
    :   End-to-end compression of multi-layer model with calibration data.

    `test_dynamic_svd_linear_forward_and_backward(self) ‑> None`
    :   Factorized DynamicSVDLinear computes correct shapes and supports autograd.

    `test_dynamic_svd_linear_from_dense(self) ‑> None`
    :   Conversion from dense nn.Linear preserves close representation.

    `test_dynamic_svd_linear_with_covariance(self) ‑> None`
    :   Activation-aware SVD initialization with empirical covariance matrix.

    `test_rank_allocation_parameter_budget(self) ‑> None`
    :   Parameter budget water-filling produces valid ranks under target ratio.

    `test_rank_allocation_physical_dimension_clamping(self) ‑> None`
    :   Layers with small physical dimensions cannot exceed min(d_in, d_out).

    `test_rank_allocation_proportional_and_bounded(self) ‑> None`
    :   Higher entropy layers must receive higher rank, respecting [min_rank, max_rank].

    `test_singular_entropy_2d_covariance_and_weight(self) ‑> None`
    :   Compute entropy on 2D covariance and 2D rectangular weight matrices.

    `test_singular_entropy_all_zeros_stability(self) ‑> None`
    :   All-zeros matrix should return 0.0 without NaN or exceptions.

    `test_singular_entropy_normalization(self) ‑> None`
    :   Normalized singular entropy must lie strictly in [0.0, 1.0].

    `test_singular_entropy_single_spike(self) ‑> None`
    :   Singular spectrum with 1-hot spike should have zero entropy.

    `test_singular_entropy_uniform_spectrum(self) ‑> None`
    :   Uniform singular spectrum should achieve maximal entropy ln(K).

`TestResSVD(methodName='runTest')`
:   Test suite for Spectral Residual LoRA (Res-SVD).
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self) ‑> None`
    :   Hook method for setting up the test fixture before exercising it.

    `test_res_svd_compressor_end_to_end(self) ‑> None`
    :   End-to-end compression with ResSVDCompressor and parameter stats check.

    `test_res_svd_dense_base_mode(self) ‑> None`
    :   ResSVDLinear supports non-factorized base mode (factorize_base=False).

    `test_res_svd_linear_construction_and_grad_freeze(self) ‑> None`
    :   ResSVDLinear freezes base weights and enables grad on adapter matrices.

    `test_res_svd_linear_forward_and_backward(self) ‑> None`
    :   ResSVDLinear forward pass and gradient backpropagation to adapters only.

    `test_res_svd_linear_merge_weights(self) ‑> None`
    :   Weight merging produces standard nn.Linear with identical output.

    `test_spectral_residual_lora_alias_class(self) ‑> None`
    :   SpectralResidualLoRA alias class works interchangeably.

    `test_spectral_residual_tail_reconstruction_error(self) ‑> None`
    :   Res-SVD adapter initialized from tail achieves lower error than base alone.

`ToyMultiLayerModel(num_layers: int = 3, hidden_size: int = 64)`
:   Multi-layer model containing several transformer blocks.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`ToyTransformerBlock(hidden_size: int = 64, intermediate_size: int = 128)`
:   Multi-layer MLP / Attention-like block for end-to-end compressor testing.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.