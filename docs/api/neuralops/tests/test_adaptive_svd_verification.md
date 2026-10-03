Module neuralops.tests.test_adaptive_svd_verification
=====================================================
Verification test script for AdaptiveSVDAttention and AdaptiveRanksCompressor.
Validates:
1. Dynamic cumulative singular value energy thresholding:
   sum_{i=1}^r sigma_i^2 / sum_{j=1}^d sigma_j^2 >= tau (tau = 0.92)
2. Symmetrical sqrt(Sigma) singular value distribution:
   U_r = U[:, :r] * sqrt(Sigma_r), V_r = sqrt(Sigma_r) * V_r^T
3. FP32 and FP16 forward and backward passes on dummy tensors x = torch.randn(2, 128, 768)
4. Numerical stability (no NaNs, no Infs, finite gradients).

Functions
---------

`test_adaptive_ranks_compressor_fp32_fp16()`
:   Test AdaptiveRanksCompressor on mock transformer model in FP32 and FP16.

`test_adaptive_ranks_factory()`
:   Verify adaptive_ranks_factory functionality.

`test_adaptive_svd_attention_fp32_fp16()`
:   Test AdaptiveSVDAttention with dummy tensor [2, 128, 768] in FP32 and FP16.

`test_energy_thresholding_math()`
:   Verify exact formula sum(s[:r]^2)/sum(s^2) >= tau.

`test_symmetric_sqrt_sigma()`
:   Verify symmetrical sqrt(Sigma) distribution reconstruction fidelity.

Classes
-------

`MockMHAModule(hidden_size: int = 768)`
:   Mock standard PyTorch Multihead Attention module for weight transfer.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`MockTransformerModel(hidden_size: int = 768, num_heads: int = 12, intermediate_size: int = 3072)`
:   Mock multi-layer transformer model with 768 hidden size and realistic spectral decay.
    
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