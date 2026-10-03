Module neuralops.tests.test_svd_audit_suite
===========================================
Comprehensive Audit & Evaluation Suite for SVD and Low-Rank Attention Mechanisms & Compression Engines.

Audited Modules:
1. attentions/svd_linear_attention.py (SVDLinear, SVDLinearAttention, exact, linear, decay gating, step_recurrent)
2. attentions/adaptive_svd_attention.py (AdaptiveSVDLinear, AdaptiveSVDAttention, energy thresholding)
3. attentions/asvd_attention.py (ASVDLinear, ASVDAttention)
4. attentions/linformer_attention.py (LinformerAttention, sequence projection)
5. compression_engine/res_svd.py (ResSVDLinear, ResSVDCompressor, SpectralResidualLoRA)
6. compression_engine/adaptive_ranks.py (AdaptiveRanksCompressor, covariance whitening, energy thresholding)
7. compression_engine/direct_svd.py (DirectSVDCompressor, symmetrical SVD)
8. compression_engine/dobi_svd.py (DobiSVDLinearLayer, DobiSVDCompressor, spectral gating)
9. compression_engine/erc_svd.py (ERCCompressor, Frobenius error bound)
10. compression_engine/entropy_budget_allocation.py (EntropyBudgetCompressor, singular entropy allocation)
11. compression_engine/kronecker_tt_lora.py (KroneckerTTLoRALayer, KroneckerTTLoRACompressor, TT cores, LoRA bypass)

Functions
---------

`get_dummy_config(hidden_size: int = 768, num_heads: int = 12, rank: int = 64)`
:   

Classes
-------

`MockMHA(hidden_size: int = 768, num_heads: int = 12)`
:   Mock standard Multi-Head Attention layer for weight copying tests.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`MockTransformerModel(hidden_size: int = 128, num_layers: int = 2)`
:   Mock multi-layer transformer model for compression engine testing.
    
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

`TestLeadSVDSpecialistAudit(methodName='runTest')`
:   Lead SVD & Factorization Specialist Comprehensive Audit Test Suite.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `test_adaptive_ranks_compressor_and_energy(self)`
    :   Audit AdaptiveRanksCompressor cumulative energy thresholding (tau=0.92) and rank floor.

    `test_adaptive_svd_attention_forward_backward_energy(self)`
    :   Audit AdaptiveSVDAttention forward/backward, cumulative energy thresholding, and residual adapter.

    `test_asvd_attention_forward_backward(self)`
    :   Audit ASVDAttention with explicit singular value vector S.

    `test_direct_svd_compressor(self)`
    :   Audit DirectSVDCompressor symmetric factor allocation U sqrt(S), sqrt(S) V^T.

    `test_dobi_svd_linear_and_compressor(self)`
    :   Audit Dobi-SVD differentiable spectral gating and forward/backward gradient flow.

    `test_entropy_budget_compressor(self)`
    :   Audit EntropyBudgetCompressor singular entropy global budget allocation.

    `test_erc_compressor_frobenius_error_bound(self)`
    :   Audit ERC-SVD bounded Frobenius reconstruction error guarantee ||W - W_r||_F / ||W||_F <= eps.

    `test_kronecker_tt_lora_forward_backward_and_adapt(self)`
    :   Audit KroneckerTTLoRALayer Kronecker product, Tensor-Train cores, LoRA bypass, and rank adaptation.

    `test_linformer_attention_forward_backward(self)`
    :   Audit LinformerAttention with sequence dimension projection matrices E and F.

    `test_res_svd_linear_and_compressor(self)`
    :   Audit ResSVDLinear residual tail initialization, forward/backward, and merge_weights.

    `test_svd_linear_attention_modes(self)`
    :   Audit SVDLinearAttention across exact, linear kernel, and decay-gated linear modes.

    `test_svd_linear_decay_step_recurrent(self)`
    :   Audit step_recurrent streaming generation vs sequential forward in DecayGatedLinearAttention.

    `test_svd_linear_forward_backward_and_stiefel(self)`
    :   Audit SVDLinear layer forward/backward, LoRA residual, and Stiefel orthogonality loss.

    `test_svd_linear_frozen_base(self)`
    :   Audit SVDLinear with frozen SVD base and active LoRA adapter.