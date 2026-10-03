Module neuralops.tests.test_adaptive_ranks_extra
================================================
Comprehensive tests for AdaptiveRanksCompressor.
Verifies:
1. Conv1D and LoRALinear wrapper detection, weight transposition, and base layer handling.
2. comp_config flexibility (None, dict, object).
3. Reconstruction error validation and fallback parameter freezing.
4. Symmetric singular value factor distribution (no norm rescaling).
5. target_attention_only logic.
6. Missing calibration data handling.
7. nn.MultiheadAttention QKV chunking and adaptive rank selection.

Classes
-------

`FakeConv1D(nf: int, nx: int)`
:   Simulates transformers.pytorch_utils.Conv1D (weight shape: [in_features, out_features]).
    
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

`FakeGPT2AttentionBlock(hidden_dim: int = 64)`
:   Simulates a transformer block with Conv1D and MLP.
    
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

`FakeLoRALinear(base_layer: nn.Module)`
:   Simulates a LoRA wrapper layer holding base_layer.
    
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

`TestAdaptiveRanksExtra(methodName='runTest')`
:   Rigorous tests for AdaptiveRanksCompressor hardening.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self) ‑> None`
    :   Hook method for setting up the test fixture before exercising it.

    `test_cfg_get_helper(self) ‑> None`
    :   _cfg_get correctly extracts keys from None, dict, and objects.

    `test_conv1d_transposition_and_compression(self) ‑> None`
    :   Conv1D layers with [nx, nf] weight are properly compressed into SVDLinear with [nf, nx].

    `test_factory_with_none_and_dict_config(self) ‑> None`
    :   adaptive_ranks_factory handles comp_config=None, dict, and object properly.

    `test_fallback_freezing(self) ‑> None`
    :   When error exceeds max_error, layer falls back and its parameters are frozen.

    `test_lora_wrapper_handling(self) ‑> None`
    :   LoRALinear wrappers with base_layer (nn.Linear or Conv1D) are correctly detected and compressed.

    `test_missing_calibration_data(self) ‑> None`
    :   Compressor gracefully proceeds with unweighted SVD when calibration data is missing.

    `test_symmetric_singular_values_no_rescaling(self) ‑> None`
    :   Factors U_r and V_r must retain symmetric sqrt(S) distribution without norm rescaling.

    `test_target_attention_only(self) ‑> None`
    :   When target_attention_only=True, attention layers are compressed while MLP/FFN are skipped.