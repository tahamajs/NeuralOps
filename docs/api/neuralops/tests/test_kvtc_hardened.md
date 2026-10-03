Module neuralops.tests.test_kvtc_hardened
=========================================
Comprehensive unit test suite for hardened KVTC (KV Cache Transform Coding / PCA-Quant) compressor.

Tests:
  1. test_weight_quantization_not_kv_cache:
     Verifies KVTCCompressor operates on model weight parameters/buffers, not runtime KV cache tensors.
  2. test_integer_quantization:
     Verifies int8 weight buffer storage, scale/zero_point buffers, and element_size() == 1.
  3. test_forward_check_success:
     Verifies forward sanity check runs and produces forward_check_success=True on a valid model.
  4. test_gqa_asymmetric_skipped:
     Verifies GQA models have asymmetric QKV projections skipped when skip_attention_qkv=True.
  5. test_percentile_clip:
     Verifies _percentile_clip clips extreme outliers.
  6. test_real_bytes_measurement:
     Verifies _estimate_actual_bytes returns realistic byte counts reflecting int8 storage.
  7. test_registry_aliases:
     Verifies both original aliases ('kvtc', 'kvtc_quantizer') and honest aliases ('pca_quant') exist.
  8. test_end_to_end_compression:
     Runs full compress() on a model and verifies output shapes, forward pass, and stats integrity.

Classes
-------

`MockGQAModel(hidden_size: int = 64)`
:   Mock model with Grouped-Query Attention (num_attention_heads > num_key_value_heads).
    
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

`TestKVTCHardened(methodName='runTest')`
:   Comprehensive test suite for hardened KVTC compressor.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `test_end_to_end_compression(self)`
    :   Runs full compress() and factory on a model, verifying output shapes and stats.

    `test_forward_check_success(self)`
    :   Verify forward sanity check runs and sets forward_check_success=True.

    `test_gqa_asymmetric_skipped(self)`
    :   Verify GQA models have asymmetric QKV projections skipped when skip_attention_qkv=True.

    `test_integer_quantization(self)`
    :   Verify int8 weight buffer storage, scale/zero_point buffers, and element_size() == 1.

    `test_percentile_clip(self)`
    :   Verify _percentile_clip clips extreme outliers.

    `test_real_bytes_measurement(self)`
    :   Verify _estimate_actual_bytes returns realistic byte counts reflecting int8 storage.

    `test_registry_aliases(self)`
    :   Verify both original aliases and honest PCA-quant aliases exist in registry.

    `test_weight_quantization_not_kv_cache(self)`
    :   Verify KVTCCompressor operates on model weight layers, not runtime KV cache tensors.

`ToyTransformerBlock(d_model: int = 64, d_ff: int = 128)`
:   Multi-layer toy transformer block for testing compression engines.
    
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