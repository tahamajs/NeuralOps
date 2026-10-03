Module neuralops.tests.test_qwen3_5_compressor_awareness
========================================================
Comprehensive test suite verifying architecture-aware compression, dynamic rank selection,
recovery parameter selection, and LoRA preservation on Qwen3.5 and across model families.

Functions
---------

`test_compressors_skip_qwen3_5_sensitive_layers()`
:   Verify that actual compressor instances skip sensitive layers on Qwen3.5.

`test_dynamic_rank_selection()`
:   Verify dynamic rank computes min(256, max(128, min(in_f, out_f) // 4)) for Qwen3.5.

`test_get_recovery_params()`
:   Verify get_recovery_params unfreezes only LoRA adapters and classification heads, freezing all else.

`test_lora_preservation_and_gradient_flow()`
:   Verify that LoRA adapters on frozen backbones receive non-zero gradients after compressor injection.

`test_qwen3_5_architecture_aware_layer_filtering()`
:   Verify should_compress_linear correctly protects sensitive layers on Qwen3.5 while allowing FFN.

`test_qwen3_5_capabilities_and_compatibility()`
:   Verify capability flags for Qwen3.5 hybrid architecture in registry.

`test_standard_family_layer_filtering_preserved()`
:   Verify standard models (BERT, RoBERTa, GPT) preserve their original compression filtering.

Classes
-------

`DummyQwen35Block(hidden_size: int = 1024, ffn_dim: int = 3584)`
:   Synthetic Qwen3.5 hybrid block containing DeltaNet, GQA, MoE/FFN, and Norms.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyQwen35Model(hidden_size: int = 1024, ffn_dim: int = 3584, num_layers: int = 2)`
:   Full dummy Qwen3.5 architecture.
    
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

`DummyStandardTransformer(family: str = 'bert', hidden_size: int = 64)`
:   Standard BERT / RoBERTa / GPT model.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module