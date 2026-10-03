Module neuralops.tests.test_aegis_v5
====================================
Hardening test suite for aegis_v5.py (AegisAttention V5).

Covers the 9 hardening rules applied to this module:
  1. config=None safety
  2. Multi-arch weight copy: DistilBERT / RoBERTa nested / GPT-2 fused / Qwen GQA
  3. Fail-loud on missing mandatory weights
  4. Frozen base weights, trainable LoRA after copy
  5. GQA kv_repeat
  6. RoPE (position_embeddings tuple)
  7. Causal masking (decoder)
  8. output_attentions returns actual weight tensor (not None)
  9. Odd-dim _rotate_half roundtrip
 10. inject_aegis_v5 safe/fail paths
 11. Registry registration

Functions
---------

`test_config_none_no_comp_config()`
:   Should not raise even with both configs absent.

`test_config_none_uses_kwargs()`
:   

`test_copy_weights_fails_loud_on_empty_attn()`
:   

`test_decoder_causal_shape()`
:   

`test_decoder_no_future_leakage()`
:   With causal masking, all-ones input should give finite non-explosive output.

`test_distilbert_weight_copy()`
:   

`test_factorisedlinear_forward_finite()`
:   

`test_factorisedlinear_from_weight()`
:   

`test_factorisedlinear_lora_rank_zero()`
:   

`test_forward_cross_attention()`
:   

`test_forward_long_sequence_encoder()`
:   Long encoder sequence takes the local+global path (N >= local_window).

`test_forward_no_nan_inf()`
:   

`test_gpt2_fused_weight_copy()`
:   

`test_inject_aegis_v5_on_distilbert_layer()`
:   

`test_inject_aegis_v5_on_gpt2_attn_layer()`
:   

`test_inject_aegis_v5_target_none_returns_none()`
:   

`test_inject_returns_none_on_copy_failure()`
:   

`test_near_zero_weight_logs_warning(caplog)`
:   

`test_output_attentions_false_returns_none_weights()`
:   

`test_output_attentions_returns_weights()`
:   

`test_qwen_gqa_weight_copy()`
:   

`test_registry_aegis_v5_registered()`
:   

`test_registry_inject_fn_registered()`
:   

`test_roberta_nested_encoder_no_double_projection()`
:   Nested encoder must return (B, N, hidden) not (B, N, 64) twice projected.

`test_roberta_nested_encoder_weight_copy()`
:   

`test_rope_odd_head_dim()`
:   Test that RoPE + odd head_dim doesn't crash.

`test_rope_via_position_embeddings()`
:   

`test_rotate_half_even_dim()`
:   

`test_rotate_half_odd_dim()`
:   

`test_rotate_half_roundtrip_even()`
:   Applying _rotate_half twice should give -x (180° rotation).

`test_rotate_half_roundtrip_odd()`
:   For odd dims, two applications should still give -x on the rotated part.

Classes
-------

`DummyDistilBertAttn(dim=64)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyEmptyAttn()`
:   Attention with no recognizable projection attributes — should trigger fail-loud.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyGPT2Attn(dim=64)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyQwenAttn(dim=64, num_kv_heads=2, head_dim=16)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyRobertaAttn(dim=64)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`DummyRobertaSelfAttn(dim=64)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, head_dim=16, is_decoder=False, model_type='bert')`
: