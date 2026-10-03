Module neuralops.attentions.aegis_v2
====================================
Aegis-Attention V2 (Corrected).

Hybrid low-rank factorized attention with:
  * SVD + LoRA projections (FactorisedLinear)
  * Local sliding-window attention over all heads
  * Global ELU+1 linear kernel attention with adaptive soft-pruning gate
  * Data-dependent multi-head fusion gate

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Apply RoPE. cos/sin must already be sliced to the current window.

`inject_aegis_into_distilbert_v2(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_aegis_v1(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_aegis_v2(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_aegis_v3(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_aegis_v4(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)`
:   AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)

    ### Instance variables

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

`AegisConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)`
:   AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)

    ### Instance variables

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

`AegisAttentionV2(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV1(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV3(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV4(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`FactorisedLinear(in_features: int, out_features: int, rank: int = 384, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, energy: float = 0.95, **kwargs: Any)`
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

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, energy: float | None = None) ‑> None`
    :

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :