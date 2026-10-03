Module neuralops.attentions.soa_attn
====================================
Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension support.

`inject_soa(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_attention(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_attn(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_fused(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_fused_attn(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_fused_module(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soa_module(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

`inject_soafusedattention(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention | torch.nn.modules.module.Module | None`
:   Safely replace attention module with hardened SOAAttention across diverse transformer architectures.

Classes
-------

`FactorizedLinear(down: torch.nn.modules.linear.Linear, up: torch.nn.modules.linear.Linear)`
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

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `in_features: int`
    :

    `out_features: int`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`FactorizedLoRAUp(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0)`
:   Up-projection with frozen base weight and trainable LoRA adapter.
    
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

    `freeze_base_weights(self) ‑> None`
    :

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True)`
:   Linear projection with frozen base weights and trainable LoRA adapters.
    
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

    `freeze_base_weights(self) ‑> None`
    :

`SOAAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, k_rank: int = 64, dropout: float = 0.1, pretrained_attn: Any | None = None, old_attn: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
    
    Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention`
    :

    ### Instance variables

    `k_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `key: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `q_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `query: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `v_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `value: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight extraction and SVD factorization across all architectures.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor`
    :   Must return (output, attention_weights_or_None).

`SOAFusedAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, k_rank: int = 64, dropout: float = 0.1, pretrained_attn: Any | None = None, old_attn: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
    
    Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention`
    :

    ### Instance variables

    `k_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `key: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `q_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `query: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `v_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `value: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight extraction and SVD factorization across all architectures.