Module neuralops.attentions.minimal_svd
=======================================

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_minimal_aurora(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_minimal_prism(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_minimal_svd(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_minimal_svd_v2(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`FactorisedLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, energy: float = 0.95, lora_rank: int = 16, lora_alpha: float = 32.0)`
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
    :   SVD-Factorized Linear Transformation for NOVA Architecture.

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

`MinimalSVDAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalAuroraAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalPRISMAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalSVDAttentionV2(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :