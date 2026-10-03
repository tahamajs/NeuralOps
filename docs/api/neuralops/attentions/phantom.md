Module neuralops.attentions.phantom
===================================
Phantom: Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping – Production-Hardened.

Combines CUR-matrix factorized linear projections based on empirical statistical leverage scores
with multi-round angular Locality-Sensitive Hashing (LSH) sparse attention and rectified Softpick activation.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen CUR base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_phantom(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.phantom.PHANTOMAttention | torch.nn.modules.module.Module | None`
:   Safe injection hook for PHANTOMAttention across diverse transformer architectures.

`inject_phantom_attention(*args: Any, **kwargs: Any) ‑> neuralops.attentions.phantom.PHANTOMAttention | torch.nn.modules.module.Module | None`
:   

`lsh_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rounds: int, bucket_size: int, softpick_temp: float = 1.0, dropout: nn.Module | None = None, token_valid: torch.Tensor | None = None, is_causal: bool = False) ‑> torch.Tensor`
:   

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   

Classes
-------

`FactorisedLinear_CUR(in_features: int, out_features: int, cur_rank: int = 16, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   Linear projection compressed via leverage-score CUR matrix decomposition with LoRA support.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `copy_cur(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
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

`PHANTOMAttention(config: Any | None = None, rank: int | None = 16, phantom_cfg: PHANTOMConfig | None = None, pretrained_attn: Any | None = None, bert_mode: bool = False, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`PHANTOMConfig(cur_rank: int = 16, lsh_rounds: int = 4, lsh_bucket_size: int = 4, softpick_enabled: bool = True, softpick_temperature: float = 1.0, dropout: float = 0.1)`
:   PHANTOMConfig(cur_rank: 'int' = 16, lsh_rounds: 'int' = 4, lsh_bucket_size: 'int' = 4, softpick_enabled: 'bool' = True, softpick_temperature: 'float' = 1.0, dropout: 'float' = 0.1)

    ### Instance variables

    `cur_rank: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `lsh_bucket_size: int`
    :   The type of the None singleton.

    `lsh_rounds: int`
    :   The type of the None singleton.

    `softpick_enabled: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.