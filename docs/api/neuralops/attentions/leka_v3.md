Module neuralops.attentions.leka_v3
===================================
LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored Landmarks (Corrected O(N)).

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- FactorizedLoRALinear (parameter-reducing) with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Exact causal masking for decoders
- Chunk-wise Landmark Selection (Solves Causal Starvation)
- Block-wise Sliding Window Attention (True O(N) memory scaling)
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_leka_v3(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 6, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`leka_v3_inject(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 6, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Rectified softmax with temperature:
        p_i = ReLU(exp((x_i - m) / T) - exp(-m / T)) / sum_j ReLU(...)
    Falls back to plain softmax for degenerate rows.

Classes
-------

`FactorizedLoRALinear(in_features: int, out_features: int, rank: int = 64, lora_rank: int = 16, lora_alpha: float = 32.0, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
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

    `freeze_base_weights(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`LEKAV3Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, past_key_value: Any | None = None, layer_idx: int = 0, output_attentions: bool | None = False, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`LEKAv3Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :