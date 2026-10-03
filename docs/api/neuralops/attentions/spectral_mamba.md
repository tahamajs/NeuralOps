Module neuralops.attentions.spectral_mamba
==========================================
Spectral-Mamba Attention Module: Hybrid Attention-State Space Module with Spectral Channel Modulation – Production-Hardened.

Combines pretrained multi-head attention with a Spectral Mamba SSM branch via learned adaptive gating.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
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

`inject_spectral_mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safely replace attention in model or layer with SpectralMambaAttention.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `reset_parameters(self) ‑> None`
    :

`Mamba(d_model: int = 768, **kwargs: Any)`
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

    ### Methods

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SpectralMamba(d_model: int = 768, d_state: int = 16, **kwargs: Any)`
:   Spectral modulation with actual frequency filtering and low-rank state projection.
    
    Uses a learned frequency mask over FFT bins followed by low-rank adaptation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Spectral modulation with actual frequency filtering.

`SpectralMambaAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.spectral_mamba.SpectralMambaAttention`
    :

`SpectralMambaConfig(hidden_size: int = 768, num_attention_heads: int = 12, head_dim: int = 64, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, gate_init_bias: float = 1.8)`
:   SpectralMambaConfig(hidden_size: 'int' = 768, num_attention_heads: 'int' = 12, head_dim: 'int' = 64, d_state: 'int' = 16, d_conv: 'int' = 4, expand: 'int' = 2, dropout: 'float' = 0.1, gate_init_bias: 'float' = 1.8)

    ### Instance variables

    `d_conv: int`
    :   The type of the None singleton.

    `d_state: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `expand: int`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.