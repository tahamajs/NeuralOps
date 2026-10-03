Module neuralops.attentions.upi_stuffed_mamba
=============================================
UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM Block.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`inject_selective_mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.upi_stuffed_mamba.SelectiveMamba | None`
:   Safe injection hook for SelectiveMamba.

`inject_upi_mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.upi_stuffed_mamba.SelectiveMamba | None`
:   Safe injection hook for SelectiveMamba.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `reset_parameters(self) ‑> None`
    :

`SelectiveMamba(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, dim: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 1, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Selective Mamba SSM block.
    
    Implements a Mamba-style selective state space model with
    input-dependent B, C, and dt parameters:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t + D · x_t
    where Ā_t = exp(dt_t · A).
    
    The "selectivity" comes from x_proj producing B_t and C_t
    from the input.
    
    NOTE: The previous name "UPI-Stuffed" has been removed
    because none of the claimed features (Uncertainty Penalty,
    Interpolation, Stuffing) were actually implemented.
    
    Registered as "upi_mamba" (legacy alias) and
    "selective_mamba" (canonical).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `step_recurrent(self, x_t: torch.Tensor, state: dict[str, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, dict[str, torch.Tensor]]`
    :   Single-step streaming inference.
        
        x_t: (B, 1, D)
        state: dict with keys 'h', 'conv_buf' (optional)
        Returns: (y_t, new_state)

`UPIMambaBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, dim: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 1, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Selective Mamba SSM block.
    
    Implements a Mamba-style selective state space model with
    input-dependent B, C, and dt parameters:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t + D · x_t
    where Ā_t = exp(dt_t · A).
    
    The "selectivity" comes from x_proj producing B_t and C_t
    from the input.
    
    NOTE: The previous name "UPI-Stuffed" has been removed
    because none of the claimed features (Uncertainty Penalty,
    Interpolation, Stuffing) were actually implemented.
    
    Registered as "upi_mamba" (legacy alias) and
    "selective_mamba" (canonical).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA.

    `step_recurrent(self, x_t: torch.Tensor, state: dict[str, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, dict[str, torch.Tensor]]`
    :   Single-step streaming inference.
        
        x_t: (B, 1, D)
        state: dict with keys 'h', 'conv_buf' (optional)
        Returns: (y_t, new_state)