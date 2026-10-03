Module neuralops.attentions.selective_ssm
=========================================
Selective State Space Model (SelectiveSSM).

Faithful implementation of Mamba-style selective state space recurrence block.
Features:
- State dimension (B, D, L, N)
- Selective (input-dependent) Delta, B, C parameters
- Structured A matrix initialization (Mamba paper Eq. 11)
- Depthwise 1D convolution with SiLU gating
- Bidirectional support with independent forward and backward parameters
- No internal LayerNorm duplication

Functions
---------

`inject_selective_ssm(layer=None, model=None, model_config=None, old_attn=None, **kwargs)`
:   

`inject_simple_ssm(layer=None, model=None, model_config=None, old_attn=None, **kwargs)`
:   

`inject_simple_ssm_attention(layer=None, model=None, model_config=None, old_attn=None, **kwargs)`
:   

Classes
-------

`SelectiveSSM(config: Any | None = None, rank: int | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, lora_rank: int = 0, weights_init: str = 'mamba', **kwargs: Any)`
:   Faithful Mamba-style selective SSM block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :

`SimpleSSM(config: Any | None = None, rank: int | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, lora_rank: int = 0, weights_init: str = 'mamba', **kwargs: Any)`
:   Faithful Mamba-style selective SSM block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.