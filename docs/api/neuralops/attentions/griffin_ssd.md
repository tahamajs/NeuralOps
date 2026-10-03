Module neuralops.attentions.griffin_ssd
=======================================
Griffin SSD: Hybrid Local Sliding-Window Attention and Global State Space Duality (SSD) Kernel.

Combines bidirectional/causal local sliding-window self-attention with multi-head
State Space Duality linear recurrence, LoRA residual adapters, and multi-architecture pretrained weight transfer.
Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_griffin_ssd(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, **kwargs: Any) ‑> Any | None`
:   Injection hook for GriffinSSDBlock supporting single layers and whole models.

Classes
-------

`GriffinSSDBlock(config: Any | None = None, rank: int | None = 64, window_size: int = 64, d_model: int | Any | None = None, num_heads: int | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any)`
:   Griffin Hybrid Block combining local window attention with global SSD linear attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights across multiple model families.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass for Griffin SSD attention.