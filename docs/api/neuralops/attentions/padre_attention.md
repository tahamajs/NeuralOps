Module neuralops.attentions.padre_attention
===========================================
PADRe: Polynomial Attention with Dropout Regularization.
Replaces softmax with a stable polynomial approximation, enabling faster
inference and kernelized attention across diverse transformer architectures.

Production-Hardened with:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Frobenius norm validation (>= 1e-4)
- Grouped-Query Attention (GQA) with repeat_kv
- Rotary Position Embedding (RoPE) support
- Numerical stability centering and causal lower-triangular masking for decoders
- Nested encoder detection and output bypass
- Trainable LoRA adapters with frozen base weights
- Safe 4D head mask handling
- Framework registration and single-layer injection helper

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_padre(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Single-layer and whole-model framework injection helper for PADReAttention.

`inject_padre_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Single-layer and whole-model framework injection helper for PADReAttention.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer augmented with optional trainable low-rank adaptation.
    
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

`PADReAttention(config: Any | None = None, order: int = 2, temperature: float = 1.0, pretrained_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Polynomial Attention with Dropout Regularization (PADRe).
    Replaces softmax with centered, numerically stable polynomial kernel (1 + S/order)^order.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Robust weight copying across GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, and DistilBERT.
        Validates Frobenius norm >= 1e-4.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass with polynomial attention:
        1. Linear projection to Q, K, V
        2. Optional RoPE embedding
        3. GQA head expansion
        4. Scaled polynomial scoring with max-centering for stability
        5. Causal and padding masking
        6. Context readout and projection

    `get_ortho_loss(self) ‑> Any`
    :