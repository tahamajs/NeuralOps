Module neuralops.attentions.asvd_attention
==========================================
ASVD (Activation-Aware Singular Value Decomposition) Attention.
Paper: https://arxiv.org/abs/2307.07382

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) to query and key tensors.

`inject_asvd(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`ASVDAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA.
    
    Computational Complexity:
        - Factorized Projections: $\mathcal{O}(B \cdot N \cdot r \cdot (d + d_{\text{all}}))$
        - Attention Scoring: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$
    
    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `dense: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `k_lin: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `k_proj: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `o_proj: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `out_lin: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `q_lin: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `q_proj: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `v_lin: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    `v_proj: neuralops.attentions.asvd_attention.ASVDLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Extracts and factorizes pretrained weights across DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_ids: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass for factorized ASVD attention.

`ASVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, lora_rank: int = 0, freeze_base: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
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
    :   Low-rank factorized linear layer with explicit singular value scaling and trainable LoRA adapters:
        
        $$y = ((x V^\top) \odot \Sigma) U^\top + (x A^\top) B^\top + b$$

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for inspection & compatibility.

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
    :   Freeze U, S, V and bias parameters so only LoRA adapters adapt during recovery fine-tuning.

    `reset_parameters(self) ‑> None`
    :

    `set_identity(self) ‑> None`
    :   Initializes factorized linear transformation to act as an identity pass-through.

    `set_svd_factors(self, U: torch.Tensor, V: torch.Tensor, S: torch.Tensor | None = None, bias: torch.Tensor | None = None) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :   Unfreeze base weights for standard end-to-end training.