Module neuralops.attentions.svd_projection_attention
====================================================
SVD Projection Attention.
Each projection (Q, K, V, Out) is factored into low-rank SVD components U and V,
with trainable LoRA adapters for recovery fine-tuning.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension support.

`inject_svd_proj_attn(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, rank: int | None = None, **kwargs: Any) ‑> Any | None`
:   Safely constructs and injects SVDProjectionAttention into a model or transformer layer.

Classes
-------

`SVDProjectionAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float = 0.1, **kwargs: Any)`
:   Multi-Head Attention using Factorized SVD Linear Projections.
    Hardened for production use across diverse transformer architectures.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `key: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `out_lin: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `q_lin: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `query: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `v_lin: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    `value: neuralops.attentions.svd_projection_attention.SVDProjectionLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

`SVDProjectionLinear(in_features: int, out_features: int, rank: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True)`
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
    :   Low-Rank SVD Linear Layer with Trainable LoRA Residual Adapters.
        
        Forward computes:
        $$y = x (U V)^\top + b + \frac{\alpha}{r_{\text{lora}}} x A^\top B^\top$$
        where $U \in \mathbb{R}^{d_{\text{out}} \times r}$, $V \in \mathbb{R}^{r \times d_{\text{in}}}$
        are frozen SVD singular subspaces, and $A, B$ are trainable LoRA adapters.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective dense weight matrix.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_parameters(self) ‑> None`
    :   Initialize SVD parameters with Kaiming uniform distribution.

    `set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) ‑> None`
    :   Set precomputed U and V factors.