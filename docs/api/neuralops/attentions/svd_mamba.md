Module neuralops.attentions.svd_mamba
=====================================
SVD-Mamba: SVD-factorized Selective State Space Model.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_SVD_Mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safe injection hook for SVDMamba with whole-model and layer support.

`inject_svd_mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safe injection hook for SVDMamba with whole-model and layer support.

`inject_svd_mamba_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safe injection hook for SVDMamba with whole-model and layer support.

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   

Classes
-------

`SVDLinear(in_features: int, out_features: int, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   SVD-factorized linear layer with trainable LoRA adapters.
        Base weight = U @ diag(S) @ V^T.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SVDMamba(config: Any | None = None, rank: int | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 1, dt_rank: int | None = None, global_rank: int = 8, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   SVD-Mamba module with SVD-factorized projections, causal convolution, and selective SSM.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: neuralops.attentions.svd_mamba.SVDLinear`
    :

    `key: neuralops.attentions.svd_mamba.SVDLinear`
    :

    `q_lin: neuralops.attentions.svd_mamba.SVDLinear`
    :

    `query: neuralops.attentions.svd_mamba.SVDLinear`
    :

    `v_lin: neuralops.attentions.svd_mamba.SVDLinear`
    :

    `value: neuralops.attentions.svd_mamba.SVDLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.