Module neuralops.attentions.salt_attention
==========================================
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.

Parameter-efficient adaptation mechanism for transformer attention projections combining
orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$).
Hardened for multi-architecture production deployment.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support.

`calibrate_gamma(model: torch.nn.modules.module.Module, calibration_data: Any, tokenizer: Any | None = None) ‑> None`
:   Run forward+backward on calibration data, capture gradients
    w.r.t. each SVD direction, and initialize gamma accordingly.

`inject_salt_attention(model: torch.nn.modules.module.Module | None = None, layer: Any | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any = None, **kwargs: Any) ‑> neuralops.attentions.salt_attention.SALTAttention | torch.nn.modules.module.Module | None`
:   Safe injection hook for SALTAttention with whole-model and layer support.

`merge_model_salt(model: torch.nn.modules.module.Module) ‑> torch.nn.modules.module.Module`
:   Traverse model and replace all SALTLinear modules with merged nn.Linear layers.
    Eliminates all LoRA / SVD overhead during inference.

Classes
-------

`SALTAttention(config: Any, rank: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, use_fisher: bool | None = None, adaptive_rank: bool | None = None, **kwargs: Any)`
:   SALT Attention module replacing standard linear projections with SALTLinear.
    Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    Supports Fisher-weighted SVD, attention-aware gamma, and adaptive rank (SALT-F).
    
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

    ### Instance variables

    `key: neuralops.attentions.salt_attention.SALTLinear`
    :

    `query: neuralops.attentions.salt_attention.SALTLinear`
    :

    `value: neuralops.attentions.salt_attention.SALTLinear`
    :

`SALTLinear(in_features: int, out_features: int, rank: int = 16, original_weight: torch.Tensor | None = None, original_bias: torch.Tensor | None = None, fisher_diag: torch.Tensor | None = None, use_fisher: bool = True)`
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
    :   Linear projection layer with Fisher-weighted SVD initialization,
        attention-aware singular value modulation (\gamma), and low-rank residual branch (A B).
        
        Formulation:
            $$W = U_{:, :r} \operatorname{diag}(\Sigma_{1:r} \odot \gamma) V_{1:r, :}^\top + A B$$

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Compute adapted linear projection efficiently in low-rank factorized form.

    `freeze_base_weights(self) ‑> None`
    :   Buffers U, S, Vt are non-trainable by construction. Gamma and A, B remain trainable.

    `init_gamma_from_sensitivity(self, sensitivity: torch.Tensor | None = None) ‑> None`
    :   Initialize gamma from attention sensitivity per SVD direction.
        If sensitivity is None, initialize to ones.

    `merge_to_linear(self) ‑> torch.nn.modules.linear.Linear`
    :   Merge SVD + gamma + LoRA into a single nn.Linear.
        After this, the layer has no LoRA overhead at inference.