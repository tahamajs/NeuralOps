Module neuralops.attentions.mamba_svd_flash
===========================================
MambaSVD-Flash v3: Hybrid Bidirectional SSM-FlashAttention Module.

Fuses bidirectional selective state-space dynamics with per-chunk online
low-rank SVD of state transitions, coupled with a Multi-Head / GQA attention
path and per-head gating.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_mamba_svd_flash(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.mamba_svd_flash.MambaSVDFlash | torch.nn.modules.module.Module | None`
:   

Classes
-------

`LoRALinear(in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False, device=None, dtype=None)`
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

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self)`
    :

    `reset_parameters(self)`
    :

    `unfreeze_base_weights(self)`
    :

`MambaSVDFlash(config: Any | None = None, rank: int | None = 16, d_model: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, bottleneck_dim: int = 64, **kwargs: Any)`
:   Hybrid SSM + MHA with per-chunk SVD and learnable per-head gating.
    
    State cache layout (for streaming):
        past_key_value = {
            "ssm_h": (B, D),           SSM hidden state h_{L-1}
            "attn_k": (B, H, L, Dh),   attention keys
            "attn_v": (B, H, L, Dh),   attention values
            "seq_len": int,
        }
    Or tuple form: (ssm_h, attn_k, attn_v, seq_len).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin`
    :

    `key`
    :

    `out_lin`
    :

    `q_lin`
    :

    `query`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `freeze_base_weights(self)`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :

    `unfreeze_base_weights(self)`
    :

`SVDS6StateSpace(d_model: int, rank: int = 16, bottleneck_dim: int = 64, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Per-token low-rank SVD state transition.
    
        U_t, Σ_t, V_t, B_t, C_t = Bottleneck(x_t)
        h_t = U_t · diag(Σ_t) · V_t · h_{t-1} + B_t · x_t
        y_t = C_t · h_t
    
    Contractive by construction (MSF-FIX-03):
      ||U_t||_2 ≤ 1, ||V_t||_2 ≤ 1, Σ_t ∈ (0, 1)   ⇒  ||UΣV||_2 ≤ 1.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.