Module neuralops.attentions.orion
=================================
ORION: Optimized Rectified Input-aware Orthogonal Network (Corrected).

Hybrid multi-head attention architecture integrating zero-shot SVD low-rank factorized
projections, head-wise bifurcated linear kernel vs. temperature-softmax full attention,
and random Fourier features.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal cumulative scan for kernel heads and causal masking for full heads
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_orion(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.orion.ORIONAttention | torch.nn.modules.module.Module | None`
:   Safe injection hook for ORIONAttention across diverse transformer architectures.

`inject_orion_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.orion.ORIONAttention | torch.nn.modules.module.Module | None`
:   Safe injection hook for ORIONAttention across diverse transformer architectures.

`temperature_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Standard temperature softmax. Replaces the ReLU-based softpick that was
    always-zero whenever max(x) <= 0 (e.g., fully-masked rows).

Classes
-------

`FactorisedLinear(in_features: int, out_features: int, bias: bool = True, energy: float = 0.95, min_rank: int = 4, max_rank: int = 64, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

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

    `freeze_base_weights(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`ORIONAttention(config: Any | None = None, rank: int | None = None, model_config: Any | None = None, orion_cfg: ORIONConfig | None = None, pretrained_attn: Any | None = None, old_attn: Any | None = None, comp_config: Any | None = None, bert_mode: bool = False, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj: FactorisedLinear`
    :

    `dense: FactorisedLinear`
    :

    `k_lin: FactorisedLinear`
    :

    `key: FactorisedLinear`
    :

    `o_proj: FactorisedLinear`
    :

    `out_lin: FactorisedLinear`
    :

    `q_lin: FactorisedLinear`
    :

    `query: FactorisedLinear`
    :

    `v_lin: FactorisedLinear`
    :

    `value: FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`ORIONConfig(svd_energy: float = 0.95, svd_min_rank: int = 4, svd_max_rank: int = 64, num_kernel_heads: int = 8, num_full_heads: int = 4, rff_features: int = 256, softpick_temperature: float = 1.0, dropout: float = 0.1)`
:   ORIONConfig(svd_energy: 'float' = 0.95, svd_min_rank: 'int' = 4, svd_max_rank: 'int' = 64, num_kernel_heads: 'int' = 8, num_full_heads: 'int' = 4, rff_features: 'int' = 256, softpick_temperature: 'float' = 1.0, dropout: 'float' = 0.1)

    ### Instance variables

    `dropout: float`
    :   The type of the None singleton.

    `num_full_heads: int`
    :   The type of the None singleton.

    `num_kernel_heads: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `svd_energy: float`
    :   The type of the None singleton.

    `svd_max_rank: int`
    :   The type of the None singleton.

    `svd_min_rank: int`
    :   The type of the None singleton.