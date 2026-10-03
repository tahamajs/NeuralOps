Module neuralops.attentions.hgrn2_gated_deltanet
================================================
HGRN-2 Gated DeltaNet Hybrid Attention Module — Production-Hardened (v2).

Combines bidirectional/causal HGRN-2 element-wise linear recurrence with multi-head
Gated DeltaNet associative matrix memory.

Fixes (v2):
  HGD-FIX-01  is_custom_attention = True (this IS attention).
  HGD-FIX-02  Balanced gate init (+/-log(0.9/0.1) instead of 4.0).
  HGD-FIX-03  Idempotency guard in injector.
  HGD-FIX-04  No padding between branches — must be same shape.
  HGD-FIX-05  Cross-attention raises clear error.
  HGD-FIX-06  freeze_base applied after copy.
  HGD-FIX-07  Storage accounting.
  HGD-FIX-08  kwargs filtered per branch.
  HGD-FIX-09  Idempotency guard.
  HGD-FIX-10  Recursion guard in injector.
  HGD-FIX-11  LoRALinear imported from neuralops.
  HGD-FIX-12  Gate trainability controlled by flag.
  HGD-FIX-13  Mask and head_mask passed separately.
  HGD-FIX-14  Removed output shape padding/truncation hack.
  HGD-FIX-15  get_ortho_loss for framework compatibility.

Functions
---------

`inject_hgrn2_gated_deltanet(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any) ‑> Any | None`
:   

Classes
-------

`HGRN2GatedDeltaNet(config: Any | None = None, rank: int | None = None, d_model: Any | None = None, num_heads: int | None = None, comp_config: Any | None = None, old_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   HGRN-2 Gated DeltaNet Hybrid Architecture.
    
    Combines:
      1. HGRN-2 branch: fast element-wise linear recurrence.
      2. Gated DeltaNet branch: associative matrix state with Delta-rule.
      3. Learnable fusion:
         $Y = \gamma \odot Y_{hgrn} + (1-\gamma) \odot Y_{deltanet}$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `get_ortho_loss(self)`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :