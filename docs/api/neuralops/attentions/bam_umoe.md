Module neuralops.attentions.bam_umoe
====================================
BAM-UMoE: Unified Mixture of Experts Attention Block (Activation-Aware v2).

Combines:
  1. Optional SVD factorization of Q/K/V/O for actual parameter reduction.
  2. Optional Mixture-of-Experts (MoE) branch for residual refinement.
  3. Cross-architecture weight transfer (DistilBERT, RoBERTa/BERT, GPT-2, Qwen/LLaMA).
  4. LoRA adapters for recovery fine-tuning.

Fixes (v2):
  BAM-FIX-01  `UnifiedMoE.is_custom_attention = False`.
  BAM-FIX-02  MoE is OPTIONAL (`moe_enabled=False` by default). When enabled,
              it uses a compact output_dim (`moe_output_dim=head_dim`) so it
              does NOT cause expansion.
  BAM-FIX-03  **Real parameter reduction**: SVD factorization applied to
              Q/K/V/O via `svd_rank` parameter.
  BAM-FIX-04  `aux_loss` stored as attribute (None initially), never a CPU
              tensor on GPU. `get_aux_loss()` returns the correct device.
  BAM-FIX-05  Idempotency guard in injector.
  BAM-FIX-06  `apply_rotary_pos_emb` slices cos/sin for seq length.
  BAM-FIX-07  Config None-safe `__init__`.
  BAM-FIX-08  `copy_weights` device/dtype-safe.
  BAM-FIX-09  Storage accounting in stats.
  BAM-FIX-10  `head_mask` shape handling unified.
  BAM-FIX-11  `moe_gate_init` configurable; default -3.0 (sigmoid ≈ 0.047).
  BAM-FIX-12  MoE expert hidden dim is `dim // experts // 4` (compact).
  BAM-FIX-13  Route only to self-attention (documented); MoE disabled for
              cross-attention.
  BAM-FIX-14  SDPA path validates mask before use.
  BAM-FIX-15  No silent `except: pass`; errors logged.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_bam_umoe(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`BAMUMoEBlock(config: Any | None = None, rank: int = 64, num_experts: int = 2, top_k: int = 1, layer_id: int | None = None, num_layers: int | None = None, old_attn: torch.nn.modules.module.Module | None = None, comp_config: Any | None = None, moe_enabled: bool = False, svd_rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_proj`
    :

    `q_proj`
    :

    `query`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `get_aux_loss(self, device: torch.device) ‑> torch.Tensor`
    :   BAM-FIX-04: MoE aux loss for training loop.

`FactorisedLoRALinear(in_features: int, out_features: int, svd_rank: int, lora_rank: int = 0, lora_alpha: float = 32.0, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Low-rank factorized linear with optional LoRA:
    
        y = (x V S U^T) + (x lora_A^T lora_B^T) * scaling + b
    
    Base factors (U, S, V) are frozen parameters (visible in .parameters()).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :

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

    `set_svd_factors(self, W: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :

`UnifiedMoE(dim: int, num_experts: int = 2, top_k: int = 1, output_dim: int | None = None, noise_std: float = 0.1, expert_hidden_mult: float = 0.25, gate_init: float = -3.0)`
:   Compact MoE with top-k gating.
    
    BAM-FIX-02 / FIX-12: expert hidden dim is `dim // (num_experts * 4)`,
    output dim is configurable (default `head_dim`, small).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_aux_loss(self, device: torch.device) ‑> torch.Tensor`
    :   BAM-FIX-04: return aux loss on the correct device.