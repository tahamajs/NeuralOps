Module neuralops.attentions
===========================
Auto-import all attention modules and export their registered classes.

Sub-modules
-----------
* neuralops.attentions.adaptive_span_attention
* neuralops.attentions.adaptive_sparse_attention
* neuralops.attentions.adaptive_svd_attention
* neuralops.attentions.adasplash_attention
* neuralops.attentions.aegis_hybrid
* neuralops.attentions.aegis_spectron_fp8
* neuralops.attentions.aegis_v2
* neuralops.attentions.aegis_v5
* neuralops.attentions.aegis_v6
* neuralops.attentions.asvd_attention
* neuralops.attentions.bam_umoe
* neuralops.attentions.block_transformer_hiba
* neuralops.attentions.chela_slay
* neuralops.attentions.cluster_attention
* neuralops.attentions.compressed_attention
* neuralops.attentions.compressed_mha_attention
* neuralops.attentions.cosformer
* neuralops.attentions.decay_linear_attention
* neuralops.attentions.eat
* neuralops.attentions.expert_choice_moe
* neuralops.attentions.factorised_matrix_state
* neuralops.attentions.flash_attention3
* neuralops.attentions.flash_attention3_fp8
* neuralops.attentions.full_attention
* neuralops.attentions.fused_svd_kernel
* neuralops.attentions.fusion_attn
* neuralops.attentions.galerkin_attention
* neuralops.attentions.gated_delta_net
* neuralops.attentions.gated_deltanet_block
* neuralops.attentions.gated_linear_attention
* neuralops.attentions.goldfinch_ahn
* neuralops.attentions.gqa_attention
* neuralops.attentions.granite_hybrid
* neuralops.attentions.griffin_ssd
* neuralops.attentions.halo_regla
* neuralops.attentions.hamming_attention
* neuralops.attentions.hgrn2_gated_deltanet
* neuralops.attentions.hgrn2_layer
* neuralops.attentions.hgrn2_outer
* neuralops.attentions.hla_attention
* neuralops.attentions.hybrid_attention
* neuralops.attentions.hymba_transx
* neuralops.attentions.infini_attention
* neuralops.attentions.jamba
* neuralops.attentions.jamba_info
* neuralops.attentions.kda_attention
* neuralops.attentions.leka_v1
* neuralops.attentions.leka_v2
* neuralops.attentions.leka_v3
* neuralops.attentions.linformer_attention
* neuralops.attentions.lola_attention
* neuralops.attentions.lora_attention
* neuralops.attentions.lsh_attention
* neuralops.attentions.mamba_block
* neuralops.attentions.mamba_svd_flash
* neuralops.attentions.minimal_nova
* neuralops.attentions.minimal_svd
* neuralops.attentions.mixattention_moska
* neuralops.attentions.mla_attention
* neuralops.attentions.moba_attention
* neuralops.attentions.moe_deltakv_paged
* neuralops.attentions.naive_svd
* neuralops.attentions.nasc_attention
* neuralops.attentions.nexus
* neuralops.attentions.nha_attention
* neuralops.attentions.nsa_attention
* neuralops.attentions.nystrom_attention
* neuralops.attentions.optimized_hybrid_attention
* neuralops.attentions.orion
* neuralops.attentions.padre_attention
* neuralops.attentions.performer_attention
* neuralops.attentions.phantom
* neuralops.attentions.primal_attention
* neuralops.attentions.progressive_token_pruning
* neuralops.attentions.projection_factorized_attn
* neuralops.attentions.qcqa_attention
* neuralops.attentions.rala_attention
* neuralops.attentions.reap_moe
* neuralops.attentions.regla_attention
* neuralops.attentions.replace
* neuralops.attentions.res_svd
* neuralops.attentions.routing_attention
* neuralops.attentions.rpc_attention
* neuralops.attentions.salt_attention
* neuralops.attentions.samba_m3
* neuralops.attentions.samba_moe
* neuralops.attentions.selective_ssm
* neuralops.attentions.selective_ssm
* neuralops.attentions.soa_attn
* neuralops.attentions.span_expanded_attention
* neuralops.attentions.spectra_v2
* neuralops.attentions.spectraformer_attention
* neuralops.attentions.spectral_mamba
* neuralops.attentions.spectron
* neuralops.attentions.squeezing_heads_distill
* neuralops.attentions.svd_attention
* neuralops.attentions.svd_compressed
* neuralops.attentions.svd_linear_attention
* neuralops.attentions.svd_linear_decay
* neuralops.attentions.svd_llm_v2_attention
* neuralops.attentions.svd_mamba
* neuralops.attentions.svd_projection_attention
* neuralops.attentions.template_attention
* neuralops.attentions.tlinformer_attention
* neuralops.attentions.token_compressed_attention
* neuralops.attentions.tpa_attention
* neuralops.attentions.triton_cosformer
* neuralops.attentions.triton_svd_linear_attention
* neuralops.attentions.unified_rope
* neuralops.attentions.upi_stuffed_mamba
* neuralops.attentions.vig_attention
* neuralops.attentions.vspec
* neuralops.attentions.wgqa_attention
* neuralops.attentions.xattention
* neuralops.attentions.zeros_attention

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

`AdaSplashAttention(config: Any, rank: int | None = None, alpha: float = 1.1, n_iter: int = 15, splash_radius: int = 128, pretrained_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   AdaSplash Attention Layer with α-Entmax and Local Splash Radius.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device: torch.device | None = None) ‑> torch.Tensor`
    :

`AdaSplashLinear(in_features: int, out_features: int, lora_rank: int = 16, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear projection layer with frozen base weights and trainable residual LoRA adapter.
    
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

    `reset_parameters(self) ‑> None`
    :

`AdaptiveSVDAttention(config: Any, svd_rank: int | None = None, residual_rank: int = 16, energy_threshold: float = 0.999, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Multi-head attention with adaptive SVD + residual factorized projections.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin_alias`
    :

    `q_lin_alias`
    :

    `v_lin_alias`
    :

    ### Methods

    `compute_bias(self, query_length, key_length, device=None)`
    :

    `storage_bytes(self) ‑> dict`
    :

`AdaptiveSVDLinear(in_features: int, out_features: int, svd_rank: int, residual_rank: int = 16, bias: bool = True, trainable_svd: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Frozen SVD base + trainable LoRA residual.
    
    Forward:
        y = x V^T U^T + x lora_A^T lora_B^T + b
    
    Storage:
        U (out, r), V (r, in), lora_A (residual_rank, in), lora_B (out, residual_rank)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_compressed_svd: bool`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `is_svd_linear: bool`
    :   The type of the None singleton.

    ### Instance variables

    `A: torch.nn.parameter.Parameter | None`
    :

    `B: torch.nn.parameter.Parameter | None`
    :

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

    `reset_parameters(self) ‑> None`
    :

    `set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) ‑> None`
    :

`AdaptiveSpanAttention(config: Any | None = None, rank: int | None = None, dropout: float = 0.1, max_span: int = 128, ramp_size: float = 32.0, lora_rank: int = 16, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, layer_idx: int | None = None, return_style: str | None = None, rolling_cache_size: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `current_span: torch.Tensor`
    :

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device=None) ‑> torch.Tensor`
    :

    `get_span_regularization_loss(self, lambda_span: float = 0.01) ‑> torch.Tensor`
    :

`AdaptiveSparseAttention(config: Any | None = None, rank: int | None = None, sparsity_ratio: float | None = None, dropout: float = 0.1, lora_rank: int = 16, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional`
    :   The type of the None singleton.

    `supports_causal`
    :   The type of the None singleton.

    `supports_gqa`
    :   The type of the None singleton.

    `supports_rope`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `key: nn.Module`
    :

    `query: nn.Module`
    :

    `value: nn.Module`
    :

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device: torch.device | None = None) ‑> torch.Tensor`
    :

    `freeze_base_weights(self) ‑> None`
    :   Freeze base projection weights when LoRA adapters are present.

    `get_sparsity_loss(self) ‑> torch.Tensor`
    :   Regularization loss encouraging head sparsity.

    `unfreeze_base_weights(self) ‑> None`
    :   Unfreeze base projection weights.

`AegisAttention(model_config: Any = None, config: Any = None, aegis_cfg: neuralops.attentions.aegis_v5.AegisConfigV5 | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `c_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `dense: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `k_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `key: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `o_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `out_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `q_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `query: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `v_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `value: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV5(model_config: Any = None, config: Any = None, aegis_cfg: neuralops.attentions.aegis_v5.AegisConfigV5 | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `c_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `dense: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `k_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `key: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `o_proj: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `out_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `q_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `query: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `v_lin: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    `value: neuralops.attentions.aegis_v5.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)`
:   AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)

    ### Instance variables

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

`AegisConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)`
:   AegisAttentionConfigV2(hidden_size: int = 768, num_attention_heads: int = 12, num_heads: int = 12, head_dim: int = 64, num_key_value_heads: int = 12, local_window: int = 64, rank: int = 384, kernel_feature_dim: int = 128, dropout: float = 0.1, use_adaptive_pruning: bool = True, gate_init_bias: float = 1.8, is_decoder: bool = False)

    ### Instance variables

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

`AegisAttentionV3(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV1(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV2(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV4(config: Any = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.aegis_v2.AegisAttentionV2`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj1`
    :

    `k_proj2`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `out_proj1`
    :

    `out_proj2`
    :

    `q_lin`
    :

    `q_proj1`
    :

    `q_proj2`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj1`
    :

    `v_proj2`
    :

    `value`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisAttentionV6(config=None, pretrained_qkv=None, pretrained_attn=None, is_decoder=None, **kwargs)`
:   Every custom attention module must inherit from this.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `get_ortho_loss(self)`
    :

`AegisConfigV3(**kwargs: Any)`
:   

`AegisConfigV5(svd_energy: float = 0.95, min_rank: int = 4, max_rank: int = 384, num_local_heads: int = 8, num_global_heads: int = 4, local_window: int = 64, softpick_temperature: float = 1.0, rff_features: int = 128, use_zero_sum_keys: bool = True, use_adaptive_gate: bool = True, gate_init_bias: float = 1.8, dropout: float = 0.1, is_decoder: bool = False, rff_trainable: bool = False, use_sdpa_fast_path: bool = True, causal_chunk_size: int = 64)`
:   Configuration for AegisAttention V5.

    ### Instance variables

    `causal_chunk_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

    `num_global_heads: int`
    :   The type of the None singleton.

    `num_local_heads: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `rff_trainable: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `svd_energy: float`
    :   The type of the None singleton.

    `use_adaptive_gate: bool`
    :   The type of the None singleton.

    `use_sdpa_fast_path: bool`
    :   The type of the None singleton.

    `use_zero_sum_keys: bool`
    :   The type of the None singleton.

`AegisHybridAttention(config: Any | None = None, hidden_size: int | None = None, num_heads: int | None = None, pretrained_attn: nn.Module | None = None, rank: int | None = None, lora_rank: int | None = None, is_decoder: bool | None = None, whitening: dict[str, torch.Tensor] | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`AegisHybridConfig(hidden_size: int = 768, num_attention_heads: int = 12, head_dim: int = 64, num_key_value_heads: int | None = None, local_window: int = 256, rank: int = 128, lora_rank: int = 32, dropout: float = 0.1, use_adaptive_pruning: bool = False, gate_init_bias: float = 1.8, is_decoder: bool = False, use_global: bool = True, kernel_feature_dim: int = 128, kernel_seed: int = 1234, chunk_size: int = 64)`
:   AegisHybridConfig(hidden_size: 'int' = 768, num_attention_heads: 'int' = 12, head_dim: 'int' = 64, num_key_value_heads: 'int | None' = None, local_window: 'int' = 256, rank: 'int' = 128, lora_rank: 'int' = 32, dropout: 'float' = 0.1, use_adaptive_pruning: 'bool' = False, gate_init_bias: 'float' = 1.8, is_decoder: 'bool' = False, use_global: 'bool' = True, kernel_feature_dim: 'int' = 128, kernel_seed: 'int' = 1234, chunk_size: 'int' = 64)

    ### Instance variables

    `chunk_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `is_decoder: bool`
    :   The type of the None singleton.

    `kernel_feature_dim: int`
    :   The type of the None singleton.

    `kernel_seed: int`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `lora_rank: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

    `num_key_value_heads: int | None`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `use_adaptive_pruning: bool`
    :   The type of the None singleton.

    `use_global: bool`
    :   The type of the None singleton.

`AegisSpectronFP8Attention(config: Any | None = None, hidden_size: int | None = None, num_heads: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k`
    :

    `k_lin`
    :

    `key`
    :

    `o`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q`
    :

    `q_lin`
    :

    `query`
    :

    `v`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`AegisSpectronFP8Config(**kwargs: Any)`
:   

`Any(*args, **kwargs)`
:   Special type indicating an unconstrained type.
    
    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.
    
    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    checks.

`AutoConfig()`
:   This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.
    
    This class cannot be instantiated directly using `__init__()` (throws an error).

    ### Static methods

    `for_model(model_type: str, *args, **kwargs) ‑> transformers.configuration_utils.PreTrainedConfig`
    :

    `from_pretrained(pretrained_model_name_or_path: str | os.PathLike[str], **kwargs)`
    :   Instantiate one of the configuration classes of the library from a pretrained model configuration.
        
        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:
        
            - **EvollaModel** -- [`EvollaConfig`] (EvollaConfig model)
            - **afmoe** -- [`AfmoeConfig`] (AfmoeConfig model)
            - **aimv2** -- [`Aimv2Config`] (Aimv2Config model)
            - **aimv2_text_model** -- [`Aimv2TextConfig`] (Aimv2TextConfig model)
            - **aimv2_vision_model** -- [`Aimv2VisionConfig`] (Aimv2VisionConfig model)
            - **albert** -- [`AlbertConfig`] (AlbertConfig model)
            - **align** -- [`AlignConfig`] (AlignConfig model)
            - **align_text_model** -- [`AlignTextConfig`] (AlignTextConfig model)
            - **align_vision_model** -- [`AlignVisionConfig`] (AlignVisionConfig model)
            - **altclip** -- [`AltCLIPConfig`] (AltCLIPConfig model)
            - **altclip_text_model** -- [`AltCLIPTextConfig`] (AltCLIPTextConfig model)
            - **altclip_vision_model** -- [`AltCLIPVisionConfig`] (AltCLIPVisionConfig model)
            - **apertus** -- [`ApertusConfig`] (ApertusConfig model)
            - **arcee** -- [`ArceeConfig`] (ArceeConfig model)
            - **aria** -- [`AriaConfig`] (AriaConfig model)
            - **aria_text** -- [`AriaTextConfig`] (AriaTextConfig model)
            - **audio-spectrogram-transformer** -- [`ASTConfig`] (ASTConfig model)
            - **audioflamingo3** -- [`AudioFlamingo3Config`] (AudioFlamingo3Config model)
            - **audioflamingo3_encoder** -- [`AudioFlamingo3EncoderConfig`] (AudioFlamingo3EncoderConfig model)
            - **autoformer** -- [`AutoformerConfig`] (AutoformerConfig model)
            - **aya_vision** -- [`AyaVisionConfig`] (AyaVisionConfig model)
            - **bamba** -- [`BambaConfig`] (BambaConfig model)
            - **bark** -- [`BarkConfig`] (BarkConfig model)
            - **bart** -- [`BartConfig`] (BartConfig model)
            - **beit** -- [`BeitConfig`] (BeitConfig model)
            - **bert** -- [`BertConfig`] (BertConfig model)
            - **bert-generation** -- [`BertGenerationConfig`] (BertGenerationConfig model)
            - **big_bird** -- [`BigBirdConfig`] (BigBirdConfig model)
            - **bigbird_pegasus** -- [`BigBirdPegasusConfig`] (BigBirdPegasusConfig model)
            - **biogpt** -- [`BioGptConfig`] (BioGptConfig model)
            - **bit** -- [`BitConfig`] (BitConfig model)
            - **bitnet** -- [`BitNetConfig`] (BitNetConfig model)
            - **blenderbot** -- [`BlenderbotConfig`] (BlenderbotConfig model)
            - **blenderbot-small** -- [`BlenderbotSmallConfig`] (BlenderbotSmallConfig model)
            - **blip** -- [`BlipConfig`] (BlipConfig model)
            - **blip-2** -- [`Blip2Config`] (Blip2Config model)
            - **blip_2_qformer** -- [`Blip2QFormerConfig`] (Blip2QFormerConfig model)
            - **blip_2_vision_model** -- [`Blip2VisionConfig`] (Blip2VisionConfig model)
            - **blip_text_model** -- [`BlipTextConfig`] (BlipTextConfig model)
            - **blip_vision_model** -- [`BlipVisionConfig`] (BlipVisionConfig model)
            - **bloom** -- [`BloomConfig`] (BloomConfig model)
            - **blt** -- [`BltConfig`] (BltConfig model)
            - **blt_global_transformer** -- [`BltGlobalTransformerConfig`] (BltGlobalTransformerConfig model)
            - **blt_local_decoder** -- [`BltLocalDecoderConfig`] (BltLocalDecoderConfig model)
            - **blt_local_encoder** -- [`BltLocalEncoderConfig`] (BltLocalEncoderConfig model)
            - **blt_patcher** -- [`BltPatcherConfig`] (BltPatcherConfig model)
            - **bridgetower** -- [`BridgeTowerConfig`] (BridgeTowerConfig model)
            - **bridgetower_text_model** -- [`BridgeTowerTextConfig`] (BridgeTowerTextConfig model)
            - **bridgetower_vision_model** -- [`BridgeTowerVisionConfig`] (BridgeTowerVisionConfig model)
            - **bros** -- [`BrosConfig`] (BrosConfig model)
            - **camembert** -- [`CamembertConfig`] (CamembertConfig model)
            - **canine** -- [`CanineConfig`] (CanineConfig model)
            - **chameleon** -- [`ChameleonConfig`] (ChameleonConfig model)
            - **chameleon_vqgan** -- [`ChameleonVQVAEConfig`] (ChameleonVQVAEConfig model)
            - **chinese_clip** -- [`ChineseCLIPConfig`] (ChineseCLIPConfig model)
            - **chinese_clip_text_model** -- [`ChineseCLIPTextConfig`] (ChineseCLIPTextConfig model)
            - **chinese_clip_vision_model** -- [`ChineseCLIPVisionConfig`] (ChineseCLIPVisionConfig model)
            - **chmv2** -- [`CHMv2Config`] (CHMv2Config model)
            - **clap** -- [`ClapConfig`] (ClapConfig model)
            - **clap_audio_model** -- [`ClapAudioConfig`] (ClapAudioConfig model)
            - **clap_text_model** -- [`ClapTextConfig`] (ClapTextConfig model)
            - **clip** -- [`CLIPConfig`] (CLIPConfig model)
            - **clip_text_model** -- [`CLIPTextConfig`] (CLIPTextConfig model)
            - **clip_vision_model** -- [`CLIPVisionConfig`] (CLIPVisionConfig model)
            - **clipseg** -- [`CLIPSegConfig`] (CLIPSegConfig model)
            - **clipseg_text_model** -- [`CLIPSegTextConfig`] (CLIPSegTextConfig model)
            - **clipseg_vision_model** -- [`CLIPSegVisionConfig`] (CLIPSegVisionConfig model)
            - **clvp** -- [`ClvpConfig`] (ClvpConfig model)
            - **clvp_decoder** -- [`ClvpDecoderConfig`] (ClvpDecoderConfig model)
            - **clvp_encoder** -- [`ClvpEncoderConfig`] (ClvpEncoderConfig model)
            - **codegen** -- [`CodeGenConfig`] (CodeGenConfig model)
            - **cohere** -- [`CohereConfig`] (CohereConfig model)
            - **cohere2** -- [`Cohere2Config`] (Cohere2Config model)
            - **cohere2_vision** -- [`Cohere2VisionConfig`] (Cohere2VisionConfig model)
            - **cohere_asr** -- [`CohereAsrConfig`] (CohereAsrConfig model)
            - **colmodernvbert** -- [`ColModernVBertConfig`] (ColModernVBertConfig model)
            - **colpali** -- [`ColPaliConfig`] (ColPaliConfig model)
            - **colqwen2** -- [`ColQwen2Config`] (ColQwen2Config model)
            - **conditional_detr** -- [`ConditionalDetrConfig`] (ConditionalDetrConfig model)
            - **convbert** -- [`ConvBertConfig`] (ConvBertConfig model)
            - **convnext** -- [`ConvNextConfig`] (ConvNextConfig model)
            - **convnextv2** -- [`ConvNextV2Config`] (ConvNextV2Config model)
            - **cpmant** -- [`CpmAntConfig`] (CpmAntConfig model)
            - **csm** -- [`CsmConfig`] (CsmConfig model)
            - **csm_depth_decoder_model** -- [`CsmDepthDecoderConfig`] (CsmDepthDecoderConfig model)
            - **ctrl** -- [`CTRLConfig`] (CTRLConfig model)
            - **cvt** -- [`CvtConfig`] (CvtConfig model)
            - **cwm** -- [`CwmConfig`] (CwmConfig model)
            - **d_fine** -- [`DFineConfig`] (DFineConfig model)
            - **dab-detr** -- [`DabDetrConfig`] (DabDetrConfig model)
            - **dac** -- [`DacConfig`] (DacConfig model)
            - **data2vec-audio** -- [`Data2VecAudioConfig`] (Data2VecAudioConfig model)
            - **data2vec-text** -- [`Data2VecTextConfig`] (Data2VecTextConfig model)
            - **data2vec-vision** -- [`Data2VecVisionConfig`] (Data2VecVisionConfig model)
            - **dbrx** -- [`DbrxConfig`] (DbrxConfig model)
            - **deberta** -- [`DebertaConfig`] (DebertaConfig model)
            - **deberta-v2** -- [`DebertaV2Config`] (DebertaV2Config model)
            - **decision_transformer** -- [`DecisionTransformerConfig`] (DecisionTransformerConfig model)
            - **deepseek_v2** -- [`DeepseekV2Config`] (DeepseekV2Config model)
            - **deepseek_v3** -- [`DeepseekV3Config`] (DeepseekV3Config model)
            - **deepseek_vl** -- [`DeepseekVLConfig`] (DeepseekVLConfig model)
            - **deepseek_vl_hybrid** -- [`DeepseekVLHybridConfig`] (DeepseekVLHybridConfig model)
            - **deformable_detr** -- [`DeformableDetrConfig`] (DeformableDetrConfig model)
            - **deimv2** -- [`Deimv2Config`] (Deimv2Config model)
            - **deit** -- [`DeiTConfig`] (DeiTConfig model)
            - **depth_anything** -- [`DepthAnythingConfig`] (DepthAnythingConfig model)
            - **depth_pro** -- [`DepthProConfig`] (DepthProConfig model)
            - **detr** -- [`DetrConfig`] (DetrConfig model)
            - **dia** -- [`DiaConfig`] (DiaConfig model)
            - **dia_decoder** -- [`DiaDecoderConfig`] (DiaDecoderConfig model)
            - **dia_encoder** -- [`DiaEncoderConfig`] (DiaEncoderConfig model)
            - **diffllama** -- [`DiffLlamaConfig`] (DiffLlamaConfig model)
            - **dinat** -- [`DinatConfig`] (DinatConfig model)
            - **dinov2** -- [`Dinov2Config`] (Dinov2Config model)
            - **dinov2_with_registers** -- [`Dinov2WithRegistersConfig`] (Dinov2WithRegistersConfig model)
            - **dinov3_convnext** -- [`DINOv3ConvNextConfig`] (DINOv3ConvNextConfig model)
            - **dinov3_vit** -- [`DINOv3ViTConfig`] (DINOv3ViTConfig model)
            - **distilbert** -- [`DistilBertConfig`] (DistilBertConfig model)
            - **doge** -- [`DogeConfig`] (DogeConfig model)
            - **donut-swin** -- [`DonutSwinConfig`] (DonutSwinConfig model)
            - **dots1** -- [`Dots1Config`] (Dots1Config model)
            - **dpr** -- [`DPRConfig`] (DPRConfig model)
            - **dpt** -- [`DPTConfig`] (DPTConfig model)
            - **edgetam** -- [`EdgeTamConfig`] (EdgeTamConfig model)
            - **edgetam_video** -- [`EdgeTamVideoConfig`] (EdgeTamVideoConfig model)
            - **edgetam_vision_model** -- [`EdgeTamVisionConfig`] (EdgeTamVisionConfig model)
            - **efficientloftr** -- [`EfficientLoFTRConfig`] (EfficientLoFTRConfig model)
            - **efficientnet** -- [`EfficientNetConfig`] (EfficientNetConfig model)
            - **electra** -- [`ElectraConfig`] (ElectraConfig model)
            - **emu3** -- [`Emu3Config`] (Emu3Config model)
            - **emu3_text_model** -- [`Emu3TextConfig`] (Emu3TextConfig model)
            - **emu3_vqgan** -- [`Emu3VQVAEConfig`] (Emu3VQVAEConfig model)
            - **encodec** -- [`EncodecConfig`] (EncodecConfig model)
            - **encoder-decoder** -- [`EncoderDecoderConfig`] (EncoderDecoderConfig model)
            - **eomt** -- [`EomtConfig`] (EomtConfig model)
            - **eomt_dinov3** -- [`EomtDinov3Config`] (EomtDinov3Config model)
            - **ernie** -- [`ErnieConfig`] (ErnieConfig model)
            - **ernie4_5** -- [`Ernie4_5Config`] (Ernie4_5Config model)
            - **ernie4_5_moe** -- [`Ernie4_5_MoeConfig`] (Ernie4_5_MoeConfig model)
            - **ernie4_5_vl_moe** -- [`Ernie4_5_VLMoeConfig`] (Ernie4_5_VLMoeConfig model)
            - **ernie4_5_vl_moe_text** -- [`Ernie4_5_VLMoeTextConfig`] (Ernie4_5_VLMoeTextConfig model)
            - **ernie4_5_vl_moe_vision** -- [`Ernie4_5_VLMoeVisionConfig`] (Ernie4_5_VLMoeVisionConfig model)
            - **esm** -- [`EsmConfig`] (EsmConfig model)
            - **eurobert** -- [`EuroBertConfig`] (EuroBertConfig model)
            - **evolla** -- [`EvollaConfig`] (EvollaConfig model)
            - **exaone4** -- [`Exaone4Config`] (Exaone4Config model)
            - **exaone_moe** -- [`ExaoneMoeConfig`] (ExaoneMoeConfig model)
            - **falcon** -- [`FalconConfig`] (FalconConfig model)
            - **falcon_h1** -- [`FalconH1Config`] (FalconH1Config model)
            - **falcon_mamba** -- [`FalconMambaConfig`] (FalconMambaConfig model)
            - **fast_vlm** -- [`FastVlmConfig`] (FastVlmConfig model)
            - **fastspeech2_conformer** -- [`FastSpeech2ConformerConfig`] (FastSpeech2ConformerConfig model)
            - **fastspeech2_conformer_hifigan** -- [`FastSpeech2ConformerHifiGanConfig`] (FastSpeech2ConformerHifiGanConfig model)
            - **fastspeech2_conformer_with_hifigan** -- [`FastSpeech2ConformerWithHifiGanConfig`] (FastSpeech2ConformerWithHifiGanConfig model)
            - **flaubert** -- [`FlaubertConfig`] (FlaubertConfig model)
            - **flava** -- [`FlavaConfig`] (FlavaConfig model)
            - **flava_image_model** -- [`FlavaImageConfig`] (FlavaImageConfig model)
            - **flava_multimodal_model** -- [`FlavaMultimodalConfig`] (FlavaMultimodalConfig model)
            - **flava_text_model** -- [`FlavaTextConfig`] (FlavaTextConfig model)
            - **flex_olmo** -- [`FlexOlmoConfig`] (FlexOlmoConfig model)
            - **florence2** -- [`Florence2Config`] (Florence2Config model)
            - **florence_vision** -- [`Florence2VisionConfig`] (Florence2VisionConfig model)
            - **fnet** -- [`FNetConfig`] (FNetConfig model)
            - **focalnet** -- [`FocalNetConfig`] (FocalNetConfig model)
            - **fsmt** -- [`FSMTConfig`] (FSMTConfig model)
            - **funnel** -- [`FunnelConfig`] (FunnelConfig model)
            - **fuyu** -- [`FuyuConfig`] (FuyuConfig model)
            - **gemma** -- [`GemmaConfig`] (GemmaConfig model)
            - **gemma2** -- [`Gemma2Config`] (Gemma2Config model)
            - **gemma3** -- [`Gemma3Config`] (Gemma3Config model)
            - **gemma3_text** -- [`Gemma3TextConfig`] (Gemma3TextConfig model)
            - **gemma3n** -- [`Gemma3nConfig`] (Gemma3nConfig model)
            - **gemma3n_audio** -- [`Gemma3nAudioConfig`] (Gemma3nAudioConfig model)
            - **gemma3n_text** -- [`Gemma3nTextConfig`] (Gemma3nTextConfig model)
            - **gemma3n_vision** -- [`Gemma3nVisionConfig`] (Gemma3nVisionConfig model)
            - **gemma4** -- [`Gemma4Config`] (Gemma4Config model)
            - **gemma4_audio** -- [`Gemma4AudioConfig`] (Gemma4AudioConfig model)
            - **gemma4_text** -- [`Gemma4TextConfig`] (Gemma4TextConfig model)
            - **gemma4_vision** -- [`Gemma4VisionConfig`] (Gemma4VisionConfig model)
            - **git** -- [`GitConfig`] (GitConfig model)
            - **git_vision_model** -- [`GitVisionConfig`] (GitVisionConfig model)
            - **glm** -- [`GlmConfig`] (GlmConfig model)
            - **glm4** -- [`Glm4Config`] (Glm4Config model)
            - **glm46v** -- [`Glm46VConfig`] (Glm46VConfig model)
            - **glm4_moe** -- [`Glm4MoeConfig`] (Glm4MoeConfig model)
            - **glm4_moe_lite** -- [`Glm4MoeLiteConfig`] (Glm4MoeLiteConfig model)
            - **glm4v** -- [`Glm4vConfig`] (Glm4vConfig model)
            - **glm4v_moe** -- [`Glm4vMoeConfig`] (Glm4vMoeConfig model)
            - **glm4v_moe_text** -- [`Glm4vMoeTextConfig`] (Glm4vMoeTextConfig model)
            - **glm4v_moe_vision** -- [`Glm4vMoeVisionConfig`] (Glm4vMoeVisionConfig model)
            - **glm4v_text** -- [`Glm4vTextConfig`] (Glm4vTextConfig model)
            - **glm4v_vision** -- [`Glm4vVisionConfig`] (Glm4vVisionConfig model)
            - **glm_image** -- [`GlmImageConfig`] (GlmImageConfig model)
            - **glm_image_text** -- [`GlmImageTextConfig`] (GlmImageTextConfig model)
            - **glm_image_vision** -- [`GlmImageVisionConfig`] (GlmImageVisionConfig model)
            - **glm_image_vqmodel** -- [`GlmImageVQVAEConfig`] (GlmImageVQVAEConfig model)
            - **glm_moe_dsa** -- [`GlmMoeDsaConfig`] (GlmMoeDsaConfig model)
            - **glm_ocr** -- [`GlmOcrConfig`] (GlmOcrConfig model)
            - **glm_ocr_text** -- [`GlmOcrTextConfig`] (GlmOcrTextConfig model)
            - **glm_ocr_vision** -- [`GlmOcrVisionConfig`] (GlmOcrVisionConfig model)
            - **glmasr** -- [`GlmAsrConfig`] (GlmAsrConfig model)
            - **glmasr_encoder** -- [`GlmAsrEncoderConfig`] (GlmAsrEncoderConfig model)
            - **glpn** -- [`GLPNConfig`] (GLPNConfig model)
            - **got_ocr2** -- [`GotOcr2Config`] (GotOcr2Config model)
            - **gpt-sw3** -- [`GPT2Config`] (GPT2Config model)
            - **gpt2** -- [`GPT2Config`] (GPT2Config model)
            - **gpt_bigcode** -- [`GPTBigCodeConfig`] (GPTBigCodeConfig model)
            - **gpt_neo** -- [`GPTNeoConfig`] (GPTNeoConfig model)
            - **gpt_neox** -- [`GPTNeoXConfig`] (GPTNeoXConfig model)
            - **gpt_neox_japanese** -- [`GPTNeoXJapaneseConfig`] (GPTNeoXJapaneseConfig model)
            - **gpt_oss** -- [`GptOssConfig`] (GptOssConfig model)
            - **gptj** -- [`GPTJConfig`] (GPTJConfig model)
            - **granite** -- [`GraniteConfig`] (GraniteConfig model)
            - **granite_speech** -- [`GraniteSpeechConfig`] (GraniteSpeechConfig model)
            - **granite_speech_encoder** -- [`GraniteSpeechEncoderConfig`] (GraniteSpeechEncoderConfig model)
            - **granitemoe** -- [`GraniteMoeConfig`] (GraniteMoeConfig model)
            - **granitemoehybrid** -- [`GraniteMoeHybridConfig`] (GraniteMoeHybridConfig model)
            - **granitemoeshared** -- [`GraniteMoeSharedConfig`] (GraniteMoeSharedConfig model)
            - **grounding-dino** -- [`GroundingDinoConfig`] (GroundingDinoConfig model)
            - **groupvit** -- [`GroupViTConfig`] (GroupViTConfig model)
            - **groupvit_text_model** -- [`GroupViTTextConfig`] (GroupViTTextConfig model)
            - **groupvit_vision_model** -- [`GroupViTVisionConfig`] (GroupViTVisionConfig model)
            - **helium** -- [`HeliumConfig`] (HeliumConfig model)
            - **hgnet_v2** -- [`HGNetV2Config`] (HGNetV2Config model)
            - **hiera** -- [`HieraConfig`] (HieraConfig model)
            - **higgs_audio_v2** -- [`HiggsAudioV2Config`] (HiggsAudioV2Config model)
            - **higgs_audio_v2_tokenizer** -- [`HiggsAudioV2TokenizerConfig`] (HiggsAudioV2TokenizerConfig model)
            - **hubert** -- [`HubertConfig`] (HubertConfig model)
            - **hunyuan_v1_dense** -- [`HunYuanDenseV1Config`] (HunYuanDenseV1Config model)
            - **hunyuan_v1_moe** -- [`HunYuanMoEV1Config`] (HunYuanMoEV1Config model)
            - **hy_v3** -- [`HYV3Config`] (HYV3Config model)
            - **ibert** -- [`IBertConfig`] (IBertConfig model)
            - **idefics** -- [`IdeficsConfig`] (IdeficsConfig model)
            - **idefics2** -- [`Idefics2Config`] (Idefics2Config model)
            - **idefics2_perceiver** -- [`Idefics2PerceiverConfig`] (Idefics2PerceiverConfig model)
            - **idefics2_vision** -- [`Idefics2VisionConfig`] (Idefics2VisionConfig model)
            - **idefics3** -- [`Idefics3Config`] (Idefics3Config model)
            - **idefics3_vision** -- [`Idefics3VisionConfig`] (Idefics3VisionConfig model)
            - **idefics_perciever** -- [`IdeficsPerceiverConfig`] (IdeficsPerceiverConfig model)
            - **idefics_vision** -- [`IdeficsVisionConfig`] (IdeficsVisionConfig model)
            - **ijepa** -- [`IJepaConfig`] (IJepaConfig model)
            - **imagegpt** -- [`ImageGPTConfig`] (ImageGPTConfig model)
            - **informer** -- [`InformerConfig`] (InformerConfig model)
            - **instructblip** -- [`InstructBlipConfig`] (InstructBlipConfig model)
            - **instructblip_qformer** -- [`InstructBlipQFormerConfig`] (InstructBlipQFormerConfig model)
            - **instructblip_vision_model** -- [`InstructBlipVisionConfig`] (InstructBlipVisionConfig model)
            - **instructblipvideo** -- [`InstructBlipVideoConfig`] (InstructBlipVideoConfig model)
            - **instructblipvideo_qformer** -- [`InstructBlipVideoQFormerConfig`] (InstructBlipVideoQFormerConfig model)
            - **instructblipvideo_vision_model** -- [`InstructBlipVideoVisionConfig`] (InstructBlipVideoVisionConfig model)
            - **internvl** -- [`InternVLConfig`] (InternVLConfig model)
            - **internvl_vision** -- [`InternVLVisionConfig`] (InternVLVisionConfig model)
            - **jais2** -- [`Jais2Config`] (Jais2Config model)
            - **jamba** -- [`JambaConfig`] (JambaConfig model)
            - **janus** -- [`JanusConfig`] (JanusConfig model)
            - **janus_vision_model** -- [`JanusVisionConfig`] (JanusVisionConfig model)
            - **janus_vqgan** -- [`JanusVQVAEConfig`] (JanusVQVAEConfig model)
            - **jetmoe** -- [`JetMoeConfig`] (JetMoeConfig model)
            - **jina_embeddings_v3** -- [`JinaEmbeddingsV3Config`] (JinaEmbeddingsV3Config model)
            - **kosmos-2** -- [`Kosmos2Config`] (Kosmos2Config model)
            - **kosmos-2.5** -- [`Kosmos2_5Config`] (Kosmos2_5Config model)
            - **kosmos_2_5_text_model** -- [`Kosmos2_5TextConfig`] (Kosmos2_5TextConfig model)
            - **kosmos_2_5_vision_model** -- [`Kosmos2_5VisionConfig`] (Kosmos2_5VisionConfig model)
            - **kosmos_2_text_model** -- [`Kosmos2TextConfig`] (Kosmos2TextConfig model)
            - **kosmos_2_vision_model** -- [`Kosmos2VisionConfig`] (Kosmos2VisionConfig model)
            - **kyutai_speech_to_text** -- [`KyutaiSpeechToTextConfig`] (KyutaiSpeechToTextConfig model)
            - **laguna** -- [`LagunaConfig`] (LagunaConfig model)
            - **lasr_ctc** -- [`LasrCTCConfig`] (LasrCTCConfig model)
            - **lasr_encoder** -- [`LasrEncoderConfig`] (LasrEncoderConfig model)
            - **layoutlm** -- [`LayoutLMConfig`] (LayoutLMConfig model)
            - **layoutlmv2** -- [`LayoutLMv2Config`] (LayoutLMv2Config model)
            - **layoutlmv3** -- [`LayoutLMv3Config`] (LayoutLMv3Config model)
            - **layoutxlm** -- [`LayoutXLMConfig`] (LayoutXLMConfig model)
            - **led** -- [`LEDConfig`] (LEDConfig model)
            - **levit** -- [`LevitConfig`] (LevitConfig model)
            - **lfm2** -- [`Lfm2Config`] (Lfm2Config model)
            - **lfm2_moe** -- [`Lfm2MoeConfig`] (Lfm2MoeConfig model)
            - **lfm2_vl** -- [`Lfm2VlConfig`] (Lfm2VlConfig model)
            - **lightglue** -- [`LightGlueConfig`] (LightGlueConfig model)
            - **lighton_ocr** -- [`LightOnOcrConfig`] (LightOnOcrConfig model)
            - **lilt** -- [`LiltConfig`] (LiltConfig model)
            - **llama** -- [`LlamaConfig`] (LlamaConfig model)
            - **llama4** -- [`Llama4Config`] (Llama4Config model)
            - **llama4_text** -- [`Llama4TextConfig`] (Llama4TextConfig model)
            - **llama4_vision_model** -- [`Llama4VisionConfig`] (Llama4VisionConfig model)
            - **llava** -- [`LlavaConfig`] (LlavaConfig model)
            - **llava_next** -- [`LlavaNextConfig`] (LlavaNextConfig model)
            - **llava_next_video** -- [`LlavaNextVideoConfig`] (LlavaNextVideoConfig model)
            - **llava_onevision** -- [`LlavaOnevisionConfig`] (LlavaOnevisionConfig model)
            - **longcat_flash** -- [`LongcatFlashConfig`] (LongcatFlashConfig model)
            - **longformer** -- [`LongformerConfig`] (LongformerConfig model)
            - **longt5** -- [`LongT5Config`] (LongT5Config model)
            - **luke** -- [`LukeConfig`] (LukeConfig model)
            - **lw_detr** -- [`LwDetrConfig`] (LwDetrConfig model)
            - **lw_detr_vit** -- [`LwDetrViTConfig`] (LwDetrViTConfig model)
            - **lxmert** -- [`LxmertConfig`] (LxmertConfig model)
            - **m2m_100** -- [`M2M100Config`] (M2M100Config model)
            - **mamba** -- [`MambaConfig`] (MambaConfig model)
            - **mamba2** -- [`Mamba2Config`] (Mamba2Config model)
            - **marian** -- [`MarianConfig`] (MarianConfig model)
            - **markuplm** -- [`MarkupLMConfig`] (MarkupLMConfig model)
            - **mask2former** -- [`Mask2FormerConfig`] (Mask2FormerConfig model)
            - **maskformer** -- [`MaskFormerConfig`] (MaskFormerConfig model)
            - **maskformer-swin** -- [`MaskFormerSwinConfig`] (MaskFormerSwinConfig model)
            - **mbart** -- [`MBartConfig`] (MBartConfig model)
            - **megatron-bert** -- [`MegatronBertConfig`] (MegatronBertConfig model)
            - **metaclip_2** -- [`MetaClip2Config`] (MetaClip2Config model)
            - **metaclip_2_text_model** -- [`MetaClip2TextConfig`] (MetaClip2TextConfig model)
            - **metaclip_2_vision_model** -- [`MetaClip2VisionConfig`] (MetaClip2VisionConfig model)
            - **mgp-str** -- [`MgpstrConfig`] (MgpstrConfig model)
            - **mimi** -- [`MimiConfig`] (MimiConfig model)
            - **minicpmv4_6** -- [`MiniCPMV4_6Config`] (MiniCPMV4_6Config model)
            - **minicpmv4_6_vision** -- [`MiniCPMV4_6VisionConfig`] (MiniCPMV4_6VisionConfig model)
            - **minimax** -- [`MiniMaxConfig`] (MiniMaxConfig model)
            - **minimax_m2** -- [`MiniMaxM2Config`] (MiniMaxM2Config model)
            - **ministral** -- [`MinistralConfig`] (MinistralConfig model)
            - **ministral3** -- [`Ministral3Config`] (Ministral3Config model)
            - **mistral** -- [`MistralConfig`] (MistralConfig model)
            - **mistral3** -- [`Mistral3Config`] (Mistral3Config model)
            - **mistral4** -- [`Mistral4Config`] (Mistral4Config model)
            - **mixtral** -- [`MixtralConfig`] (MixtralConfig model)
            - **mlcd** -- [`MLCDVisionConfig`] (MLCDVisionConfig model)
            - **mlcd_vision_model** -- [`MLCDVisionConfig`] (MLCDVisionConfig model)
            - **mllama** -- [`MllamaConfig`] (MllamaConfig model)
            - **mllama_text_model** -- [`MllamaTextConfig`] (MllamaTextConfig model)
            - **mllama_vision_model** -- [`MllamaVisionConfig`] (MllamaVisionConfig model)
            - **mm-grounding-dino** -- [`MMGroundingDinoConfig`] (MMGroundingDinoConfig model)
            - **mobilebert** -- [`MobileBertConfig`] (MobileBertConfig model)
            - **mobilenet_v1** -- [`MobileNetV1Config`] (MobileNetV1Config model)
            - **mobilenet_v2** -- [`MobileNetV2Config`] (MobileNetV2Config model)
            - **mobilevit** -- [`MobileViTConfig`] (MobileViTConfig model)
            - **mobilevitv2** -- [`MobileViTV2Config`] (MobileViTV2Config model)
            - **modernbert** -- [`ModernBertConfig`] (ModernBertConfig model)
            - **modernbert-decoder** -- [`ModernBertDecoderConfig`] (ModernBertDecoderConfig model)
            - **modernvbert** -- [`ModernVBertConfig`] (ModernVBertConfig model)
            - **moonshine** -- [`MoonshineConfig`] (MoonshineConfig model)
            - **moonshine_streaming** -- [`MoonshineStreamingConfig`] (MoonshineStreamingConfig model)
            - **moonshine_streaming_encoder** -- [`MoonshineStreamingEncoderConfig`] (MoonshineStreamingEncoderConfig model)
            - **moshi** -- [`MoshiConfig`] (MoshiConfig model)
            - **moshi_depth** -- [`MoshiDepthConfig`] (MoshiDepthConfig model)
            - **mpnet** -- [`MPNetConfig`] (MPNetConfig model)
            - **mpt** -- [`MptConfig`] (MptConfig model)
            - **mra** -- [`MraConfig`] (MraConfig model)
            - **mt5** -- [`MT5Config`] (MT5Config model)
            - **musicflamingo** -- [`MusicFlamingoConfig`] (MusicFlamingoConfig model)
            - **musicgen** -- [`MusicgenConfig`] (MusicgenConfig model)
            - **musicgen_decoder** -- [`MusicgenDecoderConfig`] (MusicgenDecoderConfig model)
            - **musicgen_melody** -- [`MusicgenMelodyConfig`] (MusicgenMelodyConfig model)
            - **musicgen_melody_decoder** -- [`MusicgenMelodyDecoderConfig`] (MusicgenMelodyDecoderConfig model)
            - **mvp** -- [`MvpConfig`] (MvpConfig model)
            - **nanochat** -- [`NanoChatConfig`] (NanoChatConfig model)
            - **nemotron** -- [`NemotronConfig`] (NemotronConfig model)
            - **nemotron_h** -- [`NemotronHConfig`] (NemotronHConfig model)
            - **nllb-moe** -- [`NllbMoeConfig`] (NllbMoeConfig model)
            - **nomic_bert** -- [`NomicBertConfig`] (NomicBertConfig model)
            - **nougat** -- [`NougatConfig`] (NougatConfig model)
            - **nystromformer** -- [`NystromformerConfig`] (NystromformerConfig model)
            - **olmo** -- [`OlmoConfig`] (OlmoConfig model)
            - **olmo2** -- [`Olmo2Config`] (Olmo2Config model)
            - **olmo3** -- [`Olmo3Config`] (Olmo3Config model)
            - **olmo_hybrid** -- [`OlmoHybridConfig`] (OlmoHybridConfig model)
            - **olmoe** -- [`OlmoeConfig`] (OlmoeConfig model)
            - **omdet-turbo** -- [`OmDetTurboConfig`] (OmDetTurboConfig model)
            - **oneformer** -- [`OneFormerConfig`] (OneFormerConfig model)
            - **openai-gpt** -- [`OpenAIGPTConfig`] (OpenAIGPTConfig model)
            - **openai_privacy_filter** -- [`OpenAIPrivacyFilterConfig`] (OpenAIPrivacyFilterConfig model)
            - **opt** -- [`OPTConfig`] (OPTConfig model)
            - **ovis2** -- [`Ovis2Config`] (Ovis2Config model)
            - **owlv2** -- [`Owlv2Config`] (Owlv2Config model)
            - **owlv2_text_model** -- [`Owlv2TextConfig`] (Owlv2TextConfig model)
            - **owlv2_vision_model** -- [`Owlv2VisionConfig`] (Owlv2VisionConfig model)
            - **owlvit** -- [`OwlViTConfig`] (OwlViTConfig model)
            - **owlvit_text_model** -- [`OwlViTTextConfig`] (OwlViTTextConfig model)
            - **owlvit_vision_model** -- [`OwlViTVisionConfig`] (OwlViTVisionConfig model)
            - **paddleocr_vl** -- [`PaddleOCRVLConfig`] (PaddleOCRVLConfig model)
            - **paddleocr_vl_text** -- [`PaddleOCRTextConfig`] (PaddleOCRTextConfig model)
            - **paddleocr_vl_vision** -- [`PaddleOCRVisionConfig`] (PaddleOCRVisionConfig model)
            - **paligemma** -- [`PaliGemmaConfig`] (PaliGemmaConfig model)
            - **parakeet_ctc** -- [`ParakeetCTCConfig`] (ParakeetCTCConfig model)
            - **parakeet_encoder** -- [`ParakeetEncoderConfig`] (ParakeetEncoderConfig model)
            - **patchtsmixer** -- [`PatchTSMixerConfig`] (PatchTSMixerConfig model)
            - **patchtst** -- [`PatchTSTConfig`] (PatchTSTConfig model)
            - **pe_audio** -- [`PeAudioConfig`] (PeAudioConfig model)
            - **pe_audio_encoder** -- [`PeAudioEncoderConfig`] (PeAudioEncoderConfig model)
            - **pe_audio_video** -- [`PeAudioVideoConfig`] (PeAudioVideoConfig model)
            - **pe_audio_video_encoder** -- [`PeAudioVideoEncoderConfig`] (PeAudioVideoEncoderConfig model)
            - **pe_video** -- [`PeVideoConfig`] (PeVideoConfig model)
            - **pe_video_encoder** -- [`PeVideoEncoderConfig`] (PeVideoEncoderConfig model)
            - **pegasus** -- [`PegasusConfig`] (PegasusConfig model)
            - **pegasus_x** -- [`PegasusXConfig`] (PegasusXConfig model)
            - **perceiver** -- [`PerceiverConfig`] (PerceiverConfig model)
            - **perception_lm** -- [`PerceptionLMConfig`] (PerceptionLMConfig model)
            - **persimmon** -- [`PersimmonConfig`] (PersimmonConfig model)
            - **phi** -- [`PhiConfig`] (PhiConfig model)
            - **phi3** -- [`Phi3Config`] (Phi3Config model)
            - **phi4_multimodal** -- [`Phi4MultimodalConfig`] (Phi4MultimodalConfig model)
            - **phi4_multimodal_audio** -- [`Phi4MultimodalAudioConfig`] (Phi4MultimodalAudioConfig model)
            - **phi4_multimodal_vision** -- [`Phi4MultimodalVisionConfig`] (Phi4MultimodalVisionConfig model)
            - **phimoe** -- [`PhimoeConfig`] (PhimoeConfig model)
            - **pi0** -- [`PI0Config`] (PI0Config model)
            - **pix2struct** -- [`Pix2StructConfig`] (Pix2StructConfig model)
            - **pix2struct_text_model** -- [`Pix2StructTextConfig`] (Pix2StructTextConfig model)
            - **pix2struct_vision_model** -- [`Pix2StructVisionConfig`] (Pix2StructVisionConfig model)
            - **pixio** -- [`PixioConfig`] (PixioConfig model)
            - **pixtral** -- [`PixtralVisionConfig`] (PixtralVisionConfig model)
            - **plbart** -- [`PLBartConfig`] (PLBartConfig model)
            - **poolformer** -- [`PoolFormerConfig`] (PoolFormerConfig model)
            - **pop2piano** -- [`Pop2PianoConfig`] (Pop2PianoConfig model)
            - **pp_chart2table** -- [`PPChart2TableConfig`] (PPChart2TableConfig model)
            - **pp_doclayout_v2** -- [`PPDocLayoutV2Config`] (PPDocLayoutV2Config model)
            - **pp_doclayout_v3** -- [`PPDocLayoutV3Config`] (PPDocLayoutV3Config model)
            - **pp_lcnet** -- [`PPLCNetConfig`] (PPLCNetConfig model)
            - **pp_lcnet_v3** -- [`PPLCNetV3Config`] (PPLCNetV3Config model)
            - **pp_ocrv5_mobile_det** -- [`PPOCRV5MobileDetConfig`] (PPOCRV5MobileDetConfig model)
            - **pp_ocrv5_mobile_rec** -- [`PPOCRV5MobileRecConfig`] (PPOCRV5MobileRecConfig model)
            - **pp_ocrv5_server_det** -- [`PPOCRV5ServerDetConfig`] (PPOCRV5ServerDetConfig model)
            - **pp_ocrv5_server_rec** -- [`PPOCRV5ServerRecConfig`] (PPOCRV5ServerRecConfig model)
            - **prompt_depth_anything** -- [`PromptDepthAnythingConfig`] (PromptDepthAnythingConfig model)
            - **prophetnet** -- [`ProphetNetConfig`] (ProphetNetConfig model)
            - **pvt** -- [`PvtConfig`] (PvtConfig model)
            - **pvt_v2** -- [`PvtV2Config`] (PvtV2Config model)
            - **qianfan_ocr** -- [`QianfanOCRConfig`] (QianfanOCRConfig model)
            - **qianfan_ocr_vision** -- [`QianfanOCRVisionConfig`] (QianfanOCRVisionConfig model)
            - **qwen2** -- [`Qwen2Config`] (Qwen2Config model)
            - **qwen2_5_omni** -- [`Qwen2_5OmniConfig`] (Qwen2_5OmniConfig model)
            - **qwen2_5_omni_audio_encoder** -- [`Qwen2_5OmniAudioEncoderConfig`] (Qwen2_5OmniAudioEncoderConfig model)
            - **qwen2_5_omni_bigvgan** -- [`Qwen2_5OmniBigVGANConfig`] (Qwen2_5OmniBigVGANConfig model)
            - **qwen2_5_omni_dit** -- [`Qwen2_5OmniDiTConfig`] (Qwen2_5OmniDiTConfig model)
            - **qwen2_5_omni_talker** -- [`Qwen2_5OmniTalkerConfig`] (Qwen2_5OmniTalkerConfig model)
            - **qwen2_5_omni_text** -- [`Qwen2_5OmniTextConfig`] (Qwen2_5OmniTextConfig model)
            - **qwen2_5_omni_thinker** -- [`Qwen2_5OmniThinkerConfig`] (Qwen2_5OmniThinkerConfig model)
            - **qwen2_5_omni_token2wav** -- [`Qwen2_5OmniToken2WavConfig`] (Qwen2_5OmniToken2WavConfig model)
            - **qwen2_5_omni_vision_encoder** -- [`Qwen2_5OmniVisionEncoderConfig`] (Qwen2_5OmniVisionEncoderConfig model)
            - **qwen2_5_vl** -- [`Qwen2_5_VLConfig`] (Qwen2_5_VLConfig model)
            - **qwen2_5_vl_text** -- [`Qwen2_5_VLTextConfig`] (Qwen2_5_VLTextConfig model)
            - **qwen2_5_vl_vision** -- [`Qwen2_5_VLVisionConfig`] (Qwen2_5_VLVisionConfig model)
            - **qwen2_audio** -- [`Qwen2AudioConfig`] (Qwen2AudioConfig model)
            - **qwen2_audio_encoder** -- [`Qwen2AudioEncoderConfig`] (Qwen2AudioEncoderConfig model)
            - **qwen2_moe** -- [`Qwen2MoeConfig`] (Qwen2MoeConfig model)
            - **qwen2_vl** -- [`Qwen2VLConfig`] (Qwen2VLConfig model)
            - **qwen2_vl_text** -- [`Qwen2VLTextConfig`] (Qwen2VLTextConfig model)
            - **qwen2_vl_vision** -- [`Qwen2VLVisionConfig`] (Qwen2VLVisionConfig model)
            - **qwen3** -- [`Qwen3Config`] (Qwen3Config model)
            - **qwen3_5** -- [`Qwen3_5Config`] (Qwen3_5Config model)
            - **qwen3_5_moe** -- [`Qwen3_5MoeConfig`] (Qwen3_5MoeConfig model)
            - **qwen3_5_moe_text** -- [`Qwen3_5MoeTextConfig`] (Qwen3_5MoeTextConfig model)
            - **qwen3_5_moe_vision** -- [`Qwen3_5MoeVisionConfig`] (Qwen3_5MoeVisionConfig model)
            - **qwen3_5_text** -- [`Qwen3_5TextConfig`] (Qwen3_5TextConfig model)
            - **qwen3_5_vision** -- [`Qwen3_5VisionConfig`] (Qwen3_5VisionConfig model)
            - **qwen3_moe** -- [`Qwen3MoeConfig`] (Qwen3MoeConfig model)
            - **qwen3_next** -- [`Qwen3NextConfig`] (Qwen3NextConfig model)
            - **qwen3_omni_moe** -- [`Qwen3OmniMoeConfig`] (Qwen3OmniMoeConfig model)
            - **qwen3_omni_moe_audio_encoder** -- [`Qwen3OmniMoeAudioEncoderConfig`] (Qwen3OmniMoeAudioEncoderConfig model)
            - **qwen3_omni_moe_talker_code_predictor** -- [`Qwen3OmniMoeTalkerCodePredictorConfig`] (Qwen3OmniMoeTalkerCodePredictorConfig model)
            - **qwen3_omni_moe_talker_text** -- [`Qwen3OmniMoeTalkerTextConfig`] (Qwen3OmniMoeTalkerTextConfig model)
            - **qwen3_omni_moe_text** -- [`Qwen3OmniMoeTextConfig`] (Qwen3OmniMoeTextConfig model)
            - **qwen3_omni_moe_thinker** -- [`Qwen3OmniMoeThinkerConfig`] (Qwen3OmniMoeThinkerConfig model)
            - **qwen3_omni_moe_vision_encoder** -- [`Qwen3OmniMoeVisionEncoderConfig`] (Qwen3OmniMoeVisionEncoderConfig model)
            - **qwen3_vl** -- [`Qwen3VLConfig`] (Qwen3VLConfig model)
            - **qwen3_vl_moe** -- [`Qwen3VLMoeConfig`] (Qwen3VLMoeConfig model)
            - **qwen3_vl_moe_text** -- [`Qwen3VLMoeTextConfig`] (Qwen3VLMoeTextConfig model)
            - **qwen3_vl_moe_vision** -- [`Qwen3VLMoeVisionConfig`] (Qwen3VLMoeVisionConfig model)
            - **qwen3_vl_text** -- [`Qwen3VLTextConfig`] (Qwen3VLTextConfig model)
            - **qwen3_vl_vision** -- [`Qwen3VLVisionConfig`] (Qwen3VLVisionConfig model)
            - **rag** -- [`RagConfig`] (RagConfig model)
            - **recurrent_gemma** -- [`RecurrentGemmaConfig`] (RecurrentGemmaConfig model)
            - **reformer** -- [`ReformerConfig`] (ReformerConfig model)
            - **regnet** -- [`RegNetConfig`] (RegNetConfig model)
            - **rembert** -- [`RemBertConfig`] (RemBertConfig model)
            - **resnet** -- [`ResNetConfig`] (ResNetConfig model)
            - **roberta** -- [`RobertaConfig`] (RobertaConfig model)
            - **roberta-prelayernorm** -- [`RobertaPreLayerNormConfig`] (RobertaPreLayerNormConfig model)
            - **roc_bert** -- [`RoCBertConfig`] (RoCBertConfig model)
            - **roformer** -- [`RoFormerConfig`] (RoFormerConfig model)
            - **rt_detr** -- [`RTDetrConfig`] (RTDetrConfig model)
            - **rt_detr_resnet** -- [`RTDetrResNetConfig`] (RTDetrResNetConfig model)
            - **rt_detr_v2** -- [`RTDetrV2Config`] (RTDetrV2Config model)
            - **rwkv** -- [`RwkvConfig`] (RwkvConfig model)
            - **sam** -- [`SamConfig`] (SamConfig model)
            - **sam2** -- [`Sam2Config`] (Sam2Config model)
            - **sam2_hiera_det_model** -- [`Sam2HieraDetConfig`] (Sam2HieraDetConfig model)
            - **sam2_video** -- [`Sam2VideoConfig`] (Sam2VideoConfig model)
            - **sam2_vision_model** -- [`Sam2VisionConfig`] (Sam2VisionConfig model)
            - **sam3** -- [`Sam3Config`] (Sam3Config model)
            - **sam3_detr_decoder** -- [`Sam3DETRDecoderConfig`] (Sam3DETRDecoderConfig model)
            - **sam3_detr_encoder** -- [`Sam3DETREncoderConfig`] (Sam3DETREncoderConfig model)
            - **sam3_geometry_encoder** -- [`Sam3GeometryEncoderConfig`] (Sam3GeometryEncoderConfig model)
            - **sam3_lite_text** -- [`Sam3LiteTextConfig`] (Sam3LiteTextConfig model)
            - **sam3_lite_text_detr_decoder** -- [`Sam3LiteTextDETRDecoderConfig`] (Sam3LiteTextDETRDecoderConfig model)
            - **sam3_lite_text_detr_encoder** -- [`Sam3LiteTextDETREncoderConfig`] (Sam3LiteTextDETREncoderConfig model)
            - **sam3_lite_text_geometry_encoder** -- [`Sam3LiteTextGeometryEncoderConfig`] (Sam3LiteTextGeometryEncoderConfig model)
            - **sam3_lite_text_mask_decoder** -- [`Sam3LiteTextMaskDecoderConfig`] (Sam3LiteTextMaskDecoderConfig model)
            - **sam3_lite_text_text_model** -- [`Sam3LiteTextTextConfig`] (Sam3LiteTextTextConfig model)
            - **sam3_mask_decoder** -- [`Sam3MaskDecoderConfig`] (Sam3MaskDecoderConfig model)
            - **sam3_tracker** -- [`Sam3TrackerConfig`] (Sam3TrackerConfig model)
            - **sam3_tracker_video** -- [`Sam3TrackerVideoConfig`] (Sam3TrackerVideoConfig model)
            - **sam3_video** -- [`Sam3VideoConfig`] (Sam3VideoConfig model)
            - **sam3_vision_model** -- [`Sam3VisionConfig`] (Sam3VisionConfig model)
            - **sam3_vit_model** -- [`Sam3ViTConfig`] (Sam3ViTConfig model)
            - **sam_hq** -- [`SamHQConfig`] (SamHQConfig model)
            - **sam_hq_vision_model** -- [`SamHQVisionConfig`] (SamHQVisionConfig model)
            - **sam_vision_model** -- [`SamVisionConfig`] (SamVisionConfig model)
            - **seamless_m4t** -- [`SeamlessM4TConfig`] (SeamlessM4TConfig model)
            - **seamless_m4t_v2** -- [`SeamlessM4Tv2Config`] (SeamlessM4Tv2Config model)
            - **seed_oss** -- [`SeedOssConfig`] (SeedOssConfig model)
            - **segformer** -- [`SegformerConfig`] (SegformerConfig model)
            - **seggpt** -- [`SegGptConfig`] (SegGptConfig model)
            - **sew** -- [`SEWConfig`] (SEWConfig model)
            - **sew-d** -- [`SEWDConfig`] (SEWDConfig model)
            - **shieldgemma2** -- [`ShieldGemma2Config`] (ShieldGemma2Config model)
            - **siglip** -- [`SiglipConfig`] (SiglipConfig model)
            - **siglip2** -- [`Siglip2Config`] (Siglip2Config model)
            - **siglip2_text_model** -- [`Siglip2TextConfig`] (Siglip2TextConfig model)
            - **siglip2_vision_model** -- [`Siglip2VisionConfig`] (Siglip2VisionConfig model)
            - **siglip_text_model** -- [`SiglipTextConfig`] (SiglipTextConfig model)
            - **siglip_vision_model** -- [`SiglipVisionConfig`] (SiglipVisionConfig model)
            - **slanet** -- [`SLANetConfig`] (SLANetConfig model)
            - **slanext** -- [`SLANeXtConfig`] (SLANeXtConfig model)
            - **smollm3** -- [`SmolLM3Config`] (SmolLM3Config model)
            - **smolvlm** -- [`SmolVLMConfig`] (SmolVLMConfig model)
            - **smolvlm_vision** -- [`SmolVLMVisionConfig`] (SmolVLMVisionConfig model)
            - **solar_open** -- [`SolarOpenConfig`] (SolarOpenConfig model)
            - **speech-encoder-decoder** -- [`SpeechEncoderDecoderConfig`] (SpeechEncoderDecoderConfig model)
            - **speech_to_text** -- [`Speech2TextConfig`] (Speech2TextConfig model)
            - **speecht5** -- [`SpeechT5Config`] (SpeechT5Config model)
            - **speecht5_hifigan** -- [`SpeechT5HifiGanConfig`] (SpeechT5HifiGanConfig model)
            - **splinter** -- [`SplinterConfig`] (SplinterConfig model)
            - **squeezebert** -- [`SqueezeBertConfig`] (SqueezeBertConfig model)
            - **stablelm** -- [`StableLmConfig`] (StableLmConfig model)
            - **starcoder2** -- [`Starcoder2Config`] (Starcoder2Config model)
            - **superglue** -- [`SuperGlueConfig`] (SuperGlueConfig model)
            - **superpoint** -- [`SuperPointConfig`] (SuperPointConfig model)
            - **swiftformer** -- [`SwiftFormerConfig`] (SwiftFormerConfig model)
            - **swin** -- [`SwinConfig`] (SwinConfig model)
            - **swin2sr** -- [`Swin2SRConfig`] (Swin2SRConfig model)
            - **swinv2** -- [`Swinv2Config`] (Swinv2Config model)
            - **switch_transformers** -- [`SwitchTransformersConfig`] (SwitchTransformersConfig model)
            - **t5** -- [`T5Config`] (T5Config model)
            - **t5_gemma_module** -- [`T5GemmaModuleConfig`] (T5GemmaModuleConfig model)
            - **t5gemma** -- [`T5GemmaConfig`] (T5GemmaConfig model)
            - **t5gemma2** -- [`T5Gemma2Config`] (T5Gemma2Config model)
            - **t5gemma2_decoder** -- [`T5Gemma2DecoderConfig`] (T5Gemma2DecoderConfig model)
            - **t5gemma2_encoder** -- [`T5Gemma2EncoderConfig`] (T5Gemma2EncoderConfig model)
            - **t5gemma2_text** -- [`T5Gemma2TextConfig`] (T5Gemma2TextConfig model)
            - **table-transformer** -- [`TableTransformerConfig`] (TableTransformerConfig model)
            - **tapas** -- [`TapasConfig`] (TapasConfig model)
            - **textnet** -- [`TextNetConfig`] (TextNetConfig model)
            - **time_series_transformer** -- [`TimeSeriesTransformerConfig`] (TimeSeriesTransformerConfig model)
            - **timesfm** -- [`TimesFmConfig`] (TimesFmConfig model)
            - **timesfm2_5** -- [`TimesFm2_5Config`] (TimesFm2_5Config model)
            - **timesformer** -- [`TimesformerConfig`] (TimesformerConfig model)
            - **timm_backbone** -- [`TimmBackboneConfig`] (TimmBackboneConfig model)
            - **timm_wrapper** -- [`TimmWrapperConfig`] (TimmWrapperConfig model)
            - **trocr** -- [`TrOCRConfig`] (TrOCRConfig model)
            - **tvp** -- [`TvpConfig`] (TvpConfig model)
            - **udop** -- [`UdopConfig`] (UdopConfig model)
            - **umt5** -- [`UMT5Config`] (UMT5Config model)
            - **unispeech** -- [`UniSpeechConfig`] (UniSpeechConfig model)
            - **unispeech-sat** -- [`UniSpeechSatConfig`] (UniSpeechSatConfig model)
            - **univnet** -- [`UnivNetConfig`] (UnivNetConfig model)
            - **upernet** -- [`UperNetConfig`] (UperNetConfig model)
            - **uvdoc** -- [`UVDocConfig`] (UVDocConfig model)
            - **uvdoc_backbone** -- [`UVDocBackboneConfig`] (UVDocBackboneConfig model)
            - **vaultgemma** -- [`VaultGemmaConfig`] (VaultGemmaConfig model)
            - **vibevoice_acoustic_tokenizer** -- [`VibeVoiceAcousticTokenizerConfig`] (VibeVoiceAcousticTokenizerConfig model)
            - **vibevoice_acoustic_tokenizer_decoder** -- [`VibeVoiceAcousticTokenizerDecoderConfig`] (VibeVoiceAcousticTokenizerDecoderConfig model)
            - **vibevoice_acoustic_tokenizer_encoder** -- [`VibeVoiceAcousticTokenizerEncoderConfig`] (VibeVoiceAcousticTokenizerEncoderConfig model)
            - **vibevoice_asr** -- [`VibeVoiceAsrConfig`] (VibeVoiceAsrConfig model)
            - **video_llama_3** -- [`VideoLlama3Config`] (VideoLlama3Config model)
            - **video_llama_3_vision** -- [`VideoLlama3VisionConfig`] (VideoLlama3VisionConfig model)
            - **video_llava** -- [`VideoLlavaConfig`] (VideoLlavaConfig model)
            - **videomae** -- [`VideoMAEConfig`] (VideoMAEConfig model)
            - **videomt** -- [`VideomtConfig`] (VideomtConfig model)
            - **vilt** -- [`ViltConfig`] (ViltConfig model)
            - **vipllava** -- [`VipLlavaConfig`] (VipLlavaConfig model)
            - **vision-encoder-decoder** -- [`VisionEncoderDecoderConfig`] (VisionEncoderDecoderConfig model)
            - **vision-text-dual-encoder** -- [`VisionTextDualEncoderConfig`] (VisionTextDualEncoderConfig model)
            - **visual_bert** -- [`VisualBertConfig`] (VisualBertConfig model)
            - **vit** -- [`ViTConfig`] (ViTConfig model)
            - **vit_mae** -- [`ViTMAEConfig`] (ViTMAEConfig model)
            - **vit_msn** -- [`ViTMSNConfig`] (ViTMSNConfig model)
            - **vitdet** -- [`VitDetConfig`] (VitDetConfig model)
            - **vitmatte** -- [`VitMatteConfig`] (VitMatteConfig model)
            - **vitpose** -- [`VitPoseConfig`] (VitPoseConfig model)
            - **vitpose_backbone** -- [`VitPoseBackboneConfig`] (VitPoseBackboneConfig model)
            - **vits** -- [`VitsConfig`] (VitsConfig model)
            - **vivit** -- [`VivitConfig`] (VivitConfig model)
            - **vjepa2** -- [`VJEPA2Config`] (VJEPA2Config model)
            - **voxtral** -- [`VoxtralConfig`] (VoxtralConfig model)
            - **voxtral_encoder** -- [`VoxtralEncoderConfig`] (VoxtralEncoderConfig model)
            - **voxtral_realtime** -- [`VoxtralRealtimeConfig`] (VoxtralRealtimeConfig model)
            - **voxtral_realtime_encoder** -- [`VoxtralRealtimeEncoderConfig`] (VoxtralRealtimeEncoderConfig model)
            - **voxtral_realtime_text** -- [`VoxtralRealtimeTextConfig`] (VoxtralRealtimeTextConfig model)
            - **wav2vec2** -- [`Wav2Vec2Config`] (Wav2Vec2Config model)
            - **wav2vec2-bert** -- [`Wav2Vec2BertConfig`] (Wav2Vec2BertConfig model)
            - **wav2vec2-conformer** -- [`Wav2Vec2ConformerConfig`] (Wav2Vec2ConformerConfig model)
            - **wavlm** -- [`WavLMConfig`] (WavLMConfig model)
            - **whisper** -- [`WhisperConfig`] (WhisperConfig model)
            - **xclip** -- [`XCLIPConfig`] (XCLIPConfig model)
            - **xclip_text_model** -- [`XCLIPTextConfig`] (XCLIPTextConfig model)
            - **xclip_vision_model** -- [`XCLIPVisionConfig`] (XCLIPVisionConfig model)
            - **xcodec** -- [`XcodecConfig`] (XcodecConfig model)
            - **xglm** -- [`XGLMConfig`] (XGLMConfig model)
            - **xlm** -- [`XLMConfig`] (XLMConfig model)
            - **xlm-roberta** -- [`XLMRobertaConfig`] (XLMRobertaConfig model)
            - **xlm-roberta-xl** -- [`XLMRobertaXLConfig`] (XLMRobertaXLConfig model)
            - **xlnet** -- [`XLNetConfig`] (XLNetConfig model)
            - **xlstm** -- [`xLSTMConfig`] (xLSTMConfig model)
            - **xmod** -- [`XmodConfig`] (XmodConfig model)
            - **yolos** -- [`YolosConfig`] (YolosConfig model)
            - **yoso** -- [`YosoConfig`] (YosoConfig model)
            - **youtu** -- [`YoutuConfig`] (YoutuConfig model)
            - **zamba** -- [`ZambaConfig`] (ZambaConfig model)
            - **zamba2** -- [`Zamba2Config`] (Zamba2Config model)
            - **zoedepth** -- [`ZoeDepthConfig`] (ZoeDepthConfig model)
        
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:
        
                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PreTrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - a path to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.
        
                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.
        
        Examples:
        
        ```python
        >>> from transformers import AutoConfig
        
        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
        
        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")
        
        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")
        
        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")
        
        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True
        
        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True
        
        >>> unused_kwargs
        {'foo': False}
        ```

    `register(model_type, config, exist_ok=False) ‑> None`
    :   Register a new configuration for this class.
        
        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PreTrainedConfig`]): The config to register.

`AutoModelForSequenceClassification(*args, **kwargs)`
:   This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created
    with the [`~AutoModelForSequenceClassification.from_pretrained`] class method or the [`~AutoModelForSequenceClassification.from_config`] class
    method.
    
    This class cannot be instantiated directly using `__init__()` (throws an error).

    ### Ancestors (in MRO)

    * transformers.models.auto.auto_factory._BaseAutoModelClass

    ### Static methods

    `from_config(config, **kwargs)`
    :

    `from_pretrained(pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs)`
    :

`AutoTokenizer()`
:   This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.
    
    This class cannot be instantiated directly using `__init__()` (throws an error).

    ### Static methods

    `from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs) ‑> transformers.tokenization_utils_tokenizers.TokenizersBackend | None`
    :   Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.
        
        The tokenizer class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:
        
            - **aimv2** -- [`CLIPTokenizer`] (Aimv2Config model)
            - **albert** -- [`AlbertTokenizer`] (AlbertConfig model)
            - **align** -- [`BertTokenizer`] (AlignConfig model)
            - **audioflamingo3** -- [`Qwen2Tokenizer`] (AudioFlamingo3Config model)
            - **aya_vision** -- [`CohereTokenizer`] (AyaVisionConfig model)
            - **bark** -- [`BertTokenizer`] (BarkConfig model)
            - **bart** -- [`RobertaTokenizer`] (BartConfig model)
            - **bert** -- [`BertTokenizer`] (BertConfig model)
            - **bert-generation** -- [`None`] (BertGenerationConfig model)
            - **big_bird** -- [`BigBirdTokenizer`] (BigBirdConfig model)
            - **bigbird_pegasus** -- [`PegasusTokenizer`] (BigBirdPegasusConfig model)
            - **biogpt** -- [`BioGptTokenizer`] (BioGptConfig model)
            - **blenderbot** -- [`BlenderbotTokenizer`] (BlenderbotConfig model)
            - **blenderbot-small** -- [`BlenderbotSmallTokenizer`] (BlenderbotSmallConfig model)
            - **blip** -- [`BertTokenizer`] (BlipConfig model)
            - **blip-2** -- [`GPT2Tokenizer`] (Blip2Config model)
            - **bridgetower** -- [`RobertaTokenizer`] (BridgeTowerConfig model)
            - **bros** -- [`BertTokenizer`] (BrosConfig model)
            - **camembert** -- [`CamembertTokenizer`] (CamembertConfig model)
            - **canine** -- [`CanineTokenizer`] (CanineConfig model)
            - **chameleon** -- [`TokenizersBackend`] (ChameleonConfig model)
            - **chinese_clip** -- [`BertTokenizer`] (ChineseCLIPConfig model)
            - **clap** -- [`RobertaTokenizer`] (ClapConfig model)
            - **clip** -- [`CLIPTokenizer`] (CLIPConfig model)
            - **clipseg** -- [`CLIPTokenizer`] (CLIPSegConfig model)
            - **clvp** -- [`ClvpTokenizer`] (ClvpConfig model)
            - **codegen** -- [`GPT2Tokenizer`] (CodeGenConfig model)
            - **cohere** -- [`CohereTokenizer`] (CohereConfig model)
            - **cohere2** -- [`CohereTokenizer`] (Cohere2Config model)
            - **cohere_asr** -- [`TokenizersBackend`] (CohereAsrConfig model)
            - **colqwen2** -- [`Qwen2Tokenizer`] (ColQwen2Config model)
            - **convbert** -- [`BertTokenizer`] (ConvBertConfig model)
            - **cpmant** -- [`CpmAntTokenizer`] (CpmAntConfig model)
            - **ctrl** -- [`CTRLTokenizer`] (CTRLConfig model)
            - **data2vec-audio** -- [`Wav2Vec2CTCTokenizer`] (Data2VecAudioConfig model)
            - **data2vec-text** -- [`RobertaTokenizer`] (Data2VecTextConfig model)
            - **dbrx** -- [`GPT2Tokenizer`] (DbrxConfig model)
            - **deberta** -- [`DebertaTokenizer`] (DebertaConfig model)
            - **deberta-v2** -- [`DebertaV2Tokenizer`] (DebertaV2Config model)
            - **deepseek_v2** -- [`TokenizersBackend`] (DeepseekV2Config model)
            - **deepseek_v3** -- [`TokenizersBackend`] (DeepseekV3Config model)
            - **deepseek_vl** -- [`TokenizersBackend`] (DeepseekVLConfig model)
            - **deepseek_vl_hybrid** -- [`TokenizersBackend`] (DeepseekVLHybridConfig model)
            - **dia** -- [`DiaTokenizer`] (DiaConfig model)
            - **distilbert** -- [`BertTokenizer`] (DistilBertConfig model)
            - **dpr** -- [`DPRQuestionEncoderTokenizer`] (DPRConfig model)
            - **electra** -- [`BertTokenizer`] (ElectraConfig model)
            - **emu3** -- [`GPT2Tokenizer`] (Emu3Config model)
            - **ernie** -- [`BertTokenizer`] (ErnieConfig model)
            - **esm** -- [`EsmTokenizer`] (EsmConfig model)
            - **falcon_mamba** -- [`GPTNeoXTokenizer`] (FalconMambaConfig model)
            - **fastspeech2_conformer** -- [`None`] (FastSpeech2ConformerConfig model)
            - **flaubert** -- [`FlaubertTokenizer`] (FlaubertConfig model)
            - **flava** -- [`BertTokenizer`] (FlavaConfig model)
            - **flex_olmo** -- [`GPT2Tokenizer`] (FlexOlmoConfig model)
            - **florence2** -- [`BartTokenizer`] (Florence2Config model)
            - **fnet** -- [`FNetTokenizer`] (FNetConfig model)
            - **fsmt** -- [`FSMTTokenizer`] (FSMTConfig model)
            - **funnel** -- [`FunnelTokenizer`] (FunnelConfig model)
            - **fuyu** -- [`TokenizersBackend`] (FuyuConfig model)
            - **gemma** -- [`GemmaTokenizer`] (GemmaConfig model)
            - **gemma2** -- [`GemmaTokenizer`] (Gemma2Config model)
            - **gemma3** -- [`GemmaTokenizer`] (Gemma3Config model)
            - **gemma3_text** -- [`GemmaTokenizer`] (Gemma3TextConfig model)
            - **gemma3n** -- [`GemmaTokenizer`] (Gemma3nConfig model)
            - **gemma3n_text** -- [`GemmaTokenizer`] (Gemma3nTextConfig model)
            - **git** -- [`BertTokenizer`] (GitConfig model)
            - **glm** -- [`TokenizersBackend`] (GlmConfig model)
            - **glm4** -- [`TokenizersBackend`] (Glm4Config model)
            - **glm4_moe** -- [`TokenizersBackend`] (Glm4MoeConfig model)
            - **glm4_moe_lite** -- [`TokenizersBackend`] (Glm4MoeLiteConfig model)
            - **glm4v** -- [`TokenizersBackend`] (Glm4vConfig model)
            - **glm4v_moe** -- [`TokenizersBackend`] (Glm4vMoeConfig model)
            - **glm_image** -- [`TokenizersBackend`] (GlmImageConfig model)
            - **glmasr** -- [`TokenizersBackend`] (GlmAsrConfig model)
            - **got_ocr2** -- [`TokenizersBackend`] (GotOcr2Config model)
            - **gpt-sw3** -- [`None`] (GPT2Config model)
            - **gpt2** -- [`GPT2Tokenizer`] (GPT2Config model)
            - **gpt_bigcode** -- [`GPT2Tokenizer`] (GPTBigCodeConfig model)
            - **gpt_neo** -- [`GPT2Tokenizer`] (GPTNeoConfig model)
            - **gpt_neox** -- [`GPTNeoXTokenizer`] (GPTNeoXConfig model)
            - **gpt_neox_japanese** -- [`GPTNeoXJapaneseTokenizer`] (GPTNeoXJapaneseConfig model)
            - **gptj** -- [`GPT2Tokenizer`] (GPTJConfig model)
            - **granite** -- [`GPT2Tokenizer`] (GraniteConfig model)
            - **granitemoe** -- [`GPT2Tokenizer`] (GraniteMoeConfig model)
            - **granitemoehybrid** -- [`GPT2Tokenizer`] (GraniteMoeHybridConfig model)
            - **granitemoeshared** -- [`GPT2Tokenizer`] (GraniteMoeSharedConfig model)
            - **grounding-dino** -- [`BertTokenizer`] (GroundingDinoConfig model)
            - **groupvit** -- [`CLIPTokenizer`] (GroupViTConfig model)
            - **hubert** -- [`Wav2Vec2CTCTokenizer`] (HubertConfig model)
            - **ibert** -- [`RobertaTokenizer`] (IBertConfig model)
            - **idefics** -- [`LlamaTokenizer`] (IdeficsConfig model)
            - **idefics2** -- [`LlamaTokenizer`] (Idefics2Config model)
            - **instructblip** -- [`GPT2Tokenizer`] (InstructBlipConfig model)
            - **instructblipvideo** -- [`GPT2Tokenizer`] (InstructBlipVideoConfig model)
            - **internvl** -- [`Qwen2Tokenizer`] (InternVLConfig model)
            - **jais2** -- [`GPT2Tokenizer`] (Jais2Config model)
            - **jamba** -- [`TokenizersBackend`] (JambaConfig model)
            - **janus** -- [`TokenizersBackend`] (JanusConfig model)
            - **jina_embeddings_v3** -- [`XLMRobertaTokenizer`] (JinaEmbeddingsV3Config model)
            - **kosmos-2** -- [`XLMRobertaTokenizer`] (Kosmos2Config model)
            - **lasr_ctc** -- [`LasrTokenizer`] (LasrCTCConfig model)
            - **lasr_encoder** -- [`LasrTokenizer`] (LasrEncoderConfig model)
            - **layoutlm** -- [`BertTokenizer`] (LayoutLMConfig model)
            - **layoutlmv2** -- [`LayoutLMv2Tokenizer`] (LayoutLMv2Config model)
            - **layoutlmv3** -- [`LayoutLMv3Tokenizer`] (LayoutLMv3Config model)
            - **layoutxlm** -- [`LayoutXLMTokenizer`] (LayoutXLMConfig model)
            - **led** -- [`LEDTokenizer`] (LEDConfig model)
            - **lighton_ocr** -- [`Qwen2TokenizerFast`] (LightOnOcrConfig model)
            - **lilt** -- [`RobertaTokenizer`] (LiltConfig model)
            - **llava** -- [`TokenizersBackend`] (LlavaConfig model)
            - **llava_next** -- [`TokenizersBackend`] (LlavaNextConfig model)
            - **longformer** -- [`RobertaTokenizer`] (LongformerConfig model)
            - **luke** -- [`LukeTokenizer`] (LukeConfig model)
            - **lxmert** -- [`LxmertTokenizer`] (LxmertConfig model)
            - **m2m_100** -- [`None`] (M2M100Config model)
            - **mamba** -- [`GPTNeoXTokenizer`] (MambaConfig model)
            - **mamba2** -- [`GPTNeoXTokenizer`] (Mamba2Config model)
            - **marian** -- [`None`] (MarianConfig model)
            - **markuplm** -- [`MarkupLMTokenizer`] (MarkupLMConfig model)
            - **mbart** -- [`MBartTokenizer`] (MBartConfig model)
            - **megatron-bert** -- [`BertTokenizer`] (MegatronBertConfig model)
            - **metaclip_2** -- [`XLMRobertaTokenizer`] (MetaClip2Config model)
            - **mgp-str** -- [`MgpstrTokenizer`] (MgpstrConfig model)
            - **minicpmv4_6** -- [`TokenizersBackend`] (MiniCPMV4_6Config model)
            - **minimax_m2** -- [`TokenizersBackend`] (MiniMaxM2Config model)
            - **ministral** -- [`TokenizersBackend`] (MinistralConfig model)
            - **ministral3** -- [`TokenizersBackend`] (Ministral3Config model)
            - **mistral** -- [`TokenizersBackend`] (MistralConfig model)
            - **mistral3** -- [`TokenizersBackend`] (Mistral3Config model)
            - **mixtral** -- [`TokenizersBackend`] (MixtralConfig model)
            - **mm-grounding-dino** -- [`BertTokenizer`] (MMGroundingDinoConfig model)
            - **mobilebert** -- [`MobileBertTokenizer`] (MobileBertConfig model)
            - **modernbert** -- [`TokenizersBackend`] (ModernBertConfig model)
            - **mpnet** -- [`MPNetTokenizer`] (MPNetConfig model)
            - **mpt** -- [`GPTNeoXTokenizer`] (MptConfig model)
            - **mra** -- [`RobertaTokenizer`] (MraConfig model)
            - **mt5** -- [`T5Tokenizer`] (MT5Config model)
            - **musicgen** -- [`T5Tokenizer`] (MusicgenConfig model)
            - **musicgen_melody** -- [`T5Tokenizer`] (MusicgenMelodyConfig model)
            - **mvp** -- [`MvpTokenizer`] (MvpConfig model)
            - **nemotron** -- [`TokenizersBackend`] (NemotronConfig model)
            - **nllb-moe** -- [`NllbTokenizer`] (NllbMoeConfig model)
            - **nomic_bert** -- [`BertTokenizer`] (NomicBertConfig model)
            - **nougat** -- [`NougatTokenizer`] (NougatConfig model)
            - **nystromformer** -- [`AlbertTokenizer`] (NystromformerConfig model)
            - **olmo** -- [`GPTNeoXTokenizer`] (OlmoConfig model)
            - **olmo2** -- [`GPTNeoXTokenizer`] (Olmo2Config model)
            - **olmo3** -- [`TokenizersBackend`] (Olmo3Config model)
            - **olmo_hybrid** -- [`TokenizersBackend`] (OlmoHybridConfig model)
            - **olmoe** -- [`GPTNeoXTokenizer`] (OlmoeConfig model)
            - **omdet-turbo** -- [`CLIPTokenizer`] (OmDetTurboConfig model)
            - **oneformer** -- [`CLIPTokenizer`] (OneFormerConfig model)
            - **openai-gpt** -- [`OpenAIGPTTokenizer`] (OpenAIGPTConfig model)
            - **opt** -- [`GPT2Tokenizer`] (OPTConfig model)
            - **ovis2** -- [`Qwen2Tokenizer`] (Ovis2Config model)
            - **owlv2** -- [`CLIPTokenizer`] (Owlv2Config model)
            - **owlvit** -- [`CLIPTokenizer`] (OwlViTConfig model)
            - **pegasus** -- [`PegasusTokenizer`] (PegasusConfig model)
            - **pegasus_x** -- [`PegasusTokenizer`] (PegasusXConfig model)
            - **perceiver** -- [`PerceiverTokenizer`] (PerceiverConfig model)
            - **phi** -- [`GPT2Tokenizer`] (PhiConfig model)
            - **phi3** -- [`TokenizersBackend`] (Phi3Config model)
            - **phimoe** -- [`TokenizersBackend`] (PhimoeConfig model)
            - **pix2struct** -- [`T5Tokenizer`] (Pix2StructConfig model)
            - **pixtral** -- [`TokenizersBackend`] (PixtralVisionConfig model)
            - **plbart** -- [`PLBartTokenizer`] (PLBartConfig model)
            - **prophetnet** -- [`ProphetNetTokenizer`] (ProphetNetConfig model)
            - **qianfan_ocr** -- [`Qwen2Tokenizer`] (QianfanOCRConfig model)
            - **qwen2** -- [`Qwen2Tokenizer`] (Qwen2Config model)
            - **qwen2_5_omni** -- [`Qwen2Tokenizer`] (Qwen2_5OmniConfig model)
            - **qwen2_5_vl** -- [`Qwen2Tokenizer`] (Qwen2_5_VLConfig model)
            - **qwen2_audio** -- [`Qwen2Tokenizer`] (Qwen2AudioConfig model)
            - **qwen2_moe** -- [`Qwen2Tokenizer`] (Qwen2MoeConfig model)
            - **qwen2_vl** -- [`Qwen2Tokenizer`] (Qwen2VLConfig model)
            - **qwen3** -- [`Qwen2Tokenizer`] (Qwen3Config model)
            - **qwen3_5** -- [`Qwen3_5Tokenizer`] (Qwen3_5Config model)
            - **qwen3_5_moe** -- [`Qwen3_5Tokenizer`] (Qwen3_5MoeConfig model)
            - **qwen3_moe** -- [`Qwen2Tokenizer`] (Qwen3MoeConfig model)
            - **qwen3_next** -- [`Qwen2Tokenizer`] (Qwen3NextConfig model)
            - **qwen3_omni_moe** -- [`Qwen2Tokenizer`] (Qwen3OmniMoeConfig model)
            - **qwen3_vl** -- [`Qwen2Tokenizer`] (Qwen3VLConfig model)
            - **qwen3_vl_moe** -- [`Qwen2Tokenizer`] (Qwen3VLMoeConfig model)
            - **rag** -- [`RagTokenizer`] (RagConfig model)
            - **recurrent_gemma** -- [`GemmaTokenizer`] (RecurrentGemmaConfig model)
            - **reformer** -- [`ReformerTokenizer`] (ReformerConfig model)
            - **rembert** -- [`RemBertTokenizer`] (RemBertConfig model)
            - **roberta** -- [`RobertaTokenizer`] (RobertaConfig model)
            - **roberta-prelayernorm** -- [`RobertaTokenizer`] (RobertaPreLayerNormConfig model)
            - **roc_bert** -- [`RoCBertTokenizer`] (RoCBertConfig model)
            - **roformer** -- [`RoFormerTokenizer`] (RoFormerConfig model)
            - **rwkv** -- [`GPTNeoXTokenizer`] (RwkvConfig model)
            - **sam3** -- [`CLIPTokenizer`] (Sam3Config model)
            - **sam3_video** -- [`CLIPTokenizer`] (Sam3VideoConfig model)
            - **seamless_m4t** -- [`SeamlessM4TTokenizer`] (SeamlessM4TConfig model)
            - **seamless_m4t_v2** -- [`SeamlessM4TTokenizer`] (SeamlessM4Tv2Config model)
            - **shieldgemma2** -- [`GemmaTokenizer`] (ShieldGemma2Config model)
            - **siglip** -- [`None`] (SiglipConfig model)
            - **siglip2** -- [`Siglip2Tokenizer`] (Siglip2Config model)
            - **speech_to_text** -- [`None`] (Speech2TextConfig model)
            - **speecht5** -- [`None`] (SpeechT5Config model)
            - **splinter** -- [`SplinterTokenizer`] (SplinterConfig model)
            - **squeezebert** -- [`BertTokenizer`] (SqueezeBertConfig model)
            - **stablelm** -- [`GPTNeoXTokenizer`] (StableLmConfig model)
            - **starcoder2** -- [`GPT2Tokenizer`] (Starcoder2Config model)
            - **switch_transformers** -- [`T5Tokenizer`] (SwitchTransformersConfig model)
            - **t5** -- [`T5Tokenizer`] (T5Config model)
            - **t5gemma** -- [`GemmaTokenizer`] (T5GemmaConfig model)
            - **tapas** -- [`TapasTokenizer`] (TapasConfig model)
            - **trocr** -- [`XLMRobertaTokenizer`] (TrOCRConfig model)
            - **tvp** -- [`BertTokenizer`] (TvpConfig model)
            - **udop** -- [`UdopTokenizer`] (UdopConfig model)
            - **umt5** -- [`T5Tokenizer`] (UMT5Config model)
            - **unispeech** -- [`Wav2Vec2CTCTokenizer`] (UniSpeechConfig model)
            - **unispeech-sat** -- [`Wav2Vec2CTCTokenizer`] (UniSpeechSatConfig model)
            - **vilt** -- [`BertTokenizer`] (ViltConfig model)
            - **vipllava** -- [`TokenizersBackend`] (VipLlavaConfig model)
            - **visual_bert** -- [`BertTokenizer`] (VisualBertConfig model)
            - **vits** -- [`VitsTokenizer`] (VitsConfig model)
            - **voxtral** -- [`TokenizersBackend`] (VoxtralConfig model)
            - **voxtral_realtime** -- [`TokenizersBackend`] (VoxtralRealtimeConfig model)
            - **wav2vec2** -- [`Wav2Vec2CTCTokenizer`] (Wav2Vec2Config model)
            - **wav2vec2-bert** -- [`Wav2Vec2CTCTokenizer`] (Wav2Vec2BertConfig model)
            - **wav2vec2-conformer** -- [`Wav2Vec2CTCTokenizer`] (Wav2Vec2ConformerConfig model)
            - **whisper** -- [`WhisperTokenizer`] (WhisperConfig model)
            - **xclip** -- [`CLIPTokenizer`] (XCLIPConfig model)
            - **xglm** -- [`XGLMTokenizer`] (XGLMConfig model)
            - **xlm** -- [`XLMTokenizer`] (XLMConfig model)
            - **xlm-roberta** -- [`XLMRobertaTokenizer`] (XLMRobertaConfig model)
            - **xlm-roberta-xl** -- [`XLMRobertaTokenizer`] (XLMRobertaXLConfig model)
            - **xlnet** -- [`XLNetTokenizer`] (XLNetConfig model)
            - **xlstm** -- [`GPTNeoXTokenizer`] (xLSTMConfig model)
            - **xmod** -- [`XLMRobertaTokenizer`] (XmodConfig model)
            - **yoso** -- [`AlbertTokenizer`] (YosoConfig model)
        
        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:
        
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - a path to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PreTrainedConfig`], *optional*)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            tokenizer_type (`str`, *optional*):
                Tokenizer type to be loaded.
            backend (`str`, *optional*, defaults to `"tokenizers"`):
                Backend to use for tokenization. Valid options are:
                - `"tokenizers"`: Use the HuggingFace tokenizers library backend (default)
                - `"sentencepiece"`: Use the SentencePiece backend
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
                `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__()` for more details.
        
        Examples:
        
        ```python
        >>> from transformers import AutoTokenizer
        
        >>> # Download vocabulary from huggingface.co and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        
        >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        
        >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        >>> # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")
        
        >>> # Download vocabulary from huggingface.co and define model-specific arguments
        >>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)
        
        >>> # Explicitly use the tokenizers backend
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="tokenizers")
        
        >>> # Explicitly use the sentencepiece backend
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="sentencepiece")
        ```

    `register(config_class, tokenizer_class=None, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False)`
    :   Register a new tokenizer in this mapping.
        
        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            tokenizer_class: The tokenizer class to register (V5 - preferred parameter).
            slow_tokenizer_class: (Deprecated) The slow tokenizer to register.
            fast_tokenizer_class: (Deprecated) The fast tokenizer to register.

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

`BaseAttention(config: Any | None = None, rank: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.adaptive_span_attention.AdaptiveSpanAttention
    * neuralops.attentions.adaptive_sparse_attention.AdaptiveSparseAttention
    * neuralops.attentions.adaptive_svd_attention.AdaptiveSVDAttention
    * neuralops.attentions.adasplash_attention.AdaSplashAttention
    * neuralops.attentions.aegis_hybrid.AegisHybridAttention
    * neuralops.attentions.aegis_spectron_fp8.AegisSpectronFP8Attention
    * neuralops.attentions.aegis_v2.AegisAttentionV2
    * neuralops.attentions.aegis_v5.AegisAttention
    * neuralops.attentions.aegis_v6.AegisAttentionV3
    * neuralops.attentions.asvd_attention.ASVDAttention
    * neuralops.attentions.bam_umoe.BAMUMoEBlock
    * neuralops.attentions.block_transformer_hiba.BlockTransformerHIBA
    * neuralops.attentions.chela_slay.ChelaSlayAttention
    * neuralops.attentions.cluster_attention.ClusterAttention
    * neuralops.attentions.compressed_attention.CompressedAttentionLayer
    * neuralops.attentions.compressed_mha_attention.CompressedDistilBertAttention
    * neuralops.attentions.cosformer.CosformerAttention
    * neuralops.attentions.decay_linear_attention.DecayLinearAttention
    * neuralops.attentions.eat.EATBlock
    * neuralops.attentions.expert_choice_moe.ExpertChoiceMoEAttention
    * neuralops.attentions.factorised_matrix_state.FactorisedMatrixState
    * neuralops.attentions.flash_attention3.FlashAttention3
    * neuralops.attentions.flash_attention3_fp8.FlashAttention3FP8
    * neuralops.attentions.full_attention.FullAttention
    * neuralops.attentions.fused_svd_kernel.FusedSVDKernelAttention
    * neuralops.attentions.fusion_attn.FUSIONATTN
    * neuralops.attentions.galerkin_attention.GalerkinAttention
    * neuralops.attentions.gated_delta_net.GatedDeltaNet
    * neuralops.attentions.gated_deltanet_block.GatedDeltaNetBlock
    * neuralops.attentions.gated_linear_attention.GatedLinearAttention
    * neuralops.attentions.goldfinch_ahn.GoldFinchAHNBlock
    * neuralops.attentions.gqa_attention.GroupedQueryAttention
    * neuralops.attentions.granite_hybrid.GraniteHybridBlock
    * neuralops.attentions.griffin_ssd.GriffinSSDBlock
    * neuralops.attentions.halo_regla.HALOReGLAAttention
    * neuralops.attentions.hamming_attention.HammingAttention
    * neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet
    * neuralops.attentions.hgrn2_layer.HGRN2Layer
    * neuralops.attentions.hgrn2_outer.HGRN2OuterBlock
    * neuralops.attentions.hla_attention.HLAAttention
    * neuralops.attentions.hybrid_attention.HybridAttention
    * neuralops.attentions.hymba_transx.HymbaTransXBlock
    * neuralops.attentions.infini_attention.InfiniAttention
    * neuralops.attentions.jamba.Jamba2Block
    * neuralops.attentions.jamba_info.JambaInfoBlock
    * neuralops.attentions.kda_attention.KDAAttention
    * neuralops.attentions.leka_v1.LEKAV1Attention
    * neuralops.attentions.leka_v2.LEKAv2Attention
    * neuralops.attentions.leka_v3.LEKAV3Attention
    * neuralops.attentions.linformer_attention.LinformerAttention
    * neuralops.attentions.lola_attention.LoLAAttention
    * neuralops.attentions.lora_attention.LoRAAttention
    * neuralops.attentions.lsh_attention.LSHAttention
    * neuralops.attentions.mamba_svd_flash.MambaSVDFlash
    * neuralops.attentions.minimal_nova.MinimalNOVAAttention
    * neuralops.attentions.minimal_svd.MinimalSVDAttention
    * neuralops.attentions.mixattention_moska.MixAttentionMoSKA
    * neuralops.attentions.mla_attention.MLAAttention
    * neuralops.attentions.moba_attention.MoBAAttention
    * neuralops.attentions.nasc_attention.NASCAttention
    * neuralops.attentions.nexus.NEXUSAttention
    * neuralops.attentions.nha_attention.NativeHybridAttention
    * neuralops.attentions.nsa_attention.NativeSparseAttention
    * neuralops.attentions.nystrom_attention.NystromAttention
    * neuralops.attentions.optimized_hybrid_attention.OptimizedHybridAttention
    * neuralops.attentions.orion.ORIONAttention
    * neuralops.attentions.padre_attention.PADReAttention
    * neuralops.attentions.performer_attention.PerformerAttention
    * neuralops.attentions.phantom.PHANTOMAttention
    * neuralops.attentions.primal_attention.PrimalAttention
    * neuralops.attentions.progressive_token_pruning.ProgressiveTokenPruning
    * neuralops.attentions.qcqa_attention.QCQAAttention
    * neuralops.attentions.rala_attention.RALALinearAttention
    * neuralops.attentions.reap_moe.REAPMoEAttention
    * neuralops.attentions.regla_attention.ReGLAAttention
    * neuralops.attentions.replace.ReplaceAttention
    * neuralops.attentions.res_svd.ResSVDAttention
    * neuralops.attentions.routing_attention.RoutingAttention
    * neuralops.attentions.rpc_attention.RPCAttention
    * neuralops.attentions.salt_attention.SALTAttention
    * neuralops.attentions.samba_moe.SambaMoEBlock
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.soa_attn.SOAAttention
    * neuralops.attentions.span_expanded_attention.SpanExpandedAttention
    * neuralops.attentions.spectra_v2.SpectraV2Attention
    * neuralops.attentions.spectraformer_attention.SpectraformerAttention
    * neuralops.attentions.spectral_mamba.SpectralMambaAttention
    * neuralops.attentions.spectron.SpectronAttention
    * neuralops.attentions.squeezing_heads_distill.SqueezingHeadsDistiller
    * neuralops.attentions.svd_attention.SVDLinearAttention
    * neuralops.attentions.svd_compressed.SVDCompressedAttention
    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.attentions.svd_llm_v2_attention.SVDLLMV2Attention
    * neuralops.attentions.svd_mamba.SVDMamba
    * neuralops.attentions.svd_projection_attention.SVDProjectionAttention
    * neuralops.attentions.template_attention.YourCustomAttention
    * neuralops.attentions.tlinformer_attention.TLinFormerAttention
    * neuralops.attentions.token_compressed_attention.TokenCompressedAttention
    * neuralops.attentions.tpa_attention.TPAttention
    * neuralops.attentions.triton_cosformer.TritonCosformerAttention
    * neuralops.attentions.triton_svd_linear_attention.TritonSVDLinearAttention
    * neuralops.attentions.unified_rope.UnifiedRoPE
    * neuralops.attentions.upi_stuffed_mamba.SelectiveMamba
    * neuralops.attentions.vig_attention.ViGAttention
    * neuralops.attentions.vspec.VSpecAttention
    * neuralops.attentions.wgqa_attention.WeightedGroupedQueryAttention
    * neuralops.attentions.xattention.XAttention
    * neuralops.attentions.zeros_attention.ZeroSAttention
    * neuralops.compression_engine.nli_softmax.NLISoftmaxAttention
    * neuralops.core.replace_attention.ReplaceAttention

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `extra_repr(self) ‑> str`
    :   Extra repr.

    `forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs) ‑> tuple[typing.Any, Any | None]`
    :   Must return (output, attention_weights_or_None).

`BlockTransformerHIBA(config: Any | None = None, rank: int = 64, block_size: int = 64, top_k_inter: int = 4, rectification_interval: int = 4, layer_id: int | None = None, layer_idx: int | None = None, old_attn: torch.nn.modules.module.Module | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).
    
    Hardened for production use across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA/Mistral.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `dense: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `k_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `k_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `key: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `o_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `out_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `q_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `q_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `query: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `v_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `v_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `value: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: torch.nn.modules.module.Module) ‑> None`
    :   Extracts and copies pretrained weights across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA.
        Strictly validates that copied weights are non-empty and verifies reconstruction norms.

`ChelaSlayAttention(config: Any | None = None, rank: int = 64, feature_dim: int = 128, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, gate_init_bias: float = 5.0, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.
    
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

    `gate`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `random_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `storage_bytes(self) ‑> dict[str, int]`
    :

`ClusterAttention(config: Any | None = None, rank: int | None = None, num_clusters: int = 16, n_iters: int = 3, window_size: int = 32, lora_rank: int = 16, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

`CompressedAttention(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer2

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `out_weight: torch.Tensor`
    :   Reconstruct the effective base Out weight from SVD factors.

    `qkv_weight: torch.Tensor`
    :   Reconstruct the effective base QKV weight from SVD factors.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `reset_lora_parameters(self) ‑> None`
    :

`CompressedAttentionLayer(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer2

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `out_weight: torch.Tensor`
    :   Reconstruct the effective base Out weight from SVD factors.

    `qkv_weight: torch.Tensor`
    :   Reconstruct the effective base QKV weight from SVD factors.

    ### Methods

    `reset_lora_parameters(self) ‑> None`
    :

`CompressedAttentionLayer2(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

`CompressedDistilBertAttention(model_config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, dropout: float = 0.1, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `out_weight: torch.Tensor`
    :   Reconstruct the effective base Out weight from SVD factors.

    `qkv_weight: torch.Tensor`
    :   Reconstruct the effective base QKV weight from SVD factors.

    ### Methods

    `reset_lora_parameters(self) ‑> None`
    :

`ContextFusionGate(hidden_size, num_heads, head_dim, init_bias=1.8)`
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

    `forward(self, hidden_states, local_out, global_out) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CosformerAttention(config: Any | None = None, rank: int | None = None, max_seq_len: int = 512, act_fun: str = 'relu', eps: float = 1e-06, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Cosformer Attention with Linear Complexity and Exact Multi-Architecture Support.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights with strict Frobenius norm verification.

`DecayGatedLinearAttention(*args, **kwargs)`
:   Decay Gated SVD Linear Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

`DecayLinearAttention(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Decay Linear Attention Module with Data-Dependent Exponential Decay Gating.
    
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

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights with strict Frobenius norm verification.

`DualStageStripeAttention(cfg: neuralops.attentions.nexus.NEXUSConfig, head_dim: int, num_heads: int = 1, is_decoder: bool = False, is_nested_encoder: bool = False, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, token_valid: torch.Tensor | None = None, seer_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`DynamicLowRankProjector(rank: int = 32, in_dim: int = 768, out_dim: int = 768, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters.

    ### Methods

    `copy_lora_from(self, sub: Any) ‑> None`
    :   Preserve source LoRA adapters.

    `forward(self, x: torch.Tensor, gate: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `set_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :

`EATBlock(config: Any | None = None, rank: int | None = 64, d_model: int | Any = 768, n_heads: int = 12, prune_layer: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn, model_config=None, **kwargs)`
    :

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

`EarlyStoppingCallback(early_stopping_patience: int = 1, early_stopping_threshold: float | None = 0.0)`
:   A [`TrainerCallback`] that handles early stopping.
    
    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `
    
    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback
    * transformers.trainer_callback.ExportableState

    ### Methods

    `check_metric_value(self, args, state, control, metric_value)`
    :

    `on_evaluate(self, args, state, control, metrics, **kwargs)`
    :   Event called after an evaluation phase.

    `on_train_begin(self, args, state, control, **kwargs)`
    :   Event called at the beginning of training.

    `state(self) ‑> dict`
    :

`ExpertChoiceMoE(d_model: int, n_experts: int = 2, capacity_factor: float = 1.0, noise_std: float = 0.05)`
:   Expert Choice Routing MoE layer.
    Each expert selects top-k tokens with the highest affinity scores.
    Equipped with zero-initialized final layers for step-0 representation preservation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward execution using Expert Choice routing.

`ExpertChoiceMoEAttention(config: Any | None = None, rank: int | None = 64, n_experts: int = 2, capacity_factor: float = 1.0, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, moe_init_gate: float = 0.0, **kwargs: Any)`
:   Expert Choice MoE Multi-Head Attention Module.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_proj: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_proj: LoRALinear`
    :

    `q_proj: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_proj: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: nn.Module) ‑> None`
    :   Copy projection weights from pretrained attention module with Frobenius norm verification.

`FUSIONATTN(config: Any | None = None, hidden_size: int = 768, num_heads: int = 12, block_size: int = 64, top_k_blocks: int = 4, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Hybrid sparse attention with block gating.
    
    Two execution paths:
      * Short sequence (N <= 128) or cross-attention  → dense attention
      * Long sequence                                  → block-gated sparse
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`FactorisedLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, energy: float = 0.95, lora_rank: int = 16, lora_alpha: float = 32.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   SVD-Factorized Linear Transformation for NOVA Architecture.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, energy: float | None = None) ‑> None`
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

    `reset_parameters(self) ‑> None`
    :

`FactorisedLinear_CUR(in_features: int, out_features: int, cur_rank: int = 16, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   Linear projection compressed via leverage-score CUR matrix decomposition with LoRA support.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `copy_cur(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
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

    `reset_parameters(self) ‑> None`
    :

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

`FactorisedMatrixState(config: Any | None = None, rank: int | None = 32, d_model: int | Any | None = None, num_heads: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, old_attn: torch.nn.modules.module.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `W_q: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `W_v: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `c_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `dense: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `k_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `key: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `o_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `out_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `out_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `q_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `query: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `v_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `value: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights with strict Frobenius norm verification.

`FactorizedLinear(in_features: int, out_features: int, rank: int = 128, lora_rank: int = 32, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

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

    `reset_parameters(self) ‑> None`
    :

`FactorizedLoRALinear(in_features: int, out_features: int, rank: int = 64, lora_rank: int = 16, lora_alpha: float = 32.0, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   SVD-factorized linear + LoRA:
        y = up(down(x)) + (lora_B @ lora_A)(x) * scaling
    Base (down, up) frozen; LoRA trainable.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

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

    `storage_bytes(self) ‑> int`
    :

`FactorizedLoRAUp(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0)`
:   Up-projection with frozen base weight and trainable LoRA adapter.
    
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

`FlashAttention3(config: Any | None = None, rank: int | None = None, use_fp8: bool = False, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   FlashAttention-3 Multi-Head Exact Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Copy projection weights from pretrained attention module with Frobenius norm verification.

`FlashAttention3FP8(config: Any | None = None, rank: int | None = None, d_model: int | Any = 768, n_heads: int = 12, use_fp8: bool = False, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.
    
    Accelerates attention using FP8 (e4m3fn) precision with FlashAttention-3 kernels on
    Hopper (SM 9.0+) GPUs. Automatically falls back to native PyTorch SDPA on pre-Hopper
    hardware or CPU.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Copy projection weights from pretrained attention module with Frobenius norm verification.

`FullAttention(config: Any, rank: int | None = None, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `k_proj: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `q_proj: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `v_proj: LoRALinear`
    :

`FusedSVDKernelAttention(config: Any | None = None, rank: int | None = None, old_attn: nn.Module | None = None, use_fused_kernel: bool = True, use_rms_norm: bool = False, dropout: float = 0.1, **kwargs: Any)`
:   Fused SVD-Kernel Linear Attention Module.
    Production-hardened across DistilBERT, RoBERTa/BERT nested, GPT-2 Conv1D, and Qwen/LLaMA.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: FusedSVDLinear`
    :

    `query: FusedSVDLinear`
    :

    `value: FusedSVDLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Multi-architecture weight copying supporting GPT-2 Conv1D, RoBERTa nested, and flat projections.

`FusedSVDLinear(in_features: int, out_features: int, rank: int = 32, bias: bool = True, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   RESEARCH SOTA: Fused SVD-Linear Module.
    Combines low-rank projection $Z = x V^T$, in-SRAM RMSNorm, and up-projection $Y = Z U^T + b$
    into a single kernel invocation, eliminating HBM roundtrips.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :   Decompose dense weight matrix into symmetric SVD factors.

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :   # RESEARCH FIX: Computes Stiefel manifold penalty to prevent singular subspace collapse.

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :   Set precomputed U and V low-rank factors.

`GalerkinAttention(config: Any | None = None, rank: int | None = None, dropout: float = 0.1, basis_type: str = 'ortho_dct', pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

`GatedDeltaNet(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, old_attn: nn.Module | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`GatedDeltaNetBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, use_full_attention: bool = False, comp_config: Any | None = None, **kwargs: Any)`
:   Gated DeltaNet Transformer Block with Pretrained Weight Transfer.
    
    Mathematical Formulation:
        $$\text{Output} = \begin{cases}
            \text{LayerNorm}(X + \text{MHA}(X, X, X)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(X + \text{GatedDeltaNet}(X)), & \text{otherwise}
        \end{cases}$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: Any`
    :

    `k_proj: Any`
    :

    `key: Any`
    :

    `out_lin: Any`
    :

    `out_proj: Any`
    :

    `q_lin: Any`
    :

    `q_proj: Any`
    :

    `query: Any`
    :

    `v_lin: Any`
    :

    `v_proj: Any`
    :

    `value: Any`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained weights to either MHA or GatedDeltaNet.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass for GatedDeltaNetBlock.

`GatedLinearAttention(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, head_dim: int | None = None, num_kv_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, chunk_size: int = 128, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`GatedRecurrentMemory(d_model: int)`
:   A light gated recurrent memory:
        g_t = sigmoid(W_g x_t + b_g)
        h_t = (1 - g_t) * h_{t-1} + g_t * tanh(W_x x_t + b_x)
        y_t = h_t
    
    Not a full DeltaNet but has the required gate structure and behaves as
    a proper recurrent memory (unlike the previous plain Linear layer).
    
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

`GatingModule(hidden_size, num_heads, head_dim, use_query, use_input)`
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

    `forward(self, hidden_states, Q) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`GoldFinchAHNBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, window_size: int = 64, is_causal: bool = False, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, gate_init_bias: float = 1.8, comp_config: Any | None = None, **kwargs: Any)`
:   GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.
    
    Semantics of `window_size`: full window width (in tokens). Effective
    radius = `window_size // 2`, so `window_size = 64` → ±32 tokens.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn, model_config=None, **kwargs)`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj`
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

`GraniteHybridBlock(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, num_experts: int = 4, top_k: int = 2, ratio: int = 2, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any)`
:   Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

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

`GroupedQueryAttention(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    ### Methods

    `get_ortho_loss(self)`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :

`HALOReGLAAttention(config: Any | None = None, rank: int | None = None, hybrid_ratio: int = 4, pretrained_attn: nn.Module | None = None, gate_init_bias: float = 1.8, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.halo_regla.HALOReGLAAttention`
    :

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

`HGRN2Layer(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, res_clip: float = 30.0, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, freeze_base: bool = True, **kwargs: Any)`
:   Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer.
    Supports causal mode for decoders, bidirectional mode for encoders, LoRA adapters,
    and pretrained weight copying from DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    NOTE: This is NOT a compression module. It replaces attention with an RNN.
    Parameter count is the same. Only use for speed improvements on
    long sequences (when parallel scan is implemented).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: nn.Module`
    :

    `query: nn.Module`
    :

    `value: nn.Module`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights across all 4 architectures.

`HGRN2OuterBlock(config: Any | None = None, rank: int | None = None, dim: int | Any | None = None, num_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, chunk_size: int = 64, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`HLAAttention(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Hadamard Linear Attention (HLA) Bidirectional / Causal Module.
    
    Maintains multi-head associative causal/anti-causal matrix states $S_t \in \mathbb{R}^{B \times H \times d_k \times d_k}$.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor`
    :   Bidirectional / Causal linear attention forward pass.

`HammingAttention(config: Any | None = None, rank: int | None = None, distill_temperature: float = 2.0, sparsity_ratio: float = 0.0, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

`HeadGate(hidden_dim: int, n_heads: int)`
:   Learned per-example head importance scalars with masked pooling.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`HeadRoPE(head_dim: int, max_seq_len: int = 4096, base: float = 10000.0)`
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

    `forward(self, x: torch.Tensor, past_len: int = 0) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`HybridAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SmartWeightedEnsembleAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`WeightedEnsembleAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`HymbaTransXBlock(config: Any | None = None, rank: int | None = 64, num_heads: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   HymbaTransX: Hybrid Differential Transformer & State Space Model Block.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `dense`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_proj`
    :

    `query`
    :

    `v_proj`
    :

    `value`
    :

`IdentityOutProj()`
:   A placeholder identity operator that is argument-insensitive.
    
    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Examples::
    
        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.linear.Identity
    * torch.nn.modules.module.Module

    ### Methods

    `freeze_base_weights(self) ‑> None`
    :

`InfiniAttention(config: Any | None = None, rank: int | None = None, local_window: int = 128, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Infini-Attention: sliding-window local attention + compressive memory.
    
    For decoders, the output is a per-head convex combination of:
      - A_local: sliding-window softmax attention
      - A_mem:   read from a transient (per-forward) compressive memory state
    For encoders, it degenerates to full softmax attention (memory is skipped).
    
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

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`InfoMambaConceptBottleneck(dim: int, bottleneck_dim: int = 256)`
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

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`Jamba2Block(config: Any | None = None, d_model: int | None = None, n_heads: int | None = None, n_experts: int = 4, top_k: int = 2, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Jamba-2 Hybrid: SVD-Mamba + Multi-Head Attention + Expert-Choice MoE.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `aux_loss: float`
    :   Mean aux loss over the accumulated forward calls.

    ### Methods

    `freeze_base_weights(self) ‑> None`
    :

    `reset_aux_loss(self) ‑> None`
    :   Call at the end of an epoch / accumulation window.

    `storage_bytes(self) ‑> dict[str, int]`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`JambaInfoBlock(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_decoder: bool | None = None, per_head_alpha: bool = True, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`KDAAttention(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `freeze_base_weights(self) ‑> None`
    :

`KernelFeatureConfig(head_dim, num_features, method='orthogonal', clamp_min=-20.0, clamp_max=8.0, use_float64_exp=False, eps=1e-05, scaling=None, seed=None)`
:   

`KernelFeatureMap(head_dim: int, num_features: int = 128, seed: int = 1234)`
:   FAVOR+ style random Fourier features with deterministic seed.
    
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

`LEKAV1Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Positive kernel feature embedding using ELU + 1.
        Guarantees strict positivity so denominator never collapses or becomes negative.
        Returns tensor of shape (B, H, N, F).

`LEKAV3Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, past_key_value: Any | None = None, layer_idx: int = 0, output_attentions: bool | None = False, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`LEKAv3Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`LEKAV2Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, past_key_value: Any | None = None, layer_idx: int = 0, output_attentions: bool | None = False, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`LEKAv2Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `block_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :

    `select_landmarks_chunked(self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor | None = None)`
    :

    `transpose_for_scores(self, x: torch.Tensor) ‑> torch.Tensor`
    :

`LSHAttention(config: Any, rank: int | None = None, rounds: int = 4, num_buckets: int = 16, lora_rank: int = 16, lora_alpha: float = 32.0, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

`LinformerAttention(config: Any | None = None, rank: int | None = None, k: int | None = None, max_seq_len: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`LoLA(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, lora_rank: int = 16, dropout: float = 0.1, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `reset_cache(self) ‑> None`
    :

`LoLAAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, lora_rank: int = 16, dropout: float = 0.1, comp_config: Any | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Methods

    `reset_cache(self) ‑> None`
    :

`LoRAAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, lora_rank: int = 16, alpha: float = 32.0, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   LoRA attention wrapper.
    
    NOTE: This module is deliberately a PEFT swap, NOT a compressor.
    `compression_rate` remains ~1.0; `num_compressed_layers` remains 0.
    Its role is to provide a trainable LoRA residual on top of frozen
    pretrained attention weights.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters.
    
    Base path:
        y_base = x @ W^T + b  (frozen when freeze_base=True)
    LoRA path:
        y_lora = (x @ lora_A^T) @ lora_B^T * scaling  (trainable)
    Total output:
        y = y_base + y_lora
    
    Uses standard `lora_A` and `lora_B` parameter naming recognized by framework recovery optimizers.
    
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
    :   Freeze base projection parameters so only LoRA parameters adapt.

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :   Unfreeze base projection parameters.

`LowRankCache(max_len: int = 128, low_rank_k: int = 16)`
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

    `get_cached(self) ‑> tuple[torch.Tensor | None, torch.Tensor | None]`
    :

    `reset(self) ‑> None`
    :

    `update(self, K: torch.Tensor, V: torch.Tensor) ‑> None`
    :

`LowRankLinear(in_features: int, out_features: int, rank: int, lora_rank: int = 16, bias: bool = True)`
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

    `reset_parameters(self) ‑> None`
    :

`LowRankSparseLinear(down: nn.Linear, up: nn.Linear, sparse_weight: torch.Tensor, bias: torch.Tensor | None = None)`
:   y = up(down(x)) + F.linear(x, S) + b
    
    S is a trainable sparse residual. Its L1 gradient encourages zero
    entries; periodic `prune_sparse` enforces hard threshold.
    
    SP-FIX-02: bias stored as a proper nn.Parameter.
    
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

    `prune_sparse(self, sparsity: float) ‑> None`
    :   Hard threshold: keep only top (1 - sparsity) fraction of entries.

    `sparse_l1_loss(self) ‑> torch.Tensor`
    :

`MLAAttention(config: Any | None = None, rank: int | None = 128, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, max_error: float = 0.35, fallback_to_original: bool = True, **kwargs: Any)`
:   Multi-Head Latent Attention (MLA) — DeepSeek-V2 style.
    
    Idea: compress K and V into a low-rank latent `c_kv` of size `rank`,
    then up-project back to per-head K and V.
    
        c_kv = kv_down(x)             # (B, N, rank)
        k    = k_up(c_kv)              # (B, N, num_kv_heads * head_dim)
        v    = v_up(c_kv)              # (B, N, num_kv_heads * head_dim)
    
    Query is projected with a LoRA-augmented linear. K/V down/up projections
    are initialized via joint SVD of [W_k; W_v] and frozen.
    
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

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `is_compatible(self) ‑> bool`
    :   Return True if reconstruction error is within tolerance.

    `storage_bytes(self) ‑> dict[str, int]`
    :

`Mamba(d_model: int = 768, **kwargs: Any)`
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

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`Mamba2(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs: Any)`
:   Simplified Mamba-2 State Space Model (SSM) block.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`MambaBlock(config: Any | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, is_decoder_override: bool | None = None, freeze_base: bool = False, **kwargs: Any)`
:   Faithful S6 / Mamba block, production-hardened.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: nn.Module) ‑> None`
    :   Distill attention weights (W_q, W_k, W_v, W_o) into Mamba's S6
        parameters via low-rank mapping:
            W_v  → in_proj (SSM input path)
            W_k  → x_proj (B branch, via SVD)
            W_q  → x_proj (C branch, via SVD)
            W_o  → out_proj
            dt   → kept at initialization (learnable, input-dependent via x_proj)

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, *args: Any, past_state: Any | None = None, past_key_value: Any | None = None, output_attentions: bool = False, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :

    `freeze_base_weights(self) ‑> None`
    :   MB-04: match the framework's LoRA-freeze convention.

    `unfreeze_base_weights(self) ‑> None`
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

`MinimalMoEFallback(d_model: int = 768, n_experts: int = 4)`
:   Fallback MoE layer when REAPMoE cannot be instantiated.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`MinimalNOVAAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalNOVA: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
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

    `key: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalSVDAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalAuroraAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalPRISMAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`MinimalSVDAttentionV2(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.naive_svd.NaiveSVDAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_svd.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`FASA(config: Any | None = None, rank: int | None = 32, fasa_keep_ratio: float = 0.5, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.
    
    For non-causal self-attention and encoder-style models, we prune the K/V
    set down to the top `fasa_keep_ratio` tokens by variance importance,
    always preserving the [CLS] token at index 0.
    
    For causal decoders, cross-attention, and incremental decoding (when
    `past_key_value` is present), we bypass pruning entirely and run exact
    dense attention via SDPA / manual softmax.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Must return (output, attention_weights_or_None).

    `storage_bytes(self) ‑> dict[str, int]`
    :

`MixAttentionMoSKA(config: Any | None = None, rank: int | None = 32, fasa_keep_ratio: float = 0.5, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.
    
    For non-causal self-attention and encoder-style models, we prune the K/V
    set down to the top `fasa_keep_ratio` tokens by variance importance,
    always preserving the [CLS] token at index 0.
    
    For causal decoders, cross-attention, and incremental decoding (when
    `past_key_value` is present), we bypass pruning entirely and run exact
    dense attention via SDPA / manual softmax.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `storage_bytes(self) ‑> dict[str, int]`
    :

`MixtureOfExperts(dim: int, ff_dim: int, num_experts: int = 4, top_k: int = 2, **kwargs: Any)`
:   Top-$k$ Token-Choice Mixture-of-Experts (MoE) Layer.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`MoBAAttention(config: Any, block_size: int = 64, top_k: int = 4, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MoBA: Mixture of Block Attention for Long-Context Modeling.
    
    For each query block, we compute block-level centroid scores, select the
    top-k most relevant key blocks, and run dense attention within the
    selected blocks (plus the causal-visible diagonal block).
    
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

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`MoEDeltaKVPagedAttention(config: Any | None = None, page_size: int = 16, k_experts: int = 4, svd_rank: int = 32, q_bits: int = 4, max_delta_error: float = 0.35, lora_rank: int = 16, lora_alpha: float = 32.0, enable_moe: bool = True, enable_deltakv: bool = True, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   MoE-DeltaKV-Paged Hybrid Attention.
    
    Two execution paths:
      * Full dense attention when `num_pages <= 1` (page_size >= seq_len)
        OR when `enable_moe=False`.
      * Paged MoE attention otherwise.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `storage_bytes(self) ‑> dict[str, int]`
    :

`ModelAdapter(model_or_config: Any, family: str | None = None)`
:   Read-only architecture contract used by module wrappers.
    
    Fix MF-05: `family` is normalized to a canonical internal name (e.g. "gpt2"
    becomes "gpt"), while the original name is preserved in `raw_family` for
    logging. Downstream code that inspects `adapter.family` will always see a
    canonical value; code that needs the raw value should use `raw_family`.

    ### Methods

    `as_dict(self) ‑> dict[str, typing.Any]`
    :

    `projection(self, module: Any, kind: str) ‑> Any`
    :

`NASCAttention(config: Any, rank: int | None = 16, window_size: int = 64, block_size: int = 32, top_k: int = 2, rff_features: int = 64, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   NASC: NEXUS-AEGIS Spectral Cascade Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
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

    `key: neuralops.attentions.nasc_attention.LoRALinear`
    :

    `query: neuralops.attentions.nasc_attention.LoRALinear`
    :

    `value: neuralops.attentions.nasc_attention.LoRALinear`
    :

`NEXUSAttention(config: Any | None = None, model_config: Any | None = None, nexus_cfg: neuralops.attentions.nexus.NEXUSConfig | None = None, layer_idx: int = 0, total_layers: int = 6, pretrained_attn: torch.nn.modules.module.Module | None = None, bert_mode: bool = False, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :

`NEXUSConfig(block_size: int = 16, top_k_blocks: int = 4, xattn_enabled: bool = True, xattn_stride: int = 2, seer_enabled: bool = True, seer_gate_rank: int = 16, seer_top_k_blocks: int = 4, seer_temperature: float = 1.0, dual_stage_enabled: bool = True, intra_block_full: bool = True, stripe_enabled: bool = True, stripe_width: int = 4, rectify_interval: int = 4, rff_features: int = 128, softpick_enabled: bool = True, softpick_temperature: float = 1.0, sparse_min_seq_len: int = 8, enable_spectraformer: bool = False, dropout: float = 0.1)`
:   NEXUSConfig(block_size: int = 16, top_k_blocks: int = 4, xattn_enabled: bool = True, xattn_stride: int = 2, seer_enabled: bool = True, seer_gate_rank: int = 16, seer_top_k_blocks: int = 4, seer_temperature: float = 1.0, dual_stage_enabled: bool = True, intra_block_full: bool = True, stripe_enabled: bool = True, stripe_width: int = 4, rectify_interval: int = 4, rff_features: int = 128, softpick_enabled: bool = True, softpick_temperature: float = 1.0, sparse_min_seq_len: int = 8, enable_spectraformer: bool = False, dropout: float = 0.1)

    ### Instance variables

    `block_size: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `dual_stage_enabled: bool`
    :   The type of the None singleton.

    `enable_spectraformer: bool`
    :   The type of the None singleton.

    `intra_block_full: bool`
    :   The type of the None singleton.

    `rectify_interval: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `seer_enabled: bool`
    :   The type of the None singleton.

    `seer_gate_rank: int`
    :   The type of the None singleton.

    `seer_temperature: float`
    :   The type of the None singleton.

    `seer_top_k_blocks: int`
    :   The type of the None singleton.

    `softpick_enabled: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `sparse_min_seq_len: int`
    :   The type of the None singleton.

    `stripe_enabled: bool`
    :   The type of the None singleton.

    `stripe_width: int`
    :   The type of the None singleton.

    `top_k_blocks: int`
    :   The type of the None singleton.

    `xattn_enabled: bool`
    :   The type of the None singleton.

    `xattn_stride: int`
    :   The type of the None singleton.

`NaiveSVDAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 1.0, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, rank: int | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Naive truncated SVD baseline attention module.
    Inherits production-hardened multi-architecture routing from MinimalSVDAttention
    with uncalibrated plain truncated SVD initialization and no recovery adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.minimal_svd.MinimalSVDAttention
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

`Namespace(**kwargs)`
:   Simple object for storing attributes.
    
    Implements equality by attribute names and values, and provides a simple
    string representation.

    ### Ancestors (in MRO)

    * argparse._AttributeHolder

`NativeHybridAttention(config: Any | None = None, rank: int = 64, window_size: int = 64, rff_features: int = 128, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Native Hybrid Attention: local sliding-window + global RFF, fused by a
    token-level learned gate.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`NSAAttention(config: Any | None = None, rank: int | None = None, window_size: int = 32, top_k: int = 16, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, output_attentions: bool | None = False, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`NativeSparseAttention(config: Any | None = None, rank: int | None = None, window_size: int = 32, top_k: int = 16, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`NystromAttention(config: Any | None = None, rank: int | None = None, num_landmarks: int | None = None, pinv_iterations: int = 3, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
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

`OptimizedHybridAttention(config=None, rank=None, window_size=64, compression_rank=32, use_gate=True, gate_bias=2.0, pretrained_attn=None, old_attn=None, comp_config=None, is_decoder=None, **kwargs)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

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

`PHANTOMAttention(config: Any | None = None, rank: int | None = 16, phantom_cfg: PHANTOMConfig | None = None, pretrained_attn: Any | None = None, bert_mode: bool = False, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`PHANTOMConfig(cur_rank: int = 16, lsh_rounds: int = 4, lsh_bucket_size: int = 4, softpick_enabled: bool = True, softpick_temperature: float = 1.0, dropout: float = 0.1)`
:   PHANTOMConfig(cur_rank: 'int' = 16, lsh_rounds: 'int' = 4, lsh_bucket_size: 'int' = 4, softpick_enabled: 'bool' = True, softpick_temperature: 'float' = 1.0, dropout: 'float' = 0.1)

    ### Instance variables

    `cur_rank: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `lsh_bucket_size: int`
    :   The type of the None singleton.

    `lsh_rounds: int`
    :   The type of the None singleton.

    `softpick_enabled: bool`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

`PerformerAttention(config: Any | None = None, rank: int | None = None, eps: float = 1e-06, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_nested_encoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> bool`
    :   Copy weights from source attention module.
        
        FIX 13: returns bool (was None).
        FIX 4: sets is_nested_encoder robustly.

`PrimalAttention(config: Any | None = None, rank: int = 64, pinv_iters: int = 2, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`ProgressiveTokenPruning(config: Any | None = None, d_model: int | None = None, keep_ratio: float = 0.8, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, shared_scorer: SharedTokenScorer | None = None, **kwargs: Any)`
:   Progressive Token Pruning attention.
    
    Forward:
      1. Score each token (score_network or shared scorer).
      2. Force-keep the first token ([CLS]).
      3. Keep the top-k tokens (k = ceil(L · keep_ratio)).
      4. Compute attention over the reduced set of keys/values.
      5. Return output with the original (B, L, D) shape.
    
    KV-cache: supported for decoder inference.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `gate`
    :

    `random_proj`
    :

`ProjectionFactorizedAttention(config: Any | None = None, old_attn: nn.Module | None = None, rank: int | None = None, mode: str = 'exact', freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, use_decay_gating: bool = False, decay_activation: str = 'softplus', decay_channel_wise: bool = False, use_fused_kernel: bool = False, use_rms_norm: bool = False, chunk_size: int = 64, **kwargs: Any)`
:   Projection Factorized SVD Linear Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

`QCQAAttention(config: Any | None = None, num_kv_heads: int | None = None, router_temp: float = 1.0, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`RALACompressorAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`RALALinearAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`REAPMoE(d_model: int = 768, n_experts: int = 2, top_k: int = 1, noise_std: float = 0.05, use_svd: bool = True, config: Any | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: Any)`
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

    ### Instance variables

    `aux_loss: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`REAPMoEAttention(config: Any | None = None, rank: int | None = 64, n_experts: int = 2, top_k: int = 1, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, pre_ln: bool | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `aux_loss: torch.Tensor`
    :

`RPCAttention(config: Any = None, rank: int = 64, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    `k_lin`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `v_lin`
    :

    ### Methods

    `storage_bytes(self) ‑> dict[str, int]`
    :   Report per-component storage.

`ReGLAAttention(config: Any | None = None, rank: int = 64, kernel_size: int = 3, pretrained_attn: torch.nn.modules.module.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, is_decoder: bool | None = None, chunk_size: int = 64, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`Replace(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, output_attentions: bool = False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`ReplaceAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`ResSVDAttention(config: Any, base_rank: int | None = None, residual_rank: int | None = None, rank: int | None = 16, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True, dropout: float | None = None, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `k_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

`ResSVDLinear(in_features: int, out_features: int, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Spectral Residual LoRA (Res-SVD) Linear Layer.
    
    Executes forward pass as:
    y = x @ W_base^T + scaling * (x @ A^T) @ B^T + bias
    where W_base is either factorized as (base_V, base_U) with requires_grad=False,
    and (lora_A, lora_B) are trainable adapter matrices initialized from the residual tail.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear, base_rank: int = 32, residual_rank: int = 8, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True) ‑> neuralops.compression_engine.res_svd.ResSVDLinear`
    :   Construct ResSVDLinear by decomposing a pretrained nn.Linear weight:
        1. Base factors U_r, V_r from the dominant top-r singular components.
        2. Residual adapter factors A, B directly initialized from the residual tail (r+1 to r+k).
        
        Args:
            linear: Source dense linear layer.
            base_rank: Rank r for the frozen base matrix.
            residual_rank: Rank k for the trainable residual adapter.
            scaling: Multiplier alpha / k for adapter output.
            factorize_base: Whether to store base weights as factorized U, V matrices.
            freeze_base: Whether to freeze base factors (requires_grad=False).
        
        Returns:
            ResSVDLinear: Spectral residual LoRA module.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward execution:
        y = Base(x) + scaling * (x @ A^T) @ B^T + bias

    `get_residual_error_norm(self, original_weight: torch.Tensor | None = None) ‑> float`
    :   Calculate relative Frobenius norm reconstruction error:
        || W_orig - (W_base + scaling * B @ A) ||_F / || W_orig ||_F

    `merge_weights(self) ‑> torch.nn.modules.linear.Linear`
    :   Merge frozen base weights and trainable residual adapter weights into a dense nn.Linear.
        
        Returns:
            nn.Linear: Standard dense linear layer with merged weights.

    `reset_parameters(self) ‑> None`
    :   Kaiming uniform parameter initialization for base and zero for adapters.

`RoutingAttention(config: Any = None, rank: int | None = None, num_clusters: int = 16, n_iters: int = 3, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

`SAMBAM3Block(config: Any | None = None, d_model: int | None = None, n_heads: int | None = None, window: int = 128, rank: int | None = None, pretrained_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   SAMBA-M3: Local Sliding-Window Attention + Spectral Mamba.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_proj`
    :

    `key`
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

    `copy_weights(self, pretrained_attn: nn.Module) ‑> None`
    :

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `get_ortho_loss(self) ‑> float`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`SOAAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, k_rank: int = 64, dropout: float = 0.1, pretrained_attn: Any | None = None, old_attn: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
    
    Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention`
    :

    ### Instance variables

    `k_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `key: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `q_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `query: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `v_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `value: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight extraction and SVD factorization across all architectures.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor`
    :   Must return (output, attention_weights_or_None).

`SOAFusedAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, k_rank: int = 64, dropout: float = 0.1, pretrained_attn: Any | None = None, old_attn: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
    
    Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: torch.nn.modules.module.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.soa_attn.SOAAttention`
    :

    ### Instance variables

    `k_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `key: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `q_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `query: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `v_proj: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    `value: neuralops.attentions.soa_attn.FactorizedLinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight extraction and SVD factorization across all architectures.

`SVDCompressedAttention(config: Any | None = None, old_attn: Any | None = None, dropout: float = 0.1, layer_idx: int = 0, ranks: dict[str, int] | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, add_sparse: bool = False, sparsity_level: float = 0.7, **kwargs: Any)`
:   Multi-head attention with truncated SVD projections, trainable LoRA adapters,
    GQA support, RoPE support, and optional sparse residuals.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

`SVDLLMV2Attention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 192, energy_threshold: float = 0.999, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float = 0.1, **kwargs: Any)`
:   SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers
    and trainable LoRA recovery adapters.
    
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

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

`SVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, use_fused_kernel: bool = False, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Low-rank factorized linear layer:
        y = (x V^T) U^T + b + (alpha / r_lora) (x A^T) B^T
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
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

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SVDLinearAttention(config: Any | None = None, old_attn: nn.Module | None = None, rank: int | None = None, mode: str = 'exact', freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, use_decay_gating: bool = False, decay_activation: str = 'softplus', decay_channel_wise: bool = False, use_fused_kernel: bool = False, use_rms_norm: bool = False, chunk_size: int = 64, **kwargs: Any)`
:   SVD Factorized Linear Attention with optional exponential decay gating.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.projection_factorized_attn.ProjectionFactorizedAttention
    * neuralops.attentions.svd_linear_decay.DecayGatedLinearAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: nn.Module) ‑> None`
    :   SVDLA-FIX-01: never raise. On missing projections, log a warning
        and fall back to the randomly-initialized state.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Must return (output, attention_weights_or_None).

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `step_recurrent(self, x_t: torch.Tensor, past_state: tuple[torch.Tensor, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]`
    :

`LowRankSVDSequenceAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 64, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float | None = 0.1, **kwargs: Any)`
:   Low-Rank SVD-Compressed Sequence Attention Mechanism.
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

    `k_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `k_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :   Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.

`SVDAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 64, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float | None = 0.1, **kwargs: Any)`
:   Low-Rank SVD-Compressed Sequence Attention Mechanism.
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

    `k_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `k_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `get_ortho_loss(self) ‑> torch.Tensor`
    :   Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.

`SVDLinearProjection(in_features: int, out_features: int, rank: int = 64, bias: bool = True)`
:   Genuine SVD Low-Rank Linear Layer: W \approx U @ V.
    
    Factorizes in_features -> out_features through bottleneck rank r:
      V \in R^{r x in_features}  (Down-projection)
      U \in R^{out_features x r} (Up-projection)
      y = (x @ V^T) @ U^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `triton_kernel`
    :   The type of the None singleton.

    ### Static methods

    `from_pretrained_linear(linear: nn.Linear, rank: int = 64) ‑> neuralops.attentions.triton_svd_linear_attention.SVDLinearProjection`
    :   Initializes low-rank factors directly from pretrained weights via Truncated SVD:
        W = U_full @ diag(S) @ V_full^T
        U = U_full[:, :r] @ sqrt(S[:r])
        V = sqrt(S[:r]) @ V_full^T[:r, :]

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
    :

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

`SVDMambaSub(d_model: int = 768, config: Any | None = None, rank: int | None = None, is_decoder: bool = False, **kwargs: Any)`
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

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, past_state: Any | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, Any | None]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

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

`SambaMoEBlock(config: Any | None = None, d_model: int | None = None, n_heads: int | None = None, n_experts: int = 8, window: int = 128, rank: int | None = 64, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `is_nested_encoder: bool`
    :

    `key: torch.nn.modules.module.Module`
    :

    `query: torch.nn.modules.module.Module`
    :

    `value: torch.nn.modules.module.Module`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: torch.nn.modules.module.Module) ‑> None`
    :   Propagate weight transfer to internal SAMBA-M3 and SSM components.

`SeerGating(head_dim: int, num_heads: int, gate_rank: int, top_k: int, temperature: float = 1.0, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, n_blocks: int, training: bool = True) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SelectiveMamba(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, dim: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 1, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Selective Mamba SSM block.
    
    Implements a Mamba-style selective state space model with
    input-dependent B, C, and dt parameters:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t + D · x_t
    where Ā_t = exp(dt_t · A).
    
    The "selectivity" comes from x_proj producing B_t and C_t
    from the input.
    
    NOTE: The previous name "UPI-Stuffed" has been removed
    because none of the claimed features (Uncertainty Penalty,
    Interpolation, Stuffing) were actually implemented.
    
    Registered as "upi_mamba" (legacy alias) and
    "selective_mamba" (canonical).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `step_recurrent(self, x_t: torch.Tensor, state: dict[str, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, dict[str, torch.Tensor]]`
    :   Single-step streaming inference.
        
        x_t: (B, 1, D)
        state: dict with keys 'h', 'conv_buf' (optional)
        Returns: (y_t, new_state)

`UPIMambaBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, dim: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 1, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Selective Mamba SSM block.
    
    Implements a Mamba-style selective state space model with
    input-dependent B, C, and dt parameters:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t + D · x_t
    where Ā_t = exp(dt_t · A).
    
    The "selectivity" comes from x_proj producing B_t and C_t
    from the input.
    
    NOTE: The previous name "UPI-Stuffed" has been removed
    because none of the claimed features (Uncertainty Penalty,
    Interpolation, Stuffing) were actually implemented.
    
    Registered as "upi_mamba" (legacy alias) and
    "selective_mamba" (canonical).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA.

    `step_recurrent(self, x_t: torch.Tensor, state: dict[str, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, dict[str, torch.Tensor]]`
    :   Single-step streaming inference.
        
        x_t: (B, 1, D)
        state: dict with keys 'h', 'conv_buf' (optional)
        Returns: (y_t, new_state)

`SelectiveSSM(config: Any | None = None, rank: int | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, lora_rank: int = 0, weights_init: str = 'mamba', **kwargs: Any)`
:   Faithful Mamba-style selective SSM block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :

`SimpleSSM(config: Any | None = None, rank: int | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, lora_rank: int = 0, weights_init: str = 'mamba', **kwargs: Any)`
:   Faithful Mamba-style selective SSM block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

`SharedTokenScorer(dim: int, hidden: int = 64)`
:   Optional token scorer that is shared across all PTP layers to reduce
    parameter overhead. When `share_scorer=True` is passed to all PTP
    instances, they reference the same scorer module.
    
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

`SimpleNamespace(*args, **kwargs)`
:   A simple attribute-based namespace.

`SpanExpandedAttention(config: Any, rank: int | None = None, expansion_span: int = 1024, num_sink_tokens: int = 4, use_lora: bool = True, lora_rank: int = 8, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`SparsePruningCallback(model: Any, prune_freq: int = 200, final_sparsity: float = 0.7)`
:   Periodically hard-thresholds the sparse residual of every
    SpectronAttention module in the model.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_step_end(self, args, state, control, **kwargs)`
    :   Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.

`SpectraV2Attention(config: Any | None = None, model_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, bert_mode: bool = False, pretrained_attn: nn.Module | None = None, rank: int | None = 64, **kwargs: Any)`
:   MHA wrapper around PyTorch's scaled_dot_product_attention.
    Previously named spectra_v2 (deprecated alias).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.spectra_v2.SpectraV2Attention`
    :

`SpectraformerAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Spectraformer Attention: Exact linear-time attention via random feature kernelization.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Robust weight copying across GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, and DistilBERT.
        Validates Frobenius norm >= 1e-4.

    `forward(self, hidden_states: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass with linear random feature kernel:
        1. Linear projection to Q, K, V
        2. Optional RoPE embedding
        3. GQA head expansion
        4. Random orthogonal feature projection with ReLU activation
        5. Linear attention context computation (recurrent for causal, global for bidirectional)

`SpectraformerRectify(head_dim: int, rff_features: int = 128, **kwargs)`
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

    `forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SpectralMamba(d_model: int = 768, d_state: int = 16, **kwargs: Any)`
:   Spectral modulation with actual frequency filtering and low-rank state projection.
    
    Uses a learned frequency mask over FFT bins followed by low-rank adaptation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Spectral modulation with actual frequency filtering.

`SpectralMambaAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.spectral_mamba.SpectralMambaAttention`
    :

`SpectralMambaConfig(hidden_size: int = 768, num_attention_heads: int = 12, head_dim: int = 64, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, gate_init_bias: float = 1.8)`
:   SpectralMambaConfig(hidden_size: 'int' = 768, num_attention_heads: 'int' = 12, head_dim: 'int' = 64, d_state: 'int' = 16, d_conv: 'int' = 4, expand: 'int' = 2, dropout: 'float' = 0.1, gate_init_bias: 'float' = 1.8)

    ### Instance variables

    `d_conv: int`
    :   The type of the None singleton.

    `d_state: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `expand: int`
    :   The type of the None singleton.

    `gate_init_bias: float`
    :   The type of the None singleton.

    `head_dim: int`
    :   The type of the None singleton.

    `hidden_size: int`
    :   The type of the None singleton.

    `num_attention_heads: int`
    :   The type of the None singleton.

`SpectronAttention(config: Any | None = None, old_attn: Any | None = None, dropout: float = 0.1, layer_idx: int = 0, ranks: dict[str, int] | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, add_sparse: bool = False, sparsity_level: float = 0.7, whitening: dict[str, torch.Tensor] | None = None, seed: int = 42, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `prune_all_sparse(self, sparsity: float) ‑> None`
    :

    `total_sparse_l1(self) ‑> torch.Tensor`
    :   Sum of L1 norms of all sparse residuals. Add to training loss.

`SpectronSpectralEntropyRouter(config: neuralops.attentions.aegis_spectron_fp8.AegisSpectronFP8Config)`
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

    `forward(self, X: torch.Tensor, token_valid: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SqueezingHeads(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SqueezingHeadsAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SqueezingHeadsDistiller(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`StableKernelFeatureMap(config)`
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
    :   FAVOR+ feature map with GLOBAL max subtraction for FP16 stability.

`TLinFormerAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   TLinFormer: linear-time attention with sequence-dimension low-rank projection.
    
    For an encoder with N tokens and rank r ≤ N:
        K' = E K,  V' = F V,  E ∈ R^{r×N},  F ∈ R^{r×N}
        scores = Q K'ᵀ / √d      # (B, H, N, r)
        O = softmax(scores) V'   # (B, H, N, d)
    
    Complexity O(N · r · d) instead of O(N² · d).
    
    For decoders (causal), we deliberately fall back to dense attention. A
    projected key aggregates information from all N positions, so a causal
    mask over the projected keys cannot enforce causality — the model would
    silently leak future information. Dense causal attention is correct.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `E`
    :

    `F`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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

`TPA(config: Any | None = None, rank: int = 8, pretrained_attn: nn.Module | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, max_cp_error: float = 0.35, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args, **kwargs) ‑> Callable[..., typing.Any]`
    :   Must return (output, attention_weights_or_None).

`TPAttention(config: Any | None = None, rank: int = 8, pretrained_attn: nn.Module | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, max_cp_error: float = 0.35, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

`TokenCompressedAttention(config=None, rank=64, compression_ratio=0.5, pretrained_attn=None, lora_rank=None, lora_alpha=None, comp_config=None, is_decoder=None, **kwargs)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    `supports_causal`
    :   The type of the None singleton.

    `supports_gqa`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `o_proj`
    :

    `out_proj_alias`
    :

    `q_lin_alias`
    :

`TokenRouter(hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8)`
:   Top-k token selector based on learned saliency scores with guaranteed [CLS] retention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None, is_decoder: bool = False) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`Trainer(model: transformers.modeling_utils.PreTrainedModel | torch.nn.modules.module.Module | None = None, args: transformers.training_args.TrainingArguments | None = None, data_collator: Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None, train_dataset: Dataset | IterableDataset | datasets.Dataset | None = None, eval_dataset: Dataset | dict[str, Dataset] | datasets.Dataset | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.image_processing_utils.BaseImageProcessor | transformers.feature_extraction_utils.FeatureExtractionMixin | transformers.processing_utils.ProcessorMixin | None = None, model_init: Callable[..., transformers.modeling_utils.PreTrainedModel] | None = None, compute_loss_func: <class 'collections.abc.Callable'> | None = None, compute_metrics: Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), optimizer_cls_and_kwargs: tuple[type[torch.optim.optimizer.Optimizer], dict[str, typing.Any]] | None = None, preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None)`
:   Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
    
            <Tip>
    
            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.
    
            </Tip>
    
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
        train_dataset (`torch.utils.data.Dataset` | `torch.utils.data.IterableDataset` | `datasets.Dataset`, *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.
    
            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (`torch.utils.data.Dataset` | dict[str, `torch.utils.data.Dataset`] | `datasets.Dataset`, *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.
    
            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to
            be able to choose different architectures according to hyperparameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).
    
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.
    
            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.
    
            Note that the labels (second parameter) will be `None` if the dataset does not have them.
    
    Important attributes:
    
        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device. Defaults to
          `True` unless model parallel, DeepSpeed, FSDP, full fp16/bf16 eval, or SageMaker MP is active. Can be
          overridden by subclassing `TrainingArguments` and overriding the `place_model_on_device` property.
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    ### Descendants

    * neuralops.trainer.methods.rdrop_r3f.MethodTrainer
    * neuralops.trainer.trainer.SVDAwareTrainer

    ### Static methods

    `get_optimizer_cls_and_kwargs(args: transformers.training_args.TrainingArguments, model: transformers.modeling_utils.PreTrainedModel | None = None) ‑> tuple[typing.Any, typing.Any]`
    :   Returns the optimizer class and optimizer parameters based on the training arguments.
        
        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
            model (`PreTrainedModel`, *optional*):
                The model being trained. Required for some optimizers (GaLore, Apollo, LOMO).
        
        Returns:
            A tuple containing the optimizer class and a dictionary of optimizer keyword arguments.

    ### Methods

    `add_callback(self, callback: type[transformers.trainer_callback.TrainerCallback] | transformers.trainer_callback.TrainerCallback) ‑> None`
    :   Add a callback to the current list of [`~transformers.TrainerCallback`].
        
        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.

    `autocast_smart_context_manager(self, cache_enabled: bool | None = True) ‑> contextlib.AbstractContextManager`
    :   A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here.

    `call_model_init(self, trial: optuna.Trial | dict[str, Any] | None = None) ‑> torch.nn.modules.module.Module`
    :   Invoke `model_init` to get a fresh model instance, optionally conditioned on a hyperparameter trial.

    `compute_loss(self, model: torch.nn.modules.module.Module, inputs: dict[str, torch.Tensor | Any], return_outputs: bool = False, num_items_in_batch: torch.Tensor | int | None = None) ‑> torch.Tensor | tuple[torch.Tensor, typing.Any]`
    :   How the loss is computed by Trainer. By default, all models return the loss in the first element.
        
        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.
        
        Returns:
            The loss of the model along with its output if return_outputs was set to True
        
        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.

    `compute_loss_context_manager(self) ‑> contextlib.ExitStack`
    :   A helper wrapper to group together context managers.

    `create_accelerator_and_postprocess(self) ‑> None`
    :   Create the accelerator and perform post-creation setup (FSDP, DeepSpeed, etc.).

    `create_model_card(self, language: str | None = None, license: str | None = None, tags: str | list[str] | None = None, model_name: str | None = None, finetuned_from: str | None = None, tasks: str | list[str] | None = None, dataset_tags: str | list[str] | None = None, dataset: str | list[str] | None = None, dataset_args: str | list[str] | None = None) ‑> None`
    :   Creates a draft of a model card using the information available to the `Trainer`.
        
        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `list[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `list[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `list[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `list[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `list[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.

    `create_optimizer(self, model=None) ‑> torch.optim.optimizer.Optimizer`
    :   Setup the optimizer.
        
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        
        Returns:
            `torch.optim.Optimizer`: The optimizer instance.

    `create_optimizer_and_scheduler(self, num_training_steps: int) ‑> None`
    :   Setup the optimizer and the learning rate scheduler.
        
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.

    `create_scheduler(self, num_training_steps: int, optimizer: torch.optim.optimizer.Optimizer | None = None) ‑> torch.optim.lr_scheduler.LRScheduler`
    :   Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        
        Args:
            num_training_steps (int): The number of training steps to do.
        
        Returns:
            `torch.optim.lr_scheduler.LRScheduler`: The learning rate scheduler instance.

    `evaluate(self, eval_dataset: torch.utils.data.dataset.Dataset | dict[str, torch.utils.data.dataset.Dataset] | None = None, ignore_keys: list[str] | None = None, metric_key_prefix: str = 'eval') ‑> dict[str, float]`
    :   Run evaluation and returns metrics.
        
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        
        You can also subclass and override this method to inject custom behavior.
        
        Args:
            eval_dataset (`Dataset` | dict[str, `Dataset`], *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.
        
                <Tip>
        
                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.
        
                </Tip>
        
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.

    `evaluation_loop(self, dataloader: torch.utils.data.dataloader.DataLoader, description: str, prediction_loss_only: bool | None = None, ignore_keys: list[str] | None = None, metric_key_prefix: str = 'eval') ‑> transformers.trainer_utils.EvalLoopOutput`
    :   Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        
        Works both with or without labels.

    `floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) ‑> int`
    :   For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.
        
        Args:
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.
        
        Returns:
            `int`: The number of floating-point operations.

    `get_batch_samples(self, epoch_iterator: <class 'collections.abc.Iterator'>, num_batches: int, device: torch.device) ‑> tuple[list, torch.Tensor | int | None]`
    :   Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.

    `get_cp_size(self) ‑> int`
    :   Get the context parallel size

    `get_decay_parameter_names(self, model: torch.nn.modules.module.Module) ‑> list[str]`
    :   Get all parameter names that weight decay will be applied to.
        
        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')

    `get_eval_dataloader(self, eval_dataset: str | torch.utils.data.dataset.Dataset | None = None) ‑> torch.utils.data.dataloader.DataLoader`
    :   Returns the evaluation [`~torch.utils.data.DataLoader`].
        
        Subclass and override this method if you want to inject some custom behavior.
        
        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.

    `get_learning_rates(self) ‑> list[float]`
    :   Returns the learning rate of each parameter from self.optimizer.

    `get_num_trainable_parameters(self) ‑> int`
    :   Get the number of trainable parameters.

    `get_optimizer_group(self, param: str | torch.nn.parameter.Parameter | None = None)`
    :   Returns optimizer group for a parameter if given, else returns all optimizer groups for params.
        
        Args:
            param (`str` or `torch.nn.parameter.Parameter`, *optional*):
                The parameter for which optimizer group needs to be returned.

    `get_sp_size(self) ‑> int`
    :   Get the sequence parallel size

    `get_test_dataloader(self, test_dataset: torch.utils.data.dataset.Dataset) ‑> torch.utils.data.dataloader.DataLoader`
    :   Returns the test [`~torch.utils.data.DataLoader`].
        
        Subclass and override this method if you want to inject some custom behavior.
        
        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.

    `get_total_train_batch_size(self, args: transformers.training_args.TrainingArguments) ‑> int`
    :   Calculates total batch size (micro_batch * grad_accum * dp_world_size).
        
        Accounts for all parallelism dimensions: TP, CP, and SP.
        
        Formula: dp_world_size = world_size // (tp_size * cp_size * sp_size)
        
        Where:
        - TP (Tensor Parallelism): Model layers split across GPUs
        - CP (Context Parallelism): Sequences split using Ring Attention (FSDP2)
        - SP (Sequence Parallelism): Sequences split using ALST/Ulysses (DeepSpeed)
        
        All dimensions are separate and multiplicative: world_size = dp_size * tp_size * cp_size * sp_size

    `get_tp_size(self) ‑> int`
    :   Get the tensor parallel size from either the model or DeepSpeed config.

    `get_train_dataloader(self) ‑> torch.utils.data.dataloader.DataLoader`
    :   Returns the training [`~torch.utils.data.DataLoader`].
        
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        
        Subclass and override this method if you want to inject some custom behavior.

    `hyperparameter_search(self, hp_space: Callable[['optuna.Trial'], dict[str, float]] | None = None, compute_objective: Callable[[dict[str, float]], float] | None = None, n_trials: int = 20, direction: str | list[str] = 'minimize', backend: str | transformers.trainer_utils.HPSearchBackend | None = None, hp_name: Callable[['optuna.Trial'], str] | None = None, **kwargs) ‑> transformers.trainer_utils.BestRun | list[transformers.trainer_utils.BestRun]`
    :   Launch a hyperparameter search using `optuna` or `Ray Tune`. The optimized quantity is determined
        by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
        the sum of all metrics otherwise.
        
        <Tip warning={true}>
        
        To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
        reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to
        subclass [`Trainer`] and override the method [`~Trainer.create_optimizer_and_scheduler`] for custom
        optimizer/scheduler.
        
        </Tip>
        
        Args:
            hp_space (`Callable[["optuna.Trial"], dict[str, float]]`, *optional*):
                A function that defines the hyperparameter search space. Will default to
                [`~trainer_utils.default_hp_space_optuna`] or [`~trainer_utils.default_hp_space_ray`]
                depending on your backend.
            compute_objective (`Callable[[dict[str, float]], float]`, *optional*):
                A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
                method. Will default to [`~trainer_utils.default_compute_objective`].
            n_trials (`int`, *optional*, defaults to 100):
                The number of trial runs to test.
            direction (`str` or `list[str]`, *optional*, defaults to `"minimize"`):
                If it's single objective optimization, direction is `str`, can be `"minimize"` or `"maximize"`, you
                should pick `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or
                several metrics. If it's multi objectives optimization, direction is `list[str]`, can be List of
                `"minimize"` and `"maximize"`, you should pick `"minimize"` when optimizing the validation loss,
                `"maximize"` when optimizing one or several metrics.
            backend (`str` or [`~training_utils.HPSearchBackend`], *optional*):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending
                on which one is installed. If all are installed, will default to optuna.
            hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
                A function that defines the trial/run name. Will default to None.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments for each backend:
        
                - `optuna`: parameters from
                  [optuna.study.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
                  and also the parameters `timeout`, `n_jobs` and `gc_after_trial` from
                  [optuna.study.Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
                - `ray`: parameters from [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run).
                  If `resources_per_trial` is not set in the `kwargs`, it defaults to 1 CPU core and 1 GPU (if available).
                  If `progress_reporter` is not set in the `kwargs`,
                  [ray.tune.CLIReporter](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html) is used.
        Returns:
            [`trainer_utils.BestRun` or `list[trainer_utils.BestRun]`]: All the information about the best run or best
            runs for multi-objective optimization. Experiment summary can be found in `run_summary` attribute for Ray
            backend.

    `init_hf_repo(self, token: str | None = None) ‑> None`
    :   Initializes a git repo in `self.args.hub_model_id`.

    `is_local_process_zero(self) ‑> bool`
    :   Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.

    `is_world_process_zero(self) ‑> bool`
    :   Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).

    `log(self, logs: dict[str, float], start_time: float | None = None) ‑> None`
    :   Log `logs` on the various objects watching training.
        
        Subclass and override this method to inject custom behavior.
        
        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.

    `log_metrics(self, split, metrics)`
    :   Log metrics in a specially formatted way.
        
        Under distributed environment this is done only for a process with rank 0.
        
        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`
            metrics (`dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict
        
        Notes on memory reports:
        
        In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.
        
        Now when this method is run, you will see a report that will include:
        
        ```
        init_mem_cpu_alloc_delta   =     1301MB
        init_mem_cpu_peaked_delta  =      154MB
        init_mem_gpu_alloc_delta   =      230MB
        init_mem_gpu_peaked_delta  =        0MB
        train_mem_cpu_alloc_delta  =     1345MB
        train_mem_cpu_peaked_delta =        0MB
        train_mem_gpu_alloc_delta  =      693MB
        train_mem_gpu_peaked_delta =        7MB
        ```
        
        **Understanding the reports:**
        
        - the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
            will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
            `__init__` will be reported along with the `eval_` metrics.
        - the third segment, is either `cpu` or `gpu`, tells you whether it's the general RAM or the gpu0 memory
            metric.
        - `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
            stage - it can be negative if a function released more memory than it allocated.
        - `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
            memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
            `peaked_delta` and you know how much memory was needed to complete that stage.
        
        The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
        main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
        use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
        memory than the rest since it stores the gradient and optimizer states for all participating GPUs. Perhaps in the
        future these reports will evolve to measure those too.
        
        The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
        memory shared with other processes. It is important to note that it does not include swapped out memory, so the
        reports could be imprecise.
        
        The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
        that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
        reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn't report memory allocations
        outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
        was dropped in favor of the memory sampling approach, which reads the current process memory usage.
        
        The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
        `torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as
        `torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very
        first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.
        
        Note that this tracker doesn't account for memory allocations outside of [`Trainer`]'s `__init__`, `train`,
        `evaluate` and `predict` calls.
        
        Because `evaluation` calls may happen during `train`, we can't handle nested invocations because
        `torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker
        will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved
        it will be possible to change this class to be re-entrant. Until then we will only track the outer level of
        `train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it's the latter
        that will account for its memory usage and that of the former.
        
        This also means that if any other tool that is used along the [`Trainer`] calls
        `torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt
        the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.
        
        For best performance you may want to consider turning the memory profiling off for production runs.

    `metrics_format(metrics: dict[str, float]) ‑> dict[str, float]`
    :   Reformat Trainer metrics values to a human-readable format.
        
        Args:
            metrics (`dict[str, float]`):
                The metrics returned from train/evaluate/predict
        
        Returns:
            metrics (`dict[str, float]`): The reformatted metrics

    `num_examples(self, dataloader: torch.utils.data.dataloader.DataLoader) ‑> int`
    :   Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can

    `pop_callback(self, callback: type[transformers.trainer_callback.TrainerCallback] | transformers.trainer_callback.TrainerCallback) ‑> transformers.trainer_callback.TrainerCallback | None`
    :   Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.
        
        If the callback is not found, returns `None` (and no error is raised).
        
        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.
        
        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.

    `predict(self, test_dataset: torch.utils.data.dataset.Dataset, ignore_keys: list[str] | None = None, metric_key_prefix: str = 'test') ‑> transformers.trainer_utils.PredictionOutput`
    :   Run prediction and returns predictions and potential metrics.
        
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        
        <Tip>
        
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        
        </Tip>
        
        Returns: *NamedTuple* A namedtuple with the following keys:
        
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).

    `prediction_step(self, model: torch.nn.modules.module.Module, inputs: dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: list[str] | None = None) ‑> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]`
    :   Perform an evaluation step on `model` using `inputs`.
        
        Subclass and override to inject custom behavior.
        
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.
        
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        
        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).

    `push_to_hub(self, commit_message: str | None = 'End of training', blocking: bool = True, token: str | None = None, revision: str | None = None, **kwargs) ‑> huggingface_hub.hf_api.CommitInfo`
    :   Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.
        
        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            token (`str`, *optional*, defaults to `None`):
                Token with write permission to overwrite Trainer's original args.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the "main" branch.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].
        
        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.

    `remove_callback(self, callback: type[transformers.trainer_callback.TrainerCallback] | transformers.trainer_callback.TrainerCallback) ‑> None`
    :   Remove a callback from the current list of [`~transformers.TrainerCallback`].
        
        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.

    `save_metrics(self, split, metrics, combined=True)`
    :   Save metrics into a json file for that split, e.g. `train_results.json`.
        
        Under distributed environment this is done only for a process with rank 0.
        
        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
            metrics (`dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (`bool`, *optional*, defaults to `True`):
                Creates combined metrics by updating `all_results.json` with metrics of this call
        
        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.

    `save_model(self, output_dir: str | None = None) ‑> None`
    :   Will save the model, so you can reload it using `from_pretrained()`.
        
        Will only save from the main process.

    `save_state(self)`
    :   Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model.
        
        Under distributed environment this is done only for a process with rank 0.

    `set_initial_training_values(self, args: transformers.training_args.TrainingArguments, dataloader: torch.utils.data.dataloader.DataLoader) ‑> tuple[int, int, int, int, int, int | None, int]`
    :   Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `total_train_batch_size`
        - `steps_in_epoch` (total batches per epoch)
        - `max_steps`

    `store_flos(self) ‑> None`
    :   Store the number of floating-point operations that went into the model.

    `train(self, resume_from_checkpoint: str | bool | None = None, trial: optuna.Trial | dict[str, Any] | None = None, ignore_keys_for_eval: list[str] | None = None) ‑> transformers.trainer_utils.TrainOutput`
    :   Main training entry point.
        
        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`list[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        
        Returns:
            [`~trainer_utils.TrainOutput`]: Object containing the global step count, training loss, and metrics.

    `training_step(self, model: torch.nn.modules.module.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: torch.Tensor | int | None = None) ‑> torch.Tensor`
    :   Perform a training step on a batch of inputs.
        
        Subclass and override to inject custom behavior.
        
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.
        
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.

`TrainingArguments(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 5e-05, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: str | list[str] | None = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool = False, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, gradient_checkpointing: bool = False, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 500, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: str | list[str] | None = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: str | Literal[False] | None = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, remove_unused_columns: bool = True, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: list[transformers.trainer_utils.FSDPOption] | str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, warmup_ratio: float | None = None, logging_dir: str | None = None, local_rank: int = -1)`
:   Configuration class for controlling all aspects of model training with the Trainer.
    TrainingArguments centralizes all hyperparameters, optimization settings, logging preferences, and infrastructure choices needed for training.
    
    [`HfArgumentParser`] can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    
    Parameters:
        output_dir (`str`, *optional*, defaults to `"trainer_output"`):
            The output directory where the model predictions and checkpoints will be written.
    
        > Training Duration and Batch Size
    
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size *per device*. The **global batch size** is computed as:
            `per_device_train_batch_size * number_of_devices` in multi-GPU or distributed setups.
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            Overrides `num_train_epochs`. If set to a positive number, the total number of training steps to perform.
            For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
            `max_steps` is reached.
    
        > Learning Rate & Scheduler
    
        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for the optimizer. This is typically the peak learning rate when using a scheduler with warmup.
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The learning rate scheduler type to use. See [`SchedulerType`] for all possible values. Common choices:
                - "linear" = [`get_linear_schedule_with_warmup`]
                - "cosine" = [`get_cosine_schedule_with_warmup`]
                - "constant" =  [`get_constant_schedule`]
                - "constant_with_warmup" = [`get_constant_schedule_with_warmup`]
        lr_scheduler_kwargs (`dict` or `str`, *optional*, defaults to `None`):
            The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values.
        warmup_steps (`int` or `float`, *optional*, defaults to 0):
            Number of steps for a linear warmup from 0 to `learning_rate`. Warmup helps stabilize training in the initial phase. Can be:
                - An integer: exact number of warmup steps
                - A float in range [0, 1): interpreted as ratio of total training steps
    
        > Optimizer
    
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"` (for torch>=2.8 `"adamw_torch_fused"`)):
            The optimizer to use. Common options:
                - `"adamw_torch"`: PyTorch's AdamW (recommended default)
                - `"adamw_torch_fused"`: Fused AdamW kernel
                - `"adamw_hf"`: HuggingFace's AdamW implementation
                - `"sgd"`: Stochastic Gradient Descent with momentum
                - `"adafactor"`: Memory-efficient optimizer for large models
                - `"adamw_8bit"`: 8-bit AdamW (requires bitsandbytes)
                See [`OptimizerNames`] for the complete list.
        optim_args (`str`, *optional*):
            Optional arguments that are supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore.
        weight_decay (`float`, *optional*, defaults to 0):
            Weight decay coefficient applied by the optimizer (not the loss function). Adds L2
            regularization to prevent overfitting by penalizing large weights. Automatically
            excluded from bias and LayerNorm parameters. Typical values: 0.01 (standard), 0.1
            (stronger regularization), 0.0 (no regularization).
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The exponential decay rate for the first moment estimates (momentum) in Adam-based
            optimizers. Controls how much history of gradients to retain.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The exponential decay rate for the second moment estimates (variance) in Adam-based
            optimizers. Controls adaptive learning rate scaling.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            Epsilon value for numerical stability in Adam-based optimizers. Prevents division by
            zero in the denominator of the update rule.
        optim_target_modules (`Union[str, list[str]]`, *optional*):
            The target modules to optimize, i.e. the module names that you would like to train.
            Currently used for the [GaLore algorithm](https://huggingface.co/papers/2403.03507) and [APOLLO algorithm](https://huggingface.co/papers/2412.05270).
            See [GaLore implementation](https://github.com/jiaweizzhao/GaLore) and [APOLLO implementation](https://github.com/zhuhanqing/APOLLO) for more details.
            You need to make sure to pass a valid GaLore or APOLLO optimizer, e.g., one of: "apollo_adamw", "galore_adamw", "galore_adamw_8bit", "galore_adafactor" and make sure that the target modules are `nn.Linear` modules only.
    
        > Regularization & Training Stability
    
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of update steps to accumulate gradients before performing a backward/update pass.
            Simulates larger batch sizes without additional memory. Effective batch size =
            `per_device_train_batch_size × num_devices × gradient_accumulation_steps`.
            > [!TIP]
            > When using gradient accumulation, one "step" is counted as one step with a backward pass. Therefore, logging, evaluation, and saving will occur every `gradient_accumulation_steps × xxx_step` training examples.
        average_tokens_across_devices (`bool`, *optional*, defaults to `True`):
            Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize
            num_tokens_in_batch for precise loss calculation. Reference:
            https://github.com/huggingface/transformers/issues/34242
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm for gradient clipping. Applied after backward pass, before
            optimizer step. Prevents gradient explosion by scaling down gradients when their global
            norm exceeds this threshold. Set to 0 to disable clipping. Typical values:
            1.0 (standard), 0.5 (more conservative), 5.0 (less aggressive).
        label_smoothing_factor (`float`, *optional*, defaults to 0.0):
            Label smoothing factor to prevent overconfidence. Replaces hard 0/1 targets with soft
            targets: 0 becomes `ε/num_labels` and 1 becomes `1 - ε + ε/num_labels`, where
            ε = `label_smoothing_factor`. Zero means no smoothing. Typical range: 0.0 to 0.1.
    
        > Mixed Precision Training
    
        bf16 (`bool`, *optional*, defaults to `False`):
            Enable bfloat16 (BF16) mixed precision training
            Generally preferred over FP16 due to better numerical stability and no loss scaling required.
        fp16 (`bool`, *optional*, defaults to `False`):
            Enable float16 (FP16) mixed precision training.
            Consider using BF16 instead if your hardware supports it.
        bf16_full_eval (`bool`, *optional*, defaults to `False`):
            Use full BF16 precision for evaluation (not just mixed precision). Faster and saves
            memory but may affect metric values slightly. Only applies during evaluation.
        fp16_full_eval (`bool`, *optional*, defaults to `False`):
            Use full FP16 precision for evaluation (not just mixed precision). Faster and saves
            memory but may affect metric values slightly. Only applies during evaluation.
        tf32 (`bool`, *optional*):
            Enable TensorFloat-32 (TF32) mode on Ampere and newer GPUs. TF32 uses 19-bit precision
            for matrix multiplications (instead of FP32's 23-bit), providing up to 8x speedup with
            negligible accuracy loss. Default depends on PyTorch version. See
            [TF32 docs](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32).
    
        > Gradient Checkpointing
    
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Enable gradient checkpointing to trade compute for memory. Reduces memory usage by
            clearing activations during forward pass and recomputing them during backward pass.
            Enables training larger models or batch sizes at the cost of ~20% slower training.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments passed to `gradient_checkpointing_enable()`.
    
        > Compilation
    
        torch_compile (`bool`, *optional*, defaults to `False`):
            Compile the model using PyTorch 2.0's `torch.compile()` for faster training. Can provide
            20-50% speedup with no code changes. Uses default compilation settings unless
            `torch_compile_backend` or `torch_compile_mode` are specified.
        torch_compile_backend (`str`, *optional*):
            Backend for `torch.compile()`. If set, automatically enables `torch_compile`. Options
            include `"inductor"` (default), `"aot_eager"`, `"cudagraphs"`. Backends vary by PyTorch
            version - see PyTorch docs for available options.
        torch_compile_mode (`str`, *optional*):
            Compilation mode for `torch.compile()`. If set, automatically enables `torch_compile`.
            Options: `"default"`, `"reduce-overhead"` (minimize Python overhead), `"max-autotune"`
            (aggressive optimization, slower compile time).
    
        > Kernels
    
        use_liger_kernel (`bool`, *optional*, defaults to `False`):
            Enable [Liger Kernel](https://github.com/linkedin/Liger-Kernel) optimizations. Increases
            multi-GPU throughput by ~20% and reduces memory usage by ~60%. Works with Flash Attention,
            FSDP, and DeepSpeed. Currently supports Llama, Mistral, Mixtral, and Gemma models.
        liger_kernel_config (`Optional[dict]`, *optional*):
            Configuration for Liger Kernel. Passed as kwargs to `_apply_liger_kernel_to_instance()`.
            Options typically include: `"rope"`, `"swiglu"`, `"cross_entropy"`,
            `"fused_linear_cross_entropy"`, `"rms_norm"`. If `None`, uses default configuration.
    
        > Additional Optimizations
    
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not to enable cache for the model. For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance
            for instruction fine-tuning. Check out the [original paper](https://huggingface.co/papers/2310.05914) and the
            [original code](https://github.com/neelsjain/NEFTune). Support transformers `PreTrainedModel` and also
            `PeftModel` from peft. The original paper used values in the range [5.0, 15.0].
        torch_empty_cache_steps (`int`, *optional*):
            Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache will not be emptied.
            This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).
        auto_find_batch_size (`bool`, *optional*, defaults to `False`)
            Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding
            CUDA Out-of-Memory errors.
    
        > Logging & Monitoring Training
    
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:
                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.
        logging_steps (`int` or `float`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
            range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log the first `global_step` or not.
        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_nan_inf_filter (`bool`, *optional*, defaults to `True`):
             Filter out NaN and Inf losses when logging. If `True`, replaces NaN/Inf losses with the
            average of recent valid losses. Does not affect gradient computation, only logging.
        include_num_input_tokens_seen (`Optional[Union[str, bool]]`, *optional*, defaults to "no"):
            Whether to track the number of input tokens seen. Must be one of ["all", "non_padding", "no"] or a boolean value which map to "all" or "no".
            May be slower in distributed training as gather operations must be called.
    
        > Logging
    
        log_level (`str`, *optional*, defaults to `passive`):
            Logging level for the main process. Options: `"debug"`, `"info"`, `"warning"`, `"error"`,
            `"critical"`, or `"passive"` (doesn't change the current Transformers logging level,
            which defaults to `"warning"`)
        log_level_replica (`str`, *optional*, defaults to `"warning"`):
            Logging level for replica processes in distributed training. Same options as `log_level`.
        disable_tqdm (`bool`, *optional*):
            Disable tqdm progress bars. Defaults to `True` if `log_level` is warning or lower, `False` otherwise.
    
        > Experiment Tracking Integration
    
        report_to (`str` or `list[str]`, *optional*, defaults to `"none"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"swanlab"`,
            `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations installed, `"none"`
            for no integrations.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for [trackio](https://github.com/gradio-app/trackio),
            [wandb](https://www.wandb.com/), [mlflow](https://www.mlflow.org/), [comet](https://www.comet.com/site) and
            [swanlab](https://swanlab.cn) logging.
        project (`str`, *optional*, defaults to `"huggingface"`):
            The name of the project to use for logging. Currently, only used by Trackio.
        trackio_space_id (`str` or `None`, *optional*, defaults to `None`):
            The Hugging Face Space ID to use for live Trackio logging with a Gradio-based Space. Should be a full
            Space name like `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` (the Space is created in
            the currently logged-in user's namespace). If `None`, metrics are logged only to a **local** directory (no
            Space on the Hub). That Gradio Space has **read and write** access to the Trackio **Bucket**, which is what you want while
            training is **in progress**—for example when **resuming** a partial run or **aggregating logs** from multiple
            machines. The Space will be **public** unless you set `hub_private_repo=True` or your organization's default is to
            create private Spaces.
        trackio_bucket_id (`str` or `None`, *optional*, defaults to `None`):
            Optional Hugging Face Bucket id for Trackio. If unset, Trackio derives one. Used together with a Gradio Space
            (`trackio_space_id`) and when deploying a static Space (`trackio_static_space_id` is not `False`).
        trackio_static_space_id (`str`, `False`, or `None`, *optional*, defaults to `None`):
            The Hugging Face Space ID to use for static Space created after training is complete. Should be a full
            Space name like `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` (the Space is created in
            the currently logged-in user's namespace). If False, no static Space will be created. If None, and model is pushed to the Hub,
            a static Space will be created with a default name and this will be linked from the model card. The Space will be public
            unless you set `hub_private_repo=True` or your organization's default is to create private Spaces.
    
        > Evaluation
    
        eval_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            When to run evaluation. Options:
            - `"no"`: No evaluation during training
            - `"steps"`: Evaluate every `eval_steps`
            - `"epoch"`: Evaluate at the end of each epoch
        eval_steps (`int` or `float`, *optional*):
            Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1,
            will be interpreted as ratio of total training steps.
        eval_delay (`float`, *optional*):
            Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
            eval_strategy.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per device accelerator core/CPU for evaluation.
        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        eval_on_start (`bool`, *optional*, defaults to `False`):
            Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly.
        eval_do_concat_batches (`bool`, *optional*, defaults to `True`):
            Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`,
            will instead store them as lists, with each batch kept separate.
        eval_use_gather_object (`bool`, *optional*, defaults to `False`):
            Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices. This should only be enabled if users are not just returning tensors, and this is actively discouraged by PyTorch.
            This is useful when the labels structure is non standard, like in computer vision tasks.
        eval_accumulation_steps (`int`, *optional*):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on the device accelerator before being moved to the CPU (faster but
            requires more memory).
    
        > Metrics Computation
    
        include_for_metrics (`list[str]`, *optional*, defaults to `[]`):
            Include additional data in the `compute_metrics` function if needed for metrics computation.
            Possible options to add to `include_for_metrics` list:
            - `"inputs"`: Input data passed to the model, intended for calculating input dependent metrics.
            - `"loss"`: Loss values computed during evaluation, intended for calculating loss dependent metrics.
        batch_eval_metrics (`bool`, *optional*, defaults to `False`):
            If set to `True`, evaluation will call compute_metrics at the end of each batch to accumulate statistics
            rather than saving all eval logits in memory. When set to `True`, you must pass a compute_metrics function
            that takes a boolean argument `compute_result`, which when passed `True`, will trigger the final global
            summary statistics from the batch-level summary statistics you've accumulated over the evaluation set.
    
        > Checkpointing & Saving
    
        save_only_model (`bool`, *optional*, defaults to `False`):
            Save only model weights, not optimizer/scheduler/RNG state. Significantly reduces
            checkpoint size but prevents resuming training from the checkpoint. Use when you only
            need the trained model for inference, not continued training.
            You can only load the model using `from_pretrained` with this option set to `True`.
        save_strategy (`str` or [`~trainer_utils.SaveStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:
                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
                - `"best"`: Save is done whenever a new `best_metric` is achieved.
        save_steps (`int` or `float`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a
            float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.
            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        save_total_limit (`int`, *optional*):
            Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. When
            `load_best_model_at_end=True`, the best checkpoint is always retained plus the most
            recent ones. For example, `save_total_limit=5` keeps the 4 most recent plus the best
        enable_jit_checkpoint (`bool`, *optional*, defaults to `False`):
            Enable Just-In-Time checkpointing on SIGTERM signal for graceful termination on
            preemptible workloads. **Important**: Configure your orchestrator's graceful shutdown
            period to allow sufficient time. For Kubernetes, set `terminationGracePeriodSeconds`
            (default 30s is usually insufficient). For Slurm, use `--signal=USR1@<seconds>`.
            Required grace period ≥ longest iteration time + checkpoint save time.
    
        > Hugging Face Hub Integration
    
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push the model to the Hub every time the model is saved. If this is activated,
            `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content
            will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
            [`~Trainer.save_model`] will also trigger a push.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `hf auth login`.
        hub_private_repo (`bool`, *optional*):
            Whether to make the repo private. If `None` (default), the repo will be public unless the organization's
            default is private. This value is ignored if the repo already exists. If reporting to Trackio with
            deployment to Hugging Face Spaces enabled, the same logic determines whether the Space is private.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output_dir_name* being the
            name of `output_dir`.
        hub_strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
            Defines what and when to push to Hub. Options:
                - `"end"`: Push only at the end of training
                - `"every_save"`: Push on each save (async to not block training)
                - `"checkpoint"`: Like `"every_save"` plus push latest checkpoint to `"last-checkpoint"` subfolder for easy resuming
                - `"all_checkpoints"`: Push all checkpoints as they appear
        hub_always_push (`bool`, *optional*, defaults to `False`):
            Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not finished.
        hub_revision (`str`, *optional*):
            The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.
    
        > Best Model Tracking
    
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Load the best checkpoint at the end of training. Requires `eval_strategy` to be set.
            When enabled, the best checkpoint is always saved (see `save_total_limit`).
            <Tip>
            When `True`, `save_strategy` must match `eval_strategy` (unless `save_strategy` is `"best"`), and if using `"steps"`,
            `save_steps` must be a multiple of `eval_steps`.
            </Tip>
        metric_for_best_model (`str`, *optional*):
            Metric to use for comparing models when `load_best_model_at_end=True`. Must be a metric
            name returned by evaluation, with or without the `"eval_"` prefix. Defaults to `"loss"`.
            If you set this, `greater_is_better` will default to `True` unless the name ends with
            `"loss"`. Examples: `"accuracy"`, `"f1"`, `"eval_bleu"`.
        greater_is_better (`bool`, *optional*):
            Whether higher metric values are better. Defaults based on `metric_for_best_model`:
            `True` if the metric name doesn't end in `"loss"`, `False` otherwise.
    
        > Resuming Training
    
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, skip fast-forwarding through the dataset to reach the previous
            state. If `True`, training starts from the beginning of the dataset (faster resume but
            results won't match interrupted training). If `False`, skips seen data (slower resume
            but exact continuation).
        restore_callback_states_from_checkpoint (`bool`, *optional*, defaults to `False`):
            Restore callback states from checkpoint when resuming. If `True`, will override callbacks
            passed to Trainer if they exist in the checkpoint.
    
        > Reproducibility
    
        full_determinism (`bool`, *optional*, defaults to `False`)
            If `True`, [`enable_full_determinism`] is called instead of [`set_seed`] to ensure reproducible results in
            distributed training. Important: this will negatively impact the performance, so only use it for debugging.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        data_seed (`int`, *optional*):
            Random seed to be used with data samplers. If not set, random generators for data sampling will use the
            same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model
            seed.
    
        > Hardware Configuration
    
        use_cpu (`bool`, *optional*, defaults to `False`):
            Whether or not to use cpu. If set to False, we will use the available torch device/backend.
    
        > Accelerate Configuration
    
        accelerator_config (`str`, `dict`, or `AcceleratorConfig`, *optional*):
            Configuration for the internal Accelerate integration. Can be:
                - Path to JSON config file: `"accelerator_config.json"`
                - Dictionary with config options
                - `AcceleratorConfig` instance
            Key options:
                - `split_batches` (`bool`, defaults to `False`): Whether to split batches across devices.
                    If `True`, actual batch size is the same on all devices (total must be divisible by
                    num_processes). If `False`, each device gets the specified batch size.
                - `dispatch_batches` (`bool`): If `True`, only main process iterates through dataloader
                    and dispatches batches to devices. Defaults to `True` for `IterableDataset`, `False`
                    otherwise.
                - `even_batches` (`bool`, defaults to `True`): Duplicate samples from dataset start to
                    ensure all workers get equal batch sizes.
                - `use_seedable_sampler` (`bool`, defaults to `True`): Use fully seedable random sampler
                    for reproducibility.
                - `use_configured_state` (`bool`, defaults to `False`): Use pre-initialized
                    `AcceleratorState`/`PartialState` instead of creating new one. May cause issues with
                    hyperparameter tuning.
    
        parallelism_config (`ParallelismConfig`, *optional*):
            Parallelism configuration for the training run. Requires Accelerate `1.10.1`
    
        > Dataloader
    
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether you want to pin memory in data loaders or not. Will default to `True`.
        dataloader_persistent_workers (`bool`, *optional*, defaults to `False`):
            If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
            increase RAM usage. Will default to `False`.
        dataloader_prefetch_factor (`int`, *optional*):
            Number of batches loaded in advance by each worker.
            2 means there will be a total of 2 * num_workers batches prefetched across all workers.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            Whether or not to automatically remove the columns unused by the model forward method.
        label_names (`list[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.
            Will eventually default to the list of argument names accepted by the model that contain the word "label",
            except if the model used is one of the `XxxForQuestionAnswering` in which case it will also include the
            `["start_positions", "end_positions"]` keys.
            You should only specify `label_names` if you're using custom label names or if your model's `forward` consumes multiple label tensors (e.g., extractive QA).
        train_sampling_strategy (`str`, *optional*, defaults to `"random"`):
            The sampler to use for the training dataloader. Possible values are:
    
                - `"random"`: Uses `RandomSampler` (default).
                - `"sequential"`: Uses `SequentialSampler`.
                - `"group_by_length"`: Uses `LengthGroupedSampler` to group samples of roughly the same length
                  together (to minimize padding and be more efficient).
    
            Note: When using an `IterableDataset`, this argument is ignored.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `train_sampling_strategy` is `"group_by_length"` and the dataset
            is an instance of `Dataset`.
    
        > DDP (DistributedDataParallel)
    
        ddp_find_unused_parameters (`bool`, *optional*):
            When using distributed training, the value of the flag `find_unused_parameters` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_bucket_cap_mb (`int`, *optional*):
            When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
        ddp_broadcast_buffers (`bool`, *optional*):
            When using distributed training, the value of the flag `broadcast_buffers` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_static_graph (`bool`, *optional*):
            When using distributed training, the value of the flag `static_graph` passed to
            `DistributedDataParallel`.
        ddp_backend (`str`, *optional*):
            The backend to use for distributed training. Must be one of `"nccl"`, `"mpi"`, `"xccl"`, `"gloo"`, `"hccl"`.
        ddp_timeout (`int`, *optional*, defaults to 1800):
            The timeout for `torch.distributed.init_process_group` calls, used to avoid GPU socket timeouts when
            performing slow operations in distributed runnings. Please refer to the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
            information.
    
        > FSDP (Fully Sharded Data Parallel)
    
        fsdp (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `None`):
            Enable PyTorch Fully Sharded Data Parallel (FSDP) for distributed training. Options:
                - `"full_shard"`: Shard parameters, gradients, and optimizer states (most memory efficient)
                - `"shard_grad_op"`: Shard only optimizer states and gradients (ZeRO-2)
                - `"hybrid_shard"`: Full shard within nodes, replicate across nodes
                - `"hybrid_shard_zero2"`: Shard gradients/optimizer within nodes, replicate across nodes
                - `"offload"`: Offload parameters and gradients to CPU (only with `"full_shard"` or
                    `"shard_grad_op"`)
                - `"auto_wrap"`: Automatically wrap layers using `default_auto_wrap_policy`
        fsdp_config (`str` or `dict`, *optional*):
            Config to be used with fsdp (Pytorch Distributed Parallel Training). The value is either a location of
            fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.
    
            A List of config and its options:
                - fsdp_version (`int`, *optional*, defaults to `1`):
                    The version of FSDP to use. Defaults to 1.
                - min_num_params (`int`, *optional*, defaults to `0`):
                    FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
                    passed).
                - transformer_layer_cls_to_wrap (`list[str]`, *optional*):
                    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
                    `T5Block` .... (useful only when `fsdp` flag is passed).
                - backward_prefetch (`str`, *optional*)
                    FSDP's backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
                    `fsdp` field is passed).
    
                    A list of options along the following:
    
                    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter's
                      gradient computation.
                    - `"backward_post"` : This prefetches the next set of parameters after the current set of
                      parameter's gradient computation.
                - forward_prefetch (`bool`, *optional*, defaults to `False`)
                    FSDP's forward prefetch mode (useful only when `fsdp` field is passed).
                     If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
                     forward pass.
                - limit_all_gathers (`bool`, *optional*, defaults to `False`)
                    FSDP's limit_all_gathers (useful only when `fsdp` field is passed).
                     If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
                     all-gathers.
                - use_orig_params (`bool`, *optional*, defaults to `True`)
                    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
                    frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please
                    refer this
                    [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019
                - sync_module_states (`bool`, *optional*, defaults to `True`)
                    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
                    ensure they are the same across all ranks after initialization
                - cpu_ram_efficient_loading (`bool`, *optional*, defaults to `False`)
                    If `"True"`, only the first process loads the pretrained model checkpoint while all other processes
                    have empty weights.  When this setting as `"True"`, `sync_module_states` also must to be `"True"`,
                    otherwise all the processes except the main process would have random weights leading to unexpected
                    behaviour during training.
                - activation_checkpointing (`bool`, *optional*, defaults to `False`):
                    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
                    certain layers and recomputing them during a backward pass. Effectively, this trades extra
                    computation time for reduced memory usage.
                - xla (`bool`, *optional*, defaults to `False`):
                    Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature
                    and its API may evolve in the future.
                - xla_fsdp_settings (`dict`, *optional*)
                    The value is a dictionary which stores the XLA FSDP wrapping parameters.
    
                    For a complete list of options, please see [here](
                    https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
                - xla_fsdp_grad_ckpt (`bool`, *optional*, defaults to `False`):
                    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
                    used when the xla flag is set to true, and an auto wrapping policy is specified through
                    fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.
    
        > DeepSpeed
    
        deepspeed (`str` or `dict`, *optional*):
             Enable [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) integration. Value is either:
                - Path to DeepSpeed JSON config file: `"ds_config.json"`
                - Loaded config as dictionary
          > [!TIP]
          > If using ZeRO initialization, instantiate your model *after* initializing
          `TrainingArguments`, otherwise ZeRO won't be applied.
    
        > Debugging & Profiling (Experimental)
    
        debug (`str` or list of [`~debug_utils.DebugOption`], *optional*, defaults to `""`):
            Enable one or more debug features. This is an experimental feature.
            Possible options are:
            - "underflow_overflow": detects overflow in model's input/outputs and reports the last frames that led to
              the event
            - "tpu_metrics_debug": print debug metrics on TPU
        skip_memory_metrics (`bool`, *optional*, defaults to `True`):
            Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
            down the training and evaluation speed.
    
        > External Script Flags (not used by Trainer)
    
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.

    ### Class variables

    `default_optim`
    :   The type of the None singleton.

    `is_torch_greater_or_equal_than_2_8`
    :   The type of the None singleton.

    ### Instance variables

    `accelerator_config: dict | str | None`
    :   The type of the None singleton.

    `adam_beta1: float`
    :   The type of the None singleton.

    `adam_beta2: float`
    :   The type of the None singleton.

    `adam_epsilon: float`
    :   The type of the None singleton.

    `auto_find_batch_size: bool`
    :   The type of the None singleton.

    `average_tokens_across_devices: bool`
    :   The type of the None singleton.

    `batch_eval_metrics: bool`
    :   The type of the None singleton.

    `bf16: bool`
    :   The type of the None singleton.

    `bf16_full_eval: bool`
    :   The type of the None singleton.

    `data_seed: int | None`
    :   The type of the None singleton.

    `dataloader_drop_last: bool`
    :   The type of the None singleton.

    `dataloader_num_workers: int`
    :   The type of the None singleton.

    `dataloader_persistent_workers: bool`
    :   The type of the None singleton.

    `dataloader_pin_memory: bool`
    :   The type of the None singleton.

    `dataloader_prefetch_factor: int | None`
    :   The type of the None singleton.

    `ddp_backend: str | None`
    :   The type of the None singleton.

    `ddp_broadcast_buffers: bool | None`
    :   The type of the None singleton.

    `ddp_bucket_cap_mb: int | None`
    :   The type of the None singleton.

    `ddp_find_unused_parameters: bool | None`
    :   The type of the None singleton.

    `ddp_static_graph: bool | None`
    :   The type of the None singleton.

    `ddp_timeout: int`
    :   The type of the None singleton.

    `ddp_timeout_delta: datetime.timedelta`
    :   The actual timeout for torch.distributed.init_process_group since it expects a timedelta variable.

    `debug: str | list[transformers.debug_utils.DebugOption]`
    :   The type of the None singleton.

    `deepspeed: dict | str | None`
    :   The type of the None singleton.

    `device: torch.device`
    :   The device used by this process.

    `disable_tqdm: bool | None`
    :   The type of the None singleton.

    `do_eval: bool`
    :   The type of the None singleton.

    `do_predict: bool`
    :   The type of the None singleton.

    `do_train: bool`
    :   The type of the None singleton.

    `enable_jit_checkpoint: bool`
    :   The type of the None singleton.

    `eval_accumulation_steps: int | None`
    :   The type of the None singleton.

    `eval_batch_size: int`
    :   The actual batch size for evaluation.

    `eval_delay: float`
    :   The type of the None singleton.

    `eval_do_concat_batches: bool`
    :   The type of the None singleton.

    `eval_on_start: bool`
    :   The type of the None singleton.

    `eval_steps: float | None`
    :   The type of the None singleton.

    `eval_strategy: transformers.trainer_utils.IntervalStrategy | str`
    :   The type of the None singleton.

    `eval_use_gather_object: bool`
    :   The type of the None singleton.

    `fp16: bool`
    :   The type of the None singleton.

    `fp16_full_eval: bool`
    :   The type of the None singleton.

    `fsdp: list[transformers.trainer_utils.FSDPOption] | str | None`
    :   The type of the None singleton.

    `fsdp_config: dict[str, typing.Any] | str | None`
    :   The type of the None singleton.

    `full_determinism: bool`
    :   The type of the None singleton.

    `gradient_accumulation_steps: int`
    :   The type of the None singleton.

    `gradient_checkpointing: bool`
    :   The type of the None singleton.

    `gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None`
    :   The type of the None singleton.

    `greater_is_better: bool | None`
    :   The type of the None singleton.

    `hub_always_push: bool`
    :   The type of the None singleton.

    `hub_model_id: str | None`
    :   The type of the None singleton.

    `hub_private_repo: bool | None`
    :   The type of the None singleton.

    `hub_revision: str | None`
    :   The type of the None singleton.

    `hub_strategy: transformers.trainer_utils.HubStrategy | str`
    :   The type of the None singleton.

    `hub_token: str | None`
    :   The type of the None singleton.

    `ignore_data_skip: bool`
    :   The type of the None singleton.

    `include_for_metrics: list[str]`
    :   The type of the None singleton.

    `include_num_input_tokens_seen: str | bool`
    :   The type of the None singleton.

    `label_names: list[str] | None`
    :   The type of the None singleton.

    `label_smoothing_factor: float`
    :   The type of the None singleton.

    `learning_rate: float`
    :   The type of the None singleton.

    `length_column_name: str`
    :   The type of the None singleton.

    `liger_kernel_config: dict[str, bool] | None`
    :   The type of the None singleton.

    `load_best_model_at_end: bool`
    :   The type of the None singleton.

    `local_process_index`
    :   The index of the local process used.

    `local_rank: int`
    :   The type of the None singleton.

    `log_level: str`
    :   The type of the None singleton.

    `log_level_replica: str`
    :   The type of the None singleton.

    `log_on_each_node: bool`
    :   The type of the None singleton.

    `logging_dir: str | None`
    :   The type of the None singleton.

    `logging_first_step: bool`
    :   The type of the None singleton.

    `logging_nan_inf_filter: bool`
    :   The type of the None singleton.

    `logging_steps: float`
    :   The type of the None singleton.

    `logging_strategy: transformers.trainer_utils.IntervalStrategy | str`
    :   The type of the None singleton.

    `lr_scheduler_kwargs: dict | str | None`
    :   The type of the None singleton.

    `lr_scheduler_type: transformers.trainer_utils.SchedulerType | str`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `max_steps: int`
    :   The type of the None singleton.

    `metric_for_best_model: str | None`
    :   The type of the None singleton.

    `n_gpu`
    :   The number of GPUs used by this process.
        
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.

    `neftune_noise_alpha: float | None`
    :   The type of the None singleton.

    `num_train_epochs: float`
    :   The type of the None singleton.

    `optim: transformers.training_args.OptimizerNames | str`
    :   The type of the None singleton.

    `optim_args: str | None`
    :   The type of the None singleton.

    `optim_target_modules: str | list[str] | None`
    :   The type of the None singleton.

    `output_dir: str | None`
    :   The type of the None singleton.

    `parallel_mode`
    :   The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:
        
        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.

    `parallelism_config: accelerate.parallelism_config.ParallelismConfig | None`
    :   The type of the None singleton.

    `per_device_eval_batch_size: int`
    :   The type of the None singleton.

    `per_device_train_batch_size: int`
    :   The type of the None singleton.

    `place_model_on_device: bool | None`
    :   Can be subclassed and overridden for some specific integrations.

    `prediction_loss_only: bool`
    :   The type of the None singleton.

    `process_index`
    :   The index of the current process used.

    `project: str`
    :   The type of the None singleton.

    `push_to_hub: bool`
    :   The type of the None singleton.

    `remove_unused_columns: bool`
    :   The type of the None singleton.

    `report_to: str | list[str] | None`
    :   The type of the None singleton.

    `restore_callback_states_from_checkpoint: bool`
    :   The type of the None singleton.

    `resume_from_checkpoint: str | None`
    :   The type of the None singleton.

    `run_name: str | None`
    :   The type of the None singleton.

    `save_on_each_node: bool`
    :   The type of the None singleton.

    `save_only_model: bool`
    :   The type of the None singleton.

    `save_steps: float`
    :   The type of the None singleton.

    `save_strategy: transformers.trainer_utils.SaveStrategy | str`
    :   The type of the None singleton.

    `save_total_limit: int | None`
    :   The type of the None singleton.

    `seed: int`
    :   The type of the None singleton.

    `should_log`
    :   Whether or not the current process should produce log.

    `should_save`
    :   Whether or not the current process should write to disk, e.g., to save models and checkpoints.

    `skip_memory_metrics: bool`
    :   The type of the None singleton.

    `tf32: bool | None`
    :   The type of the None singleton.

    `torch_compile: bool`
    :   The type of the None singleton.

    `torch_compile_backend: str | None`
    :   The type of the None singleton.

    `torch_compile_mode: str | None`
    :   The type of the None singleton.

    `torch_empty_cache_steps: int | None`
    :   The type of the None singleton.

    `trackio_bucket_id: str | None`
    :   The type of the None singleton.

    `trackio_space_id: str | None`
    :   The type of the None singleton.

    `trackio_static_space_id: str | Literal[False] | None`
    :   The type of the None singleton.

    `train_batch_size: int`
    :   The actual batch size for training.

    `train_sampling_strategy: str`
    :   The type of the None singleton.

    `use_cache: bool`
    :   The type of the None singleton.

    `use_cpu: bool`
    :   The type of the None singleton.

    `use_liger_kernel: bool`
    :   The type of the None singleton.

    `warmup_ratio: float | None`
    :   The type of the None singleton.

    `warmup_steps: float`
    :   The type of the None singleton.

    `weight_decay: float`
    :   The type of the None singleton.

    `world_size`
    :   The number of processes used in parallel.

    ### Methods

    `get_process_log_level(self)`
    :   Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.
        
        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.
        
        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.
        
        The choice between the main and replica process settings is made according to the return value of `should_log`.

    `get_warmup_steps(self, num_training_steps: int)`
    :   Get number of steps used for a linear warmup.

    `main_process_first(self, local=True, desc='work')`
    :   A context manager for torch distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.
        
        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.
        
        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

    `set_dataloader(self, train_batch_size: int = 8, eval_batch_size: int = 8, drop_last: bool = False, num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False, prefetch_factor: int | None = None, auto_find_batch_size: bool = False, ignore_data_skip: bool = False, sampler_seed: int | None = None)`
    :   A method that regroups all arguments linked to the dataloaders creation.
        
        Args:
            drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not.
            num_workers (`int`, *optional*, defaults to 0):
                Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                the main process.
            pin_memory (`bool`, *optional*, defaults to `True`):
                Whether you want to pin memory in data loaders or not. Will default to `True`.
            persistent_workers (`bool`, *optional*, defaults to `False`):
                If True, the data loader will not shut down the worker processes after a dataset has been consumed
                once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
                but will increase RAM usage. Will default to `False`.
            prefetch_factor (`int`, *optional*):
                Number of batches loaded in advance by each worker.
                2 means there will be a total of 2 * num_workers batches prefetched across all workers.
            auto_find_batch_size (`bool`, *optional*, defaults to `False`)
                Whether to find a batch size that will fit into memory automatically through exponential decay,
                avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
            ignore_data_skip (`bool`, *optional*, defaults to `False`):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have.
            sampler_seed (`int`, *optional*):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
        >>> args.per_device_train_batch_size
        16
        ```

    `set_evaluate(self, strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', steps: int = 500, batch_size: int = 8, accumulation_steps: int | None = None, delay: float | None = None, loss_only: bool = False)`
    :   A method that regroups all arguments linked to evaluation.
        
        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:
        
                    - `"no"`: No evaluation is done during training.
                    - `"steps"`: Evaluation is done (and logged) every `steps`.
                    - `"epoch"`: Evaluation is done at the end of each epoch.
        
                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for evaluation.
            accumulation_steps (`int`, *optional*):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
                If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
                but requires more memory).
            delay (`float`, *optional*):
                Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                eval_strategy.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```

    `set_logging(self, strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', steps: int = 500, report_to: str | list[str] = 'none', level: str = 'passive', first_step: bool = False, nan_inf_filter: bool = False, on_each_node: bool = False, replica_level: str = 'passive')`
    :   A method that regroups all arguments linked to logging.
        
        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The logging strategy to adopt during training. Possible values are:
        
                    - `"no"`: No logging is done during training.
                    - `"epoch"`: Logging is done at the end of each epoch.
                    - `"steps"`: Logging is done every `logging_steps`.
        
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two logs if `strategy="steps"`.
            level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on the main process. Possible choices are the log levels as strings: `"debug"`,
                `"info"`, `"warning"`, `"error"` and `"critical"`, plus a `"passive"` level which doesn't set anything
                and lets the application set the level.
            report_to (`str` or `list[str]`, *optional*, defaults to `"none"`):
                The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
                `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`,
                `"swanlab"`, `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations
                installed, `"none"` for no integrations.
            first_step (`bool`, *optional*, defaults to `False`):
                Whether to log and evaluate the first `global_step` or not.
            nan_inf_filter (`bool`, *optional*, defaults to `True`):
                Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is
                `nan` or `inf` is filtered and the average loss of the current logging window is taken instead.
        
                <Tip>
        
                `nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
                gradient is computed or applied to the model.
        
                </Tip>
        
            on_each_node (`bool`, *optional*, defaults to `True`):
                In multinode distributed training, whether to log using `log_level` once per node, or only on the main
                node.
            replica_level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on replicas. Same choices as `log_level`
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_logging(strategy="steps", steps=100)
        >>> args.logging_steps
        100
        ```

    `set_lr_scheduler(self, name: transformers.trainer_utils.SchedulerType | str = 'linear', num_epochs: float = 3.0, max_steps: int = -1, warmup_steps: float = 0, warmup_ratio: float | None = None)`
    :   A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.
        
        Args:
            name (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
                The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
            num_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            warmup_steps (`float`, *optional*, defaults to 0):
                Number of steps used for a linear warmup from 0 to `learning_rate`.  Should be an integer or a float in range `[0,1)`.
                If smaller than 1, will be interpreted as ratio of steps used for a linear warmup from 0 to `learning_rate`.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_steps=0.05)
        >>> args.warmup_steps
        0.05
        ```

    `set_optimizer(self, name: transformers.training_args.OptimizerNames | str = 'adamw_torch', learning_rate: float = 5e-05, weight_decay: float = 0, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08, args: str | None = None)`
    :   A method that regroups all arguments linked to the optimizer and its hyperparameters.
        
        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"`):
                The optimizer to use: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`,
                `"adamw_anyprecision"` or `"adafactor"`.
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
            beta1 (`float`, *optional*, defaults to 0.9):
                The beta1 hyperparameter for the adam optimizer or its variants.
            beta2 (`float`, *optional*, defaults to 0.999):
                The beta2 hyperparameter for the adam optimizer or its variants.
            epsilon (`float`, *optional*, defaults to 1e-8):
                The epsilon hyperparameter for the adam optimizer or its variants.
            args (`str`, *optional*):
                Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
                `optim="adamw_anyprecision"`).
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
        >>> args.optim
        'adamw_torch'
        ```

    `set_push_to_hub(self, model_id: str, strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', token: str | None = None, private_repo: bool | None = None, always_push: bool = False, revision: str | None = None)`
    :   A method that regroups all arguments linked to synchronizing checkpoints with the Hub.
        
        <Tip>
        
        Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git
        directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is
        triggered (depending on your `self.save_strategy`). Calling [`~Trainer.save_model`] will also trigger a push.
        
        </Tip>
        
        Args:
            model_id (`str`):
                The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
                which case the model will be pushed in your namespace. Otherwise it should be the whole repository
                name, for instance `"user_name/model"`, which allows you to push to an organization you are a member of
                with `"organization_name/model"`.
            strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
                Defines the scope of what is pushed to the Hub and when. Possible values are:
        
                - `"end"`: push the model, its configuration, the processing_class e.g. tokenizer (if passed along to the [`Trainer`]) and a
                draft of a model card when the [`~Trainer.save_model`] method is called.
                - `"every_save"`: push the model, its configuration, the processing_class e.g. tokenizer (if passed along to the [`Trainer`])
                  and
                a draft of a model card each time there is a model save. The pushes are asynchronous to not block
                training, and in case the save are very frequent, a new push is only attempted if the previous one is
                finished. A last push is made with the final model at the end of training.
                - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
                last-checkpoint, allowing you to resume training easily with
                `trainer.train(resume_from_checkpoint="last-checkpoint")`.
                - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the
                  output
                folder (so you will get one checkpoint folder per folder in your final repository)
        
            token (`str`, *optional*):
                The token to use to push the model to the Hub. Will default to the token in the cache folder obtained
                with `hf auth login`.
            private_repo (`bool`, *optional*, defaults to `False`):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            always_push (`bool`, *optional*, defaults to `False`):
                Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not
                finished.
            revision (`str`, *optional*):
                The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_push_to_hub("me/awesome-model")
        >>> args.hub_model_id
        'me/awesome-model'
        ```

    `set_save(self, strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', steps: int = 500, total_limit: int | None = None, on_each_node: bool = False)`
    :   A method that regroups all arguments linked to checkpoint saving.
        
        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The checkpoint save strategy to adopt during training. Possible values are:
        
                    - `"no"`: No save is done during training.
                    - `"epoch"`: Save is done at the end of each epoch.
                    - `"steps"`: Save is done every `save_steps`.
        
            steps (`int`, *optional*, defaults to 500):
                Number of updates steps before two checkpoint saves if `strategy="steps"`.
            total_limit (`int`, *optional*):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`.
            on_each_node (`bool`, *optional*, defaults to `False`):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.
        
                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```

    `set_testing(self, batch_size: int = 8, loss_only: bool = False)`
    :   A method that regroups all basic arguments linked to testing on a held-out dataset.
        
        <Tip>
        
        Calling this method will automatically set `self.do_predict` to `True`.
        
        </Tip>
        
        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```

    `set_training(self, learning_rate: float = 5e-05, batch_size: int = 8, weight_decay: float = 0, num_epochs: float = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1, seed: int = 42, gradient_checkpointing: bool = False)`
    :   A method that regroups all basic arguments linked to the training.
        
        <Tip>
        
        Calling this method will automatically set `self.do_train` to `True`.
        
        </Tip>
        
        Args:
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate for the optimizer.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for training.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
                optimizer.
            num_train_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            gradient_accumulation_steps (`int`, *optional*, defaults to 1):
                Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        
                <Tip warning={true}>
        
                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
                examples.
        
                </Tip>
        
            seed (`int`, *optional*, defaults to 42):
                Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
                the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized
                parameters.
            gradient_checkpointing (`bool`, *optional*, defaults to `False`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        
        Example:
        
        ```py
        >>> from transformers import TrainingArguments
        
        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
        >>> args.learning_rate
        1e-4
        ```

    `to_dict(self)`
    :   Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.

    `to_json_string(self)`
    :   Serializes this instance to a JSON string.

    `to_sanitized_dict(self) ‑> dict[str, typing.Any]`
    :   Sanitized serialization to use with TensorBoard's hparams

`TritonCosformerAttention(config: Any = None, embed_dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, is_decoder: bool = False, eps: float = 1e-06, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,
    RoPE support, Pretrained Weight Loading, and Masked Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `triton_kernel`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.

    `copy_weights_from_pretrained(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.

`TritonSVDLinearAttention(config: Any = None, embed_dim: int | None = None, num_heads: int | None = None, num_kv_heads: int | None = None, rank: int | None = 64, is_decoder: bool = False, activation_mode: str = 'elu', eps: float = 1e-06, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections,
    RoPE, GQA, and Masked Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `triton_kernel`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.

    `copy_weights_from_pretrained(self, src: Any) ‑> None`
    :   Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers.

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

`UnifiedRoPE(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, max_seq_len: int = 4096, base: int = 10000, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Universal Rotary Position Embedding Attention module.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor`
    :   Forward pass computing RoPE attention, returning (output, attention_weights).

`UniversalAdapter()`
:   Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc.

    ### Static methods

    `get_kv_heads(config: Any) ‑> int`
    :   Return number of key-value heads for GQA/MQA.

    `get_rotary_emb(old_attn: Any) ‑> Any | None`
    :   Return rotary embeddings if present.

    `get_weight_bias(old_attn: Any, proj: str) ‑> tuple`
    :   Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.
        
        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]

`VSpecAttention(config: Any | None = None, rank: int = 64, pretrained_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
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

`VariationalSpectralGate(hidden_dim: int, rank: int)`
:   Input-dependent variational gate over truncated singular values with masked pooling.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`ViGAttention(config: Any = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

`WeightedGroupedQueryAttention(config: Any | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, num_kv_heads: int | None = None, is_decoder: bool | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, **kwargs: Any)`
:   Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance weights,
    robust GQA pooling, LoRA adapters, RoPE support with fallback, and causal masking.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
    :

    `out`
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

    `copy_weights(self, src: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

`XAttention(config: Any, rank: int | None = None, block_size: int = 32, sparsity: float = 0.5, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `k_proj: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `q_proj: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `v_proj: LoRALinear`
    :

`XAttentionScorer()`
:   

    ### Static methods

    `score_blocks(Q: torch.Tensor, K: torch.Tensor, block_size: int, stride: int = 2) ‑> torch.Tensor`
    :

`YourCustomAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Template attention layer with optional low-rank pooling modulation.
    
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

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `v_lin`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2, Qwen/LLaMA.

`ZeroSAttention(config: Any, rank: int | None = None, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

`SDPBackend(...)`
:   An enum-like class that contains the different backends for scaled dot product attention.
    
    ... warning:: This class is in beta and subject to change.
    
    This backend class is designed to be used with the sdpa_kernel context manager.See :func: torch.nn.attention.sdpa_kernel for more details.
    
    Members:
    
      ERROR
    
      MATH
    
      FLASH_ATTENTION
    
      EFFICIENT_ATTENTION
    
      CUDNN_ATTENTION
    
      OVERRIDEABLE
    
    __init__(self: torch._C._SDPBackend, value: typing.SupportsInt) -> None

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Class variables

    `CUDNN_ATTENTION`
    :   The type of the None singleton.

    `EFFICIENT_ATTENTION`
    :   The type of the None singleton.

    `ERROR`
    :   The type of the None singleton.

    `FLASH_ATTENTION`
    :   The type of the None singleton.

    `MATH`
    :   The type of the None singleton.

    `OVERRIDEABLE`
    :   The type of the None singleton.

    ### Instance variables

    `name: str`
    :

    `value`
    :