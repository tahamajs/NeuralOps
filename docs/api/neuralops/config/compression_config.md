Module neuralops.config.compression_config
==========================================

Classes
-------

`CompressionConfig(attn_type: str = 'full', dropout: float = 0.1, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, rdrop_alpha: float = 5.0, r3f_epsilon: float = 1e-05, llrd_decay: float = 0.95, mixout_p: float = 0.9, dislora_beta: float = 0.5, dislora_capacity_factor: float = 1.0, fisher_num_batches: int = 20, fisher_sample_size: int = 200, fisher_scale_max: float = 2.0, gelora_margin: float = 1.0, gelora_min_rank: int = 2, gelora_max_rank: int = 32, use_lion: bool = False, gradient_checkpointing: bool = False, early_stopping_patience: int | None = None, max_grad_norm: float = 1.0, rank: int = 32, svd_rank: int = 32, svd_energy: float = 0.95, block_size: int = 64, local_window: int = 64, top_k_inter: int = 4, num_landmarks: int = 16, feature_dim: int = 128, rff_features: int = 128, k_early: int = 32, k_mid: int = 16, k_late: int = 12, use_gate: bool = True, progressive: bool = True, max_seq_len: int = 512, low_rank_cache_size: int = 16, cur_rank: int = 16, lsh_rounds: int = 4, lsh_bucket_size: int = 32, window_size: int = 128, compression_rank: int = 64, use_adaptive_gate: bool = True, gate_bias: float = 2.0, softpick_temperature: float = 1.0, rectify_interval: int = 4, aks_enabled: bool = True, rel_pos_max_distance: int = 128, use_rel_pos_bias: bool = True, adaptive_span: bool = True, max_span: float = 512.0, num_kernel_heads: int = 8, num_full_heads: int = 4, soa_rank: int = 32, svd_rank_qkv: int = 32, svd_rank_out: int = 32, min_rank: int = 4, max_rank: int = 64, num_local_heads: int = 8, num_global_heads: int = 4, num_experts: int = 8, top_k: int = 2, ratio: int = 9, n_heads: int | None = None, n_layers: int | None = None, use_full_attention: bool = False, use_fp8: bool = False)`
:   Unified configuration for attention injection and compression.

    ### Instance variables

    `adaptive_span: bool`
    :   The type of the None singleton.

    `aks_enabled: bool`
    :   The type of the None singleton.

    `attn_type: str`
    :   The type of the None singleton.

    `block_size: int`
    :   The type of the None singleton.

    `compression_rank: int`
    :   The type of the None singleton.

    `cur_rank: int`
    :   The type of the None singleton.

    `dislora_beta: float`
    :   The type of the None singleton.

    `dislora_capacity_factor: float`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `early_stopping_patience: int | None`
    :   The type of the None singleton.

    `feature_dim: int`
    :   The type of the None singleton.

    `fisher_num_batches: int`
    :   The type of the None singleton.

    `fisher_sample_size: int`
    :   The type of the None singleton.

    `fisher_scale_max: float`
    :   The type of the None singleton.

    `gate_bias: float`
    :   The type of the None singleton.

    `gelora_margin: float`
    :   The type of the None singleton.

    `gelora_max_rank: int`
    :   The type of the None singleton.

    `gelora_min_rank: int`
    :   The type of the None singleton.

    `gradient_checkpointing: bool`
    :   The type of the None singleton.

    `k_early: int`
    :   The type of the None singleton.

    `k_late: int`
    :   The type of the None singleton.

    `k_mid: int`
    :   The type of the None singleton.

    `llrd_decay: float`
    :   The type of the None singleton.

    `local_window: int`
    :   The type of the None singleton.

    `lora_alpha: int`
    :   The type of the None singleton.

    `lora_dropout: float`
    :   The type of the None singleton.

    `lora_r: int`
    :   The type of the None singleton.

    `low_rank_cache_size: int`
    :   The type of the None singleton.

    `lsh_bucket_size: int`
    :   The type of the None singleton.

    `lsh_rounds: int`
    :   The type of the None singleton.

    `max_grad_norm: float`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `max_seq_len: int`
    :   The type of the None singleton.

    `max_span: float`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

    `mixout_p: float`
    :   The type of the None singleton.

    `n_heads: int | None`
    :   The type of the None singleton.

    `n_layers: int | None`
    :   The type of the None singleton.

    `num_experts: int`
    :   The type of the None singleton.

    `num_full_heads: int`
    :   The type of the None singleton.

    `num_global_heads: int`
    :   The type of the None singleton.

    `num_kernel_heads: int`
    :   The type of the None singleton.

    `num_landmarks: int`
    :   The type of the None singleton.

    `num_local_heads: int`
    :   The type of the None singleton.

    `progressive: bool`
    :   The type of the None singleton.

    `r3f_epsilon: float`
    :   The type of the None singleton.

    `rank: int`
    :   The type of the None singleton.

    `ratio: int`
    :   The type of the None singleton.

    `rdrop_alpha: float`
    :   The type of the None singleton.

    `rectify_interval: int`
    :   The type of the None singleton.

    `rel_pos_max_distance: int`
    :   The type of the None singleton.

    `rff_features: int`
    :   The type of the None singleton.

    `soa_rank: int`
    :   The type of the None singleton.

    `softpick_temperature: float`
    :   The type of the None singleton.

    `svd_energy: float`
    :   The type of the None singleton.

    `svd_rank: int`
    :   The type of the None singleton.

    `svd_rank_out: int`
    :   The type of the None singleton.

    `svd_rank_qkv: int`
    :   The type of the None singleton.

    `top_k: int`
    :   The type of the None singleton.

    `top_k_inter: int`
    :   The type of the None singleton.

    `use_adaptive_gate: bool`
    :   The type of the None singleton.

    `use_fp8: bool`
    :   The type of the None singleton.

    `use_full_attention: bool`
    :   The type of the None singleton.

    `use_gate: bool`
    :   The type of the None singleton.

    `use_lion: bool`
    :   The type of the None singleton.

    `use_rel_pos_bias: bool`
    :   The type of the None singleton.

    `window_size: int`
    :   The type of the None singleton.