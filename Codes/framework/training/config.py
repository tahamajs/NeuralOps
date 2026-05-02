@dataclass
class CompressionConfig:
    # ... existing attention fields ...
    # ---------- Method parameters ----------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    rdrop_alpha: float = 5.0
    r3f_epsilon: float = 1e-5
    llrd_decay: float = 0.95
    mixout_p: float = 0.9
    use_lion: bool = False
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32
    fisher_scale_max: float = 2.0
    gradient_checkpointing: bool = False
    early_stopping_patience: Optional[int] = None
    max_grad_norm: float = 1.0