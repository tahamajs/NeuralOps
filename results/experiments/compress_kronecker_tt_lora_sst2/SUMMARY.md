# Run Summary: `compress_kronecker_tt_lora_sst2`
> Generated: 2026-08-23 05:29:30

## Configuration
```
  attn_implementation: full
  attn_type: full
  batch_size: 64
  compression_rank: 64
  compressor: kronecker_tt_lora
  config: None
  dislora_beta: 0.5
  dislora_capacity_factor: 1.0
  early_stopping_patience: None
  epochs: 3.0
  exp_name: None
  feature_dim: 128
  fisher_num_batches: 20
  fisher_sample_size: 200
  fisher_scale_max: 2.0
  gelora_margin: 1.0
  gelora_max_rank: 32
  gelora_min_rank: 2
  glue_base_path: None
  grad_accum: 1
  gradient_checkpointing: False
  is_compression: True
  k_early: 32
  k_late: 12
  k_mid: 16
  llrd_decay: 0.95
  log_base_dir: ./logs
  log_dir: ./logs
  lora_alpha: 16
  lora_dropout: 0.1
  lora_r: 8
  lr: 0.0002
  max_grad_norm: 1.0
  max_len: 128
  max_train_samples: 1000
  method: baseline
  mixout_p: 0.9
  model_name: distilbert-base-uncased
  num_landmarks: 16
  num_seeds: 1
  output_dir: C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments
  progressive: True
  r3f_epsilon: 1e-05
  rdrop_alpha: 5.0
  run_id: None
  save_model: False
  seed: 42
  task: sst2
  trainer_type: hf
  use_gate: True
  use_lion: False
```

## Results

| Metric | Value |
|:---|:---|
| compressor | kronecker_tt_lora |
| original_size_mb | 255.410000 |
| compressed_size_mb | 260.950000 |
| reduction_ratio | 0.979000 |
| compression_rank | 64 |

**Total runtime:** 0s

**Output dir:** `C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments\compress_kronecker_tt_lora_sst2`