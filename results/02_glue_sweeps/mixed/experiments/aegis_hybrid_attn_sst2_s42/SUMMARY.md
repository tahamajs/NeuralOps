# Run Summary: `aegis_hybrid_attn_sst2_s42`
> Generated: 2026-08-22 04:13:37

## Configuration
```
  attn_implementation: full
  attn_type: aegis_hybrid_attn
  batch_size: 32
  compression_rank: 64
  compressor: None
  config: None
  dislora_beta: 0.5
  dislora_capacity_factor: 1.0
  early_stopping_patience: None
  epochs: 1.0
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
  k_early: 32
  k_late: 12
  k_mid: 16
  llrd_decay: 0.95
  log_base_dir: ./logs
  log_dir: ./logs
  lora_alpha: 16
  lora_dropout: 0.1
  lora_r: 8
  lr: 5e-05
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
| eval_loss | 0.702719 |
| eval_accuracy | 0.509174 |
| eval_f1_macro | 0.337386 |
| eval_f1_weighted | 0.343577 |
| eval_runtime | 1.377600 |
| eval_samples_per_second | 633.004000 |
| eval_steps_per_second | 10.163000 |
| epoch | 1.000000 |
| gpu_memory_allocated | 650.151855 |
| gpu_memory_reserved | 5928.000000 |
| elapsed_time | 8.190490 |

**Total runtime:** 6s

**Output dir:** `C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments\aegis_hybrid_attn_sst2_s42`