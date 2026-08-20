# Run Summary: `svd_sst2_s42`
> Generated: 2026-08-20 05:29:18

## Configuration
```
  attn_implementation: full
  attn_type: svd
  batch_size: 32
  compression_rank: 64
  compressor: None
  config: None
  dislora_beta: 0.5
  dislora_capacity_factor: 1.0
  early_stopping_patience: None
  epochs: 0.5
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
  method: baseline
  mixout_p: 0.9
  model_name: distilbert-base-uncased
  num_landmarks: 16
  num_seeds: 1
  output_dir: ./experiments
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
| eval_loss | 0.429034 |
| eval_accuracy | 0.806193 |
| eval_f1_macro | 0.806101 |
| eval_f1_weighted | 0.806023 |
| eval_runtime | 45.508700 |
| eval_samples_per_second | 19.161000 |
| eval_steps_per_second | 0.308000 |
| epoch | 0.500238 |
| gpu_memory_allocated | 822.360840 |
| gpu_memory_reserved | 2172.000000 |
| elapsed_time | 1765.191182 |

**Total runtime:** 28m 39s

**Output dir:** `experiments\svd_sst2_s42`