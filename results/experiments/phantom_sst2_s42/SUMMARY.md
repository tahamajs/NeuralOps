# Run Summary: `phantom_sst2_s42`
> Generated: 2026-08-22 08:35:26

## Configuration
```
  attn_implementation: full
  attn_type: phantom
  batch_size: 64
  compression_rank: 64
  compressor: None
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
  is_compression: False
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
| eval_loss | 0.705271 |
| eval_accuracy | 0.509174 |
| eval_f1_macro | 0.337386 |
| eval_f1_weighted | 0.343577 |
| eval_runtime | 753.019700 |
| eval_samples_per_second | 1.158000 |
| eval_steps_per_second | 0.009000 |
| epoch | 3.000000 |
| gpu_memory_allocated | 629.665527 |
| gpu_memory_reserved | 2240.000000 |
| elapsed_time | 5000.576203 |

**Total runtime:** 1h 10m 47s

**Output dir:** `C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments\phantom_sst2_s42`