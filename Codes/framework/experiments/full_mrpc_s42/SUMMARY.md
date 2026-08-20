# Run Summary: `full_mrpc_s42`
> Generated: 2026-08-20 05:59:33

## Configuration
```
  attn_implementation: full
  attn_type: full
  batch_size: 16
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
  lr: 3e-05
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
  task: mrpc
  trainer_type: hf
  use_gate: True
  use_lion: False
```

## Results

| Metric | Value |
|:---|:---|
| eval_loss | 0.508335 |
| eval_accuracy | 0.705882 |
| eval_f1 | 0.818182 |
| eval_accuracy_f1 | 0.762032 |
| eval_runtime | 0.405700 |
| eval_samples_per_second | 1005.677000 |
| eval_steps_per_second | 32.044000 |
| epoch | 0.500000 |
| gpu_memory_allocated | 803.860840 |
| gpu_memory_reserved | 1472.000000 |
| elapsed_time | 12.234634 |

**Total runtime:** 12s

**Output dir:** `experiments\full_mrpc_s42`