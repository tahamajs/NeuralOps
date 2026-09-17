# Model Cards

## Purpose

Document every published checkpoint.

## Template

```yaml
model:
  name: <module>-<backbone>
  version: v1.0
  base_model: <backbone>-base
  compression_method: <module>
  rank: 64
  lora_rank: 32

training:
  dataset: ag_news
  recovery_steps: 100
  batch_size: 32
  optimizer: adamw
  learning_rate: 2.0e-4

evaluation:
  accuracy: 72.4
  f1_macro: 0.71
  ece: 0.03
  per_class: [0.70, 0.74, 0.72, 0.73]

limitations:
  - Trained only on AG News
  - Not validated on >512 token sequences
  - Base weights frozen; only LoRA recovered

ethics:
  - English-only
  - News domain only
  - No PII seen during training
```

## Output

- `papers/modules/<module>/model_cards/<backbone>.yaml`
