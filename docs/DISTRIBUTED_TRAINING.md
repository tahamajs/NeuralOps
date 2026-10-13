# Distributed Training & MLOps Guide

NeuralOps provides multi-GPU distributed fine-tuning, recovery, and knowledge distillation along with centralized experiment tracking.

---

## 1. MLOps Experiment Tracking

NeuralOps natively integrates with enterprise experiment trackers via the `--tracker` flag:

- **Weights & Biases (`wandb`)**
- **MLflow (`mlflow`)**
- **TensorBoard (`tensorboard`)**
- **Multiplexed (`all`)**
- **Local disk JSONL/CSV (`none` / default)**

### CLI Usage:
```bash
# Track run in Weights & Biases
neuralops run \
  --model roberta-base \
  --dataset sst2 \
  --module cosformer \
  --tracker wandb \
  --project-name "production-attention-eval" \
  --run-name "cosformer-sst2-run1"

# Track across both WandB and MLflow
neuralops run \
  --model distilbert-base-uncased \
  --dataset mnli \
  --module adasvd \
  --rank 32 \
  --tracker all \
  --project-name "svd-compression-benchmark"
```

---

## 2. Distributed Training (DDP, FSDP, DeepSpeed)

For large-scale teacher-student knowledge distillation or multi-GPU recovery:

```bash
# Distributed Data Parallel (DDP) with mixed precision bf16
neuralops run \
  --model meta-llama/Llama-3.2-1B \
  --dataset wikitext2 \
  --module gdn_attention \
  --distributed-backend ddp \
  --mixed-precision bf16 \
  --batch-size 8

# Multi-GPU execution via Accelerate CLI
accelerate launch -m neuralops.benchmarks.cli run \
  --model roberta-base \
  --dataset sst2 \
  --all-attn \
  --mixed-precision fp16
```

---

## 3. Knowledge Distillation (KD) Configuration

Compress large models while recovering accuracy using student-teacher distillation:

```bash
neuralops run \
  --model distilbert-base-uncased \
  --dataset sst2 \
  --module adasvd \
  --rank 32 \
  --training-type kd \
  --kd-teacher bert-base-uncased \
  --kd-temperature 2.0 \
  --kd-alpha 0.5 \
  --tracker wandb
```
