# Distributed Training, Knowledge Distillation & MLOps Guide (v1.3.0)

NeuralOps delivers a unified distributed training architecture designed to train, recover, and distill subquadratic attention architectures and compressed transformer models across multi-GPU and multi-node clusters.

---

## 1. Overview of the Training Engine

Replacing full multi-head attention with low-rank or linear attention mechanisms typically leads to initial performance drops (referred to as initialization shock). To recover or exceed the teacher model's original accuracy, NeuralOps provides three coordinated training strategies:

1. **Recovery Fine-Tuning**: Standard task fine-tuning on downstream datasets (e.g., GLUE SST-2, MNLI, QQP) using AdamW with cosine learning rate scheduling and early NaN stopping.
2. **Knowledge Distillation (KD)**: Soft-target distillation where student models learn to match dense teacher output distributions via temperature-scaled Kullback-Leibler (KL) divergence and cross-entropy loss.
3. **LoRA Recovery**: Freezes factorized base weights and trains lightweight Low-Rank Adapters ($r=16, \alpha=32$) to adapt to the compressed manifold rapidly.

---

## 2. Multi-GPU Distributed Training via Accelerate

NeuralOps integrates directly with Hugging Face `accelerate`, abstracting away distributed boilerplate and supporting Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and DeepSpeed.

### Automatic Fallback Behavior
If the `--distributed-backend` flag is set to `accelerate`, NeuralOps checks for the package's presence. If `accelerate` is missing, NeuralOps logs a warning and transparently falls back to single-device execution without crashing:
```text
[WARNING] Distributed backend 'accelerate' requested but 'accelerate' package is not installed. Falling back to single-device execution.
```

When present, `Accelerator` automatically wraps:
- Model (`model = accelerator.prepare(model)`)
- Optimizer (`optimizer = accelerator.prepare(optimizer)`)
- DataLoaders (`train_loader = accelerator.prepare(train_loader)`)

### CLI Distributed Launch
```bash
# Launch a 4-GPU distributed sweep using Accelerate and DDP
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m neuralops.benchmarks run \
  --model meta-llama/Llama-3.2-1B \
  --dataset wikitext2 \
  --module gla_attention \
  --distributed-backend ddp \
  --batch-size 8 \
  --epochs 3
```

### Programmatic Python Usage
```python
import neuralops
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# 1. Create model and perform attention surgery
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = neuralops.replace_attention(model, "cosformer")

# 2. Prepare synthetic dataset
dataset = TensorDataset(
    torch.randint(0, 1000, (256, 128)),
    torch.randint(0, 2, (256,)),
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Distributed training with automatic accelerator wrapping
metrics = neuralops.train(
    model=model,
    dataset=train_loader,
    epochs=3,
    lr=2e-5,
    mixed_precision="fp16" if torch.cuda.is_available() else "no",
    distributed_backend="accelerate",
)
print("Training metrics:", metrics)
```

---

## 3. Knowledge Distillation (KD) Engine

Knowledge Distillation is critical for achieving high accuracy with subquadratic attention blocks. NeuralOps provides a dedicated KD loss formulation:

$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_{\text{student}}, y_{\text{true}}) + \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_{\text{student}}}{T}\right) \parallel \sigma\left(\frac{z_{\text{teacher}}}{T}\right)\right)$$

Where:
- $T$ is the temperature scaling factor (default: $2.0$).
- $\alpha$ is the balance weight between hard and soft targets (default: $0.5$).

### Python KD Invocation
```python
import neuralops
from transformers import AutoModelForSequenceClassification

# Load dense teacher and initialized student
teacher = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
student = neuralops.replace_attention(student, "cosformer")

# Execute distillation loop
kd_metrics = neuralops.distill(
    student=student,
    teacher=teacher,
    dataset=train_loader,
    epochs=3,
    lr=5e-5,
    temperature=2.0,
    alpha=0.5,
)
print("Distillation metrics:", kd_metrics)
```

### CLI Distillation Sweep
```bash
# Run distillation across all delta-rule modules
neuralops run \
  --all-delta \
  --training-type kd \
  --kd-teacher distilbert-base-uncased-finetuned-sst-2-english \
  --dataset sst2 \
  --epochs 3
```

---

## 4. MLOps Telemetry & Logging Integration

NeuralOps supports real-time step-level and epoch-level telemetry streaming to leading MLOps backends:

| Backend | Flag | Features |
| :--- | :--- | :--- |
| **Weights & Biases** | `--tracker wandb` | Live loss curves, VRAM tracking, parameter histograms, artifacts |
| **MLflow** | `--tracker mlflow` | Metric history, run parameters, model registry tags |
| **TensorBoard** | `--tracker tensorboard` | Local scalar graphs without network requirements |
| **Multiplexed** | `--tracker all` | Broadcasts simultaneously to WandB, MLflow, and TensorBoard |
| **Local Disk** | `--tracker none` | Deterministic append-only JSONL files saved to `results/runs/` |

### Setting up Weights & Biases (WandB)
```bash
# Stream benchmark runs directly to WandB project
neuralops run \
  --module cosformer \
  --model distilbert-base-uncased \
  --dataset sst2 \
  --tracker wandb \
  --project-name "subquadratic-attention-benchmarks" \
  --run-name "distilbert-cosformer-run-1"
```

In Python:
```python
metrics = neuralops.train(
    model=model,
    dataset=train_loader,
    epochs=3,
    tracker="wandb",
    project_name="neuralops-experiments",
)
```

---

## 5. Fault Tolerance & Automatic Recovery

Long-running sweeps across multi-GPU nodes incorporate automatic failure mitigation:
- **NaN/Inf Watchdog**: Automatically aborts diverging runs and restores the best epoch checkpoint.
- **OOM Protection**: Detects GPU Out-Of-Memory errors, flushes the CUDA cache with `torch.cuda.empty_cache()`, and logs hardware status.
- **Resumable Executions**: Run directories persist progress. Interrupted sweeps can be resumed instantly with `neuralops resume --output-dir results/`.
