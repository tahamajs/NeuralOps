# Distributed Training & MLOps Infrastructure Guide (v1.3.0)

NeuralOps supports multi-GPU distributed fine-tuning, knowledge distillation, and recovery campaigns across Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and DeepSpeed, paired with real-time MLOps telemetry.

---

## 1. MLOps Experiment Tracking Architecture

NeuralOps abstracts telemetry logging into a unified backend interface (`BaseTracker`) that streams training loss, evaluation accuracy, parameter counts, compression ratios, and VRAM consumption in real-time.

Supported backends:
- **Weights & Biases (`wandb`)**: Live dashboards, metric charts, parameter configurations.
- **MLflow (`mlflow`)**: Experiment versioning, artifact tracking, run tagging.
- **TensorBoard (`tensorboard`)**: Local visualization without network access.
- **Multiplexed (`all`)**: Simultaneous dual-broadcast to WandB, MLflow, and TensorBoard.
- **Local Disk (`none`)**: Deterministic append-only JSONL files saved to `results/runs/<run_id>.json`.

### CLI Invocation
```bash
# Stream run metrics to Weights & Biases
python -m neuralops.benchmarks run \
  --model roberta-base \
  --dataset sst2 \
  --module cosformer \
  --tracker wandb \
  --project-name "subquadratic-attention-benchmark" \
  --run-name "roberta-cosformer-sst2"

# Track compression sweep across both MLflow and TensorBoard
python -m neuralops.benchmarks run \
  --model distilbert-base-uncased \
  --dataset mnli \
  --module direct_svd \
  --rank 32 \
  --tracker all \
  --project-name "svd-compression-campaign"
```

---

## 2. Distributed Execution Backends

NeuralOps incorporates `accelerate.Accelerator` inside its training execution loop (`neuralops.benchmarks.core.training.finetune_classification` and `neuralops.benchmarks.executor.execute_benchmark`), enabling zero-code multi-GPU scaling.

### Supported Distributed Strategies:
1. **DDP (DistributedDataParallel)**: Standard multi-GPU data parallelism across single or multi-node clusters.
2. **FSDP (Fully Sharded Data Parallel)**: Model, optimizer, and gradient sharding for large 7B+ decoder backbones.
3. **DeepSpeed (ZeRO-Stage 2 & ZeRO-Stage 3)**: CPU offloading and partitioned parameter updates.

### Multi-GPU Launch Syntax:
```bash
# Launch multi-GPU sweep using Accelerate
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m neuralops.benchmarks run \
  --model meta-llama/Llama-3.2-1B \
  --dataset wikitext2 \
  --module gla_attention \
  --distributed-backend ddp \
  --mixed-precision bf16 \
  --batch-size 8

# Using torchrun directly
torchrun --nproc_per_node=4 -m neuralops.benchmarks run \
  --model roberta-base \
  --dataset sst2 \
  --module cosformer \
  --distributed-backend ddp \
  --mixed-precision fp16
```

---

## 3. Mixed Precision Configuration

NeuralOps implements transparent Automatic Mixed Precision (AMP) with dynamic gradient scaling:

- `--mixed-precision no`: Pure 32-bit float execution (default for CPU/MPS or numerical reference checks).
- `--mixed-precision fp16`: 16-bit float training using `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` and `torch.cuda.amp.GradScaler` to prevent underflow.
- `--mixed-precision bf16`: Brain float 16 for Ampere, Ada Lovelace, and Hopper architectures, providing 8 bits of dynamic range without requiring a gradient scaler.

---

## 4. Multi-Server GPU Cluster Management

When executing large matrix sweeps across remote GPU servers (e.g., Server 1 / `tahoora` and Server 2):

1. **VRAM Safety Checks**: Pre-run health checks automatically query `torch.cuda.mem_get_info()`. If available VRAM is below 500 MB, the run logs a warning to avoid disrupting concurrent tmux sessions.
2. **Deterministic Process Isolation**: Each benchmark process creates an isolated temporary directory and executes inside a threaded worker with a hard wall-clock timeout (`--timeout-s 1800`), ensuring that hung kernels do not freeze GPU nodes.
3. **Resumable State**: Results are flushed immediately to disk (`results/runs/`) after each run. If a job is preempted or terminated, executing with `--resume` skips already-completed module/dataset tuples automatically.
