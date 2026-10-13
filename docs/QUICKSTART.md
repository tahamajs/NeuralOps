# NeuralOps v1.3.0 Quickstart Guide

`NeuralOps` is a high-performance framework for subquadratic attention mechanisms, model compression engines, and accelerated Triton kernel operations. It integrates seamlessly with PyTorch and HuggingFace Transformers.

---

## 1. Installation

Install NeuralOps via pip or build directly from source:

```bash
# Core CPU installation
pip install neuralops

# With CUDA acceleration & Triton kernels
pip install neuralops[cuda]

# Production export tools (Safetensors, ONNX, ONNXRuntime)
pip install neuralops[export]

# Full suite including MLOps (WandB, MLflow) and visualization
pip install neuralops[all]
```

To verify your installation:

```bash
python -c "import neuralops; print(neuralops.info())"
```

---

## 2. Drop-in Attention Surgery (1 Line of Code)

Transform any standard Transformer (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Mistral, Qwen2, T5) into an efficient subquadratic model with `neuralops.replace_attention`:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

# 1. Load your pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Drop-in replace attention with Cosformer (O(N) linear complexity)
model = neuralops.replace_attention(model, "cosformer")

# 3. Model is immediately ready for inference or fine-tuning
input_ids = torch.randint(0, 50257, (2, 256))
outputs = model(input_ids)
print("Logits shape:", outputs.logits.shape)
```

NeuralOps automatically:
- Preserves rotary positional embeddings (RoPE) and Grouped-Query Attention (GQA).
- Transplants original weights when compatible (`copy_weights=True`).
- Attaches surgery metadata onto `model._neuralops_metadata` for export and reloading.

---

## 3. Deep Model Compression

Compress weight matrices using singular value decomposition (SVD), low-rank factorization, or pruning:

```python
import neuralops
from transformers import AutoModelForSequenceClassification

# Load sequence classification model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Compress using AdaSVD with rank 32
compressed_model = neuralops.compress(model, method="adasvd", rank=32)

# Inspect compression metadata
print("Surgery metadata:", compressed_model._neuralops_metadata)
```

---

## 4. Production Export & Round-Trip Reloading

Export your modified model to production formats (`safetensors`, `onnx`, `torchscript`) with automatic architecture configuration persistence:

```python
import neuralops

# Export model with embedded neuralops_config.json
export_dir = neuralops.export_model(
    model=compressed_model,
    output_dir="./exports/distilbert-adasvd",
    format="safetensors",
)

# Reload model from disk without manual configuration typing
reloaded_model = neuralops.load_model(export_dir)
reloaded_model.eval()
```

---

## 5. Benchmarking & CLI Execution

NeuralOps provides an industrial-grade benchmark orchestrator with automated OOM recovery, timeout guards, and metric persistence:

```bash
# Run a single attention benchmark on SST-2
python -m neuralops.benchmarks run \
    --module cosformer \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --epochs 3 \
    --batch-size 32

# Run an SVD compressor benchmark with recovery fine-tuning
python -m neuralops.benchmarks run \
    --module direct_svd \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64 \
    --training-type recovery

# Verify environment, registries, and Triton kernels
python -m neuralops.benchmarks verify

# Display tabular summaries of all benchmark runs
python -m neuralops.benchmarks results --summary
```

---

## 6. Training, Knowledge Distillation & MLOps

Fine-tune student models or train with teacher-student Knowledge Distillation (KD) using mixed precision (`fp16`/`bf16`) and telemetry:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import neuralops

# Create synthetic dummy data for demonstration
train_data = TensorDataset(
    torch.randint(0, 1000, (64, 128)),
    torch.ones(64, 128, dtype=torch.long),
    torch.randint(0, 2, (64,)),
)
train_loader = DataLoader(train_data, batch_size=16)

# Fine-tune model
metrics = neuralops.train(
    model=model,
    train_loader=train_loader,
    epochs=1,
    lr=2e-5,
    mixed_precision="fp16" if torch.cuda.is_available() else "no",
)
print("Training results:", metrics)
```

---

## 7. Serving with vLLM and TGI

Prepare your customized models for high-throughput serving:

```python
from neuralops.export.serving import VLLMAdapter, TGIAdapter

# Wrap attention module with paged KV cache handling for vLLM
paged_module = VLLMAdapter.wrap_forward_for_paged_kv(model.transformer.h[0].attn)

# Prepare model and generate serving config for Text Generation Inference (TGI)
tgi_ready_model = TGIAdapter.prepare_for_tgi(model, output_dir="./serving_tgi")
```
