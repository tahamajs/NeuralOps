# NeuralOps v1.3.0 Quickstart Guide

`NeuralOps` is an industrial-strength framework for subquadratic attention mechanisms, model compression engines, and accelerated Triton kernel operations. It integrates seamlessly with PyTorch and HuggingFace Transformers with zero code rewrite.

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

# Full suite including MLOps (WandB, MLflow, Accelerate) and serving
pip install neuralops[all]
```

To verify your installation and detected hardware backends:

```bash
python -c "import neuralops; print(neuralops.info())"
```

Output:
```json
{
  "version": "1.3.0",
  "attentions_count": 131,
  "compressors_count": 72,
  "backends": ["cpu", "cuda", "triton"],
  "has_triton": true,
  "torch_version": "2.x",
  "cuda_available": true
}
```

---

## 2. The 19-Function Public API

NeuralOps 1.3.0 exposes 19 canonical, fully-typed functions directly under the top-level `neuralops` namespace:

### Model & Export (4 Functions)
- `neuralops.replace_attention(model, attention="cosformer", rank=None, copy_weights=True, layers=None, **kwargs)`: Replaces multi-head attention blocks with subquadratic variants while automatically recording surgery history in `model._neuralops_metadata`.
- `neuralops.compress(model, method="direct_svd", rank=32, layers=None, **kwargs)`: Compresses linear/attention weights using SVD, low-rank factorization, or pruning.
- `neuralops.export_model(model, output_dir, format="safetensors", **kwargs)`: Exports the modified model to disk, persisting `neuralops_config.json` containing complete surgery metadata alongside weights.
- `neuralops.load_model(export_dir, base_model=None)`: Loads an exported artifact and automatically replays all surgery and compression operations before loading weights.

### Module Discovery (5 Functions)
- `neuralops.list_attentions() -> list[str]`: Returns a sorted list of all 131 registered canonical attention modules.
- `neuralops.list_compressors() -> list[str]`: Returns a sorted list of all 72 registered compression engines.
- `neuralops.get_attention_class(name: str) -> type`: Retrieves the module class from `ATTENTION_REGISTRY`.
- `neuralops.get_compressor_class(name: str) -> type`: Retrieves the compressor class from `COMPRESSOR_REGISTRY`.
- `neuralops.list_families() -> dict[str, list[str]]`: Groups attention mechanisms and compressors into canonical families (linear, gated, SSM, sparse, hybrid, SVD, pruning, quantization).

### Runners & Execution (5 Functions)
- `neuralops.train(model, dataset, epochs=3, lr=2e-5, **kwargs)`: Fine-tunes a model on classification or sequence tasks with mixed precision and early-stopping.
- `neuralops.distill(student, teacher, dataset, epochs=3, lr=2e-5, **kwargs)`: Runs teacher-student Knowledge Distillation (KD) with temperature-scaled KL loss.
- `neuralops.evaluate(model, dataset="sst2", **kwargs)`: Evaluates task performance, throughput, latency, and memory footprint.
- `neuralops.benchmark(module="cosformer", model="distilbert-base-uncased", dataset="sst2", **kwargs)`: Executes an isolated single benchmark run with watchdog timeouts.
- `neuralops.benchmark_all(modules="all", models="distilbert-base-uncased", dataset="sst2", **kwargs)`: Orchestrates multi-module sweeps and outputs structured CSV/Parquet summaries.

### Utils & Diagnostics (5 Functions)
- `neuralops.info() -> dict[str, Any]`: Returns system diagnostics, active device, and registration counts.
- `neuralops.available_backends() -> list[str]`: Lists available hardware backends (`["cpu", "cuda", "mps", "triton"]`).
- `neuralops.has_triton() -> bool`: Returns `True` if Triton GPU kernel compilation is supported.
- `neuralops.validate(model, input_shape=(2, 32), device="cpu") -> bool`: Checks model device movement, forward output shapes, NaN/Inf absence, and backward gradient flow.
- `neuralops.get_version() -> str`: Returns release version `"1.3.0"`.

---

## 3. The 3-Line Model Surgery Workflow

Transform any standard Transformer (LLaMA, Mistral, Qwen2, GPT-2, DistilBERT, RoBERTa, BERT, T5) into an efficient subquadratic model in 3 lines:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

# Line 1: Load base pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Line 2: Drop-in replace attention with subquadratic Cosformer
model = neuralops.replace_attention(model, "cosformer")

# Line 3: Forward pass with linear O(N) complexity
output = model(torch.randint(0, 50257, (2, 512)))
print("Logits shape:", output.logits.shape)
```

NeuralOps automatically handles:
- Positional embeddings: Preserves RoPE for LLaMA, Mistral, and Qwen2.
- GQA: Automatically aligns query and key-value head counts (`num_key_value_heads`).
- Weight transplant: Slices and copies projection weights (`W_q`, `W_k`, `W_v`, `W_o`) with zero shape mismatch errors.

---

## 4. Deep Model Compression & Factorization

Compress transformer linear projections and attention modules using truncated SVD or low-rank adapters:

```python
from transformers import AutoModelForSequenceClassification
import neuralops

# 1. Load classification model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Compress weights using Direct SVD with rank 32
model = neuralops.compress(model, method="direct_svd", rank=32)

# 3. Validate gradient flow and forward pass
assert neuralops.validate(model, input_shape=(2, 64), device="cpu")
print("Surgery metadata history:", model._neuralops_metadata)
```

---

## 5. Production Export & Reloading

Persist modified models and reload them without writing bespoke reconstruction code:

```python
import neuralops

# 1. Export model to Safetensors with neuralops_config.json
export_dir = neuralops.export_model(
    model=model,
    output_dir="./exports/distilbert-svd32",
    format="safetensors",
)

# 2. Reconstruct model and reload weights seamlessly
reloaded_model = neuralops.load_model(export_dir)
reloaded_model.eval()
```

---

## 6. Benchmarking via CLI

NeuralOps provides an extensive command-line interface for running experiments:

```bash
# Run isolated benchmark on SST-2
python -m neuralops.benchmarks run \
    --module cosformer \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --epochs 3 \
    --batch-size 32

# Plan a grid sweep across all linear attention mechanisms
python -m neuralops.benchmarks plan --all-linear --model distilbert-base-uncased

# Verify system readiness and kernel fallbacks
python -m neuralops.benchmarks verify
```
