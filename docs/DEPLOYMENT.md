# Production Deployment & Inference Serving Guide

NeuralOps provides production export utilities and serving adapters designed for high-throughput inference runtimes.

---

## 1. Exporting Models via Python API

Export models to Safetensors, ONNX, or TorchScript with a single call:

```python
import neuralops
from transformers import AutoModel

# 1. Load and modify base model
model = AutoModel.from_pretrained("distilbert-base-uncased")
model = neuralops.replace_attention(model, "cosformer")
model = neuralops.compress(model, method="adasvd", rank=32)

# 2. Export model for production deployment
out_dir = neuralops.export_model(
    model=model,
    output_dir="./distilbert_cosformer_adasvd",
    format="safetensors",  # or "onnx", "torchscript"
    model_name_or_path="distilbert-base-uncased",
    attention_type="cosformer",
    compression_method="adasvd",
    rank=32,
)
print(f"Model exported to {out_dir}")
```

### Generated Files in Export Directory
- `model.safetensors`: Serialized model tensor weights.
- `neuralops_config.json`: Metadata capturing attention type, compression method, and rank for exact architecture reconstruction.
- `config.json`: Original Hugging Face model configuration.

---

## 2. Exporting via Command-Line Interface

You can export models directly from the terminal without writing Python code:

```bash
# Export to Safetensors
neuralops export \
  --model distilbert-base-uncased \
  --attn cosformer \
  --comp adasvd \
  --rank 32 \
  --format safetensors \
  --output-dir ./exported_model

# Export to ONNX (with deterministic fallback execution)
neuralops export \
  --model distilbert-base-uncased \
  --attn linear_attn \
  --format onnx \
  --output-dir ./exported_onnx
```

---

## 3. Acceleration with `torch.compile`

All 26 NeuralOps Triton kernels and subquadratic modules are registered with custom op meta kernels, allowing full compilation without graph breaks:

```python
import torch
import neuralops
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")
model = neuralops.replace_attention(model, "linear_attn")

# Compile with PyTorch 2.x Inductor backend
compiled_model = torch.compile(model, mode="reduce-overhead")

# Inference with compiled speedups
inputs = torch.randint(0, 1000, (1, 128))
with torch.no_grad():
    outputs = compiled_model(inputs)
```

---

## 4. Serving with vLLM

NeuralOps provides adapters to integrate custom subquadratic attention with vLLM's paged memory tables:

```python
from neuralops.export.serving import VLLMAdapter

# Register custom attention layer with vLLM's attention dispatcher
VLLMAdapter.register_attention_backend("cosformer", model.transformer.layer[0].attention)
```
