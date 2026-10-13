# Production Deployment & Inference Serving Guide (v1.3.0)

NeuralOps provides production-grade model export utilities, Triton kernel compilation strategies, and high-throughput serving adapters designed for modern LLM deployment runtimes (vLLM, TGI, ONNX Runtime, and TensorRT-LLM).

---

## 1. Exporting Models for Production

NeuralOps models can be exported to standard artifact formats with a single line of code or via CLI. During export, surgery and compression metadata are automatically embedded into `neuralops_config.json`, enabling seamless zero-configuration round-trip reloading.

### Python API

```python
import neuralops
from transformers import AutoModelForCausalLM

# 1. Load model and apply transformations
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = neuralops.replace_attention(model, "cosformer")
model = neuralops.compress(model, method="adasvd", rank=32)

# 2. Export to Safetensors with metadata
export_dir = neuralops.export_model(
    model=model,
    output_dir="./exports/gpt2-cosformer-adasvd",
    format="safetensors",
)

# 3. Reload model with architecture reconstruction
reloaded_model = neuralops.load_model(export_dir)
reloaded_model.eval()
```

### Export Directory Layout
```text
exports/gpt2-cosformer-adasvd/
├── model.safetensors         # Quantized or factorized model weights
├── neuralops_config.json     # Architecture surgery & rank configuration
└── config.json               # Host HuggingFace configuration
```

---

## 2. Command-Line Export Tool

Export any supported model directly from your terminal:

```bash
# Export HuggingFace model to Safetensors
neuralops export \
  --model distilbert-base-uncased \
  --attn cosformer \
  --comp direct_svd \
  --rank 32 \
  --format safetensors \
  --output-dir ./distilbert_prod

# Export to ONNX with automatic CPU/GPU fallback tracing
neuralops export \
  --model distilbert-base-uncased \
  --attn linear_attn \
  --format onnx \
  --output-dir ./distilbert_onnx
```

---

## 3. High-Throughput Serving with vLLM

NeuralOps provides the `VLLMAdapter` to wire custom linear, gated, and recurrent attention layers into vLLM's paged KV cache tables:

```python
import torch
from neuralops.export.serving import VLLMAdapter
from transformers import AutoModelForCausalLM
import neuralops

# 1. Prepare model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = neuralops.replace_attention(model, "gqa_attention")

# 2. Wrap attention forward pass for paged KV cache handling
for block in model.transformer.h:
    block.attn = VLLMAdapter.wrap_forward_for_paged_kv(block.attn)

# 3. Register backend if running inside vLLM engine process
VLLMAdapter.register_attention_backend("neuralops_gqa", type(model.transformer.h[0].attn))
```

---

## 4. Text Generation Inference (TGI) Integration

To deploy models into HuggingFace Text Generation Inference (TGI) container runtimes:

```python
from neuralops.export.serving import TGIAdapter

# Freeze weights, set eval mode, and export TGI manifest
tgi_model = TGIAdapter.prepare_for_tgi(
    model=model,
    output_dir="./tgi_model_store",
    max_batch_size=64,
    max_sequence_length=8192,
)
```

The resulting `tgi_config.json` specifies:
```json
{
  "deployment_target": "text-generation-inference",
  "max_batch_size": 64,
  "max_sequence_length": 8192,
  "dtype": "float16",
  "custom_attention": [
    {
      "action": "replace_attention",
      "attention_type": "gqa_attention"
    }
  ]
}
```

---

## 5. Acceleration with `torch.compile`

All 26 NeuralOps Triton kernels are registered via `torch.library.custom_op` and register FakeTensor meta shapes to guarantee zero graph breaks during Inductor compilation:

```python
import torch
import neuralops
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = neuralops.replace_attention(model, "linear_attn")

# Compile with reduce-overhead mode
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

# Warmup and execute
dummy_input = torch.randint(0, 50257, (1, 64))
with torch.no_grad():
    for _ in range(3):
        _ = compiled_model(dummy_input)
```

---

## 6. Docker Container Deployment

Sample Dockerfile for containerized NeuralOps microservices:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Install NeuralOps with CUDA and export extras
RUN pip install --no-cache-dir "neuralops[cuda,export]>=1.3.0"

# Copy exported model artifacts
COPY ./exports/gpt2-cosformer-adasvd /workspace/model

# Serve with FastAPI or custom inference engine
CMD ["python", "-m", "neuralops.inference.server", "--model-dir", "/workspace/model", "--port", "8000"]
```
