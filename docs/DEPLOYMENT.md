# Production Deployment & Inference Serving Guide (v1.3.0)

NeuralOps provides production-grade model export utilities, Triton kernel compilation strategies, and high-throughput serving adapters designed for modern LLM deployment runtimes (vLLM, Text Generation Inference (TGI), ONNX Runtime, and TensorRT-LLM).

---

## 1. Exporting Models for Production

NeuralOps models can be exported to standard artifact formats with a single line of code or via CLI. During export, surgery and compression metadata are automatically embedded into `neuralops_config.json`, enabling seamless zero-configuration round-trip reloading.

### Supported Export Formats
1. **Safetensors (`format="safetensors"`)**: Production standard for Hugging Face models. Zero-copy memory mapping and secure serialization.
2. **ONNX (`format="onnx"`)**: Universal graph format compatible with ONNX Runtime, TensorRT, and OpenVINO. Automatically traces using CPU fallback to prevent Triton compilation breaks.
3. **TorchScript (`format="torchscript"`)**: Standalone serialized model format for C++ libtorch environments.

### Python API

```python
import neuralops
from transformers import AutoModelForCausalLM

# 1. Load model and apply transformations
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = neuralops.replace_attention(model, "cosformer")
model = neuralops.compress(model, method="direct_svd", rank=32)

# 2. Export to Safetensors with metadata
export_dir = neuralops.export_model(
    model=model,
    output_dir="./exports/gpt2-cosformer-svd32",
    format="safetensors",
)

# 3. Reload model with full architecture reconstruction
reloaded_model = neuralops.load_model(export_dir)
reloaded_model.eval()
```

### Export Directory Layout
```text
exports/gpt2-cosformer-svd32/
├── model.safetensors         # Quantized or factorized model weights
├── neuralops_config.json     # Architecture surgery, ranks, and metadata history
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

# Export to ONNX with dynamic batch and sequence axes
neuralops export \
  --model distilbert-base-uncased \
  --attn cosformer \
  --format onnx \
  --output-dir ./distilbert_onnx
```

---

## 3. High-Throughput Serving with vLLM

NeuralOps provides native integration for vLLM through the `VLLMAdapter` in `neuralops.export.serving` and `neuralops.integrations.serving`.

### Safe Dependency Handling
If `vllm` is not installed in the current environment, NeuralOps will not crash upon import. When an explicit vLLM registration is invoked without the dependency, it raises a clear and actionable exception:
```python
from neuralops.export.serving import VLLMAdapter

try:
    VLLMAdapter.register_attention_backend("cosformer", CosformerBackend)
except RuntimeError as e:
    # Raises: RuntimeError("vLLM is not installed. Please install neuralops[serving] or run: pip install vllm")
    print(e)
```

### Non-Contiguous Paged KV-Cache Forwarding
To serve subquadratic and linear attention models inside vLLM, memory lookups must handle non-contiguous page tables. `VLLMAdapter.paged_attention_forward` routes query states and block tables directly:

```python
import torch
from neuralops.export.serving import VLLMAdapter

# Inputs prepared by vLLM engine:
# query: [num_seqs, num_heads, head_dim]
# key_cache, value_cache: [num_blocks, block_size, num_heads, head_dim]
# block_tables: [num_seqs, max_blocks]
# seq_lens: [num_seqs]

output = VLLMAdapter.paged_attention_forward(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    block_tables=block_tables,
    seq_lens=seq_lens,
    block_size=16,
)
```

### Wrapping Layer Forward Passes for Paged Attention
You can patch a standard NeuralOps attention module in-place to support vLLM's paged execution:
```python
from neuralops.export.serving import VLLMAdapter

# Wrap attention module to automatically intercept paged KV-cache calls
model.transformer.h[0].attn = VLLMAdapter.wrap_forward_for_paged_kv(
    model.transformer.h[0].attn
)
```

---

## 4. Serving with Text Generation Inference (TGI)

NeuralOps provides `TGIAdapter` to package models for Hugging Face Text Generation Inference containers:

```python
from neuralops.export.serving import TGIAdapter

# 1. Freeze weights, set eval mode, and create tgi_config.json
tgi_model = TGIAdapter.prepare_for_tgi(
    model=model,
    output_dir="./tgi_artifacts",
    max_batch_size=32,
    max_sequence_length=4096,
)

# 2. Deploy using Docker
# docker run --gpus all -v ./tgi_artifacts:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data
```

---

## 5. Torch.compile Graph Execution

All 26 Triton kernel modules in NeuralOps are registered with `torch.library.custom_op` and FakeTensor metadata shapes in `neuralops.core.triton_kernels.op_registry`. This allows compiling full models without graph breaks:

```python
import torch
import neuralops
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")
model = neuralops.replace_attention(model, "cosformer")

# Fully compilable in PyTorch 2.x without graph breaks
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```
