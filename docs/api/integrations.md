# Serving Integrations API Reference

NeuralOps provides production serving adapters for high-throughput model deployment.

## Supported Frameworks

### 1. Hugging Face Transformers (`neuralops.integrations.hf_automodel`)
- Drop-in `replace_attention()` surgery.
- Dynamic `AutoConfig` and `AutoModel` registration in `transformers.CONFIG_MAPPING`.
- Bit-exact serialization via `export_model` and `load_exported_model`.

### 2. ONNX Runtime (`neuralops.export.exporter`)
- PyTorch graph tracing with deterministic `dynamo=False` fallback.
- Dynamic batch and sequence length axes.

### 3. vLLM (`neuralops.models.paged_state`)
- `PagedStateTable`: Virtual-to-physical GPU recurrent state allocation.
- Reference counting and zero-copy sequence forking.
- Copy-on-Write (CoW) page isolation.

### 4. Text Generation Inference (`neuralops.integrations.tgi`)
- `TGIServingAdapter`: Continuous batching and token streaming.
- Standalone FastAPI server hook (`/generate`, `/generate_stream`).

### 5. SGLang (`neuralops.integrations.sglang`)
- `SGLangRadixAdapter`: Radix tree prefix caching for prompt reuse.
- Least-Recently-Used (LRU) leaf node eviction.

### 6. TensorRT-LLM (`neuralops.integrations.tensorrt_llm`)
- `TensorRTLLMPluginBuilder`: Engine builder, C++ source generator, and runtime runner.
