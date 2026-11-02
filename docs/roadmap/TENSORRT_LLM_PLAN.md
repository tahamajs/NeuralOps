# TensorRT-LLM Integration Roadmap

## Status: Not Implemented (v1.5.0)

TensorRT-LLM native plugin support is **not implemented** in NeuralOps v1.5.0. This document records the current state, the required work, and the alternative paths for users.

## Current Behavior

NeuralOps does not produce TensorRT engines. Instead of writing placeholder artifacts that look like valid engines but are not, all TensorRT-related code paths explicitly raise `NotImplementedError`.

Specifically:
- `TensorRTLLMPluginBuilder.build_plugin()` raises `NotImplementedError` when TensorRT SDK is unavailable
- `TensorRTLLMPluginBuilder.export_engine()` does not write a `.trt` file when TensorRT is unavailable
- `TensorRTRuntimeRunner.__init__()` raises `NotImplementedError` when no real engine exists
- No `NEURALOPS_TRT_ENGINE_V1` placeholder bytes are written
- No random logits are returned by any runtime stub

The rationale for this design is that silent fallbacks to fake outputs are worse than explicit failures. A user who encounters `NotImplementedError` immediately knows that the capability does not exist and can choose an alternative. A user who receives a `.trt` file that looks valid but is not would waste hours debugging before realizing the artifact was a placeholder.

## Required for Implementation

Completing TensorRT-LLM support requires the following work, estimated at 8-12 engineer-weeks.

### Phase 1: Native TensorRT Plugin (4-6 weeks)

Each NeuralOps attention mechanism (131 total) must be implemented as a TensorRT plugin using the `nvinfer1::IPluginV2DynamicExt` C++ interface. The plugin must implement:

- `getOutputDimensions()` with correct tensor layout
- `enqueue()` with the attention forward computation
- `serialize()` / `deserialize()` for engine caching
- `attachToContext()` / `detachFromContext()` for graph capture

For SSM-based attentions (Mamba, HGRN2), the plugin must expose state-cache interfaces that TensorRT recognizes.

### Phase 2: Version Compatibility (2-3 weeks)

TensorRT plugins are version-specific. Each TensorRT major release (9.0, 9.1, 10.0, 10.1) requires recompilation. The build system must:

- Detect installed TensorRT version
- Compile the plugin `.so` against the matching API
- Handle ABI incompatibilities between versions
- Provide fallback messaging when version is unsupported

### Phase 3: Integration with auto_deploy (2-3 weeks)

TensorRT-LLM 0.15+ provides `auto_deploy`, a PyTorch-first deployment path. The NeuralOps integration must:

- Register custom ops via `torch.library.custom_op`
- Publish a custom op registry to `auto_deploy`
- Implement the compilation graph
- Verify output parity against PyTorch reference

### Phase 4: Testing Infrastructure (1-2 weeks)

Testing requires:

- NVIDIA GPU with TensorRT installed (RTX 4070 or better)
- Reference outputs from PyTorch for all 131 attention mechanisms
- Assertions that engine output matches PyTorch within 1e-3 tolerance
- CI integration on Linux + CUDA

## Alternative Paths

Users who need TensorRT-like performance without waiting for full implementation have several options.

**Option A: vLLM with NeuralOps attention.** Once the vLLM backend is completed (see `docs/roadmap/VLLM_PLAN.md`), models can be served with vLLM's PagedAttention and continuous batching. vLLM does not use TensorRT but achieves similar throughput on NVIDIA hardware.

**Option B: Torch.compile with Inductor.** For models that fit in VRAM, `torch.compile` with the Inductor backend generates optimized kernels that approach TensorRT performance on common workloads. NeuralOps custom ops are compatible with `torch.compile(fullgraph=True)`.

**Option C: HuggingFace Transformers with NeuralOps surgery.** For correctness-critical deployments, use `neuralops.replace_attention()` on a HuggingFace model and run standard `model.generate()`. This is the safest path but the slowest.

**Option D: Wait for upstream.** Some NeuralOps attention mechanisms (linear attention, sliding-window, GQA) may become native in future TensorRT-LLM releases. Track the TensorRT-LLM GitHub repository for updates.

## Design Philosophy

The decision to raise `NotImplementedError` rather than write placeholder artifacts follows from a broader principle in NeuralOps: silent failures and fake successes are worse than honest limitations. The earlier audit of this repository found that several "working" integration paths were returning random tensors or writing empty files. Those paths have been removed or converted to explicit failures.

The TensorRT roadmap is intentionally empty of code because the code would be unverified without the SDK. When the SDK is available and the implementation is complete, this document will be updated with concrete usage examples and benchmark numbers.

## Related Documents

- `docs/roadmap/VLLM_PLAN.md` — vLLM backend status
- `docs/roadmap/SGLANG_PLAN.md` — SGLang backend status  
- `docs/audit/INFERENCE_ENGINE_INTEGRATION_AUDIT_2026_10_21.md` — full integration audit
- `docs/integrations/TENSORRT_LLM.md` — integration guide (when ready)

## Version History

- **v1.5.0 (2026-10-22)**: All TensorRT paths now raise `NotImplementedError`. Previous versions wrote placeholder artifacts that were not real engines. This is a breaking change for anyone who relied on those artifacts, but such reliance was incorrect.
- **v1.4.0**: Placeholder `.trt` files written with header `NEURALOPS_TRT_ENGINE_V1`. These were not valid TensorRT engines.
- **v1.3.x**: No TensorRT integration.

## Contact

For questions about TensorRT-LLM integration or to contribute to the implementation, open an issue on the NeuralOps GitHub repository with the label `tensorrt-llm`.
