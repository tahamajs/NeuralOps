# Deferred to GPU Validation

Date: 2026-10-21

No CUDA kernels, CUDA sources, or `_common.cuh` files were modified. The prior source review did not establish a new kernel-level defect that could be safely repaired without hardware execution.

## Explicitly deferred

- TensorRT integration still contains a no-SDK placeholder engine artifact and a random-output fallback in `neuralops/integrations/tensorrt_llm.py`. The current mission explicitly prohibits touching this file without GPU validation; consequently these paths remain known correctness hazards, not supported inference behavior.
- CUDA/Triton execution, numerical parity, latency, throughput, and VRAM claims remain unverified in this CPU-only phase.
- vLLM and SGLang server loading/generation require a compatible GPU environment; no server tests were run.

Required evidence when a permitted GPU host is available: pinned driver/CUDA/PyTorch/Triton versions, `nvidia-smi`, variant-specific test output, real engine deserialize/load, and numerical comparison against eager PyTorch. Never accept file existence as engine validation.
