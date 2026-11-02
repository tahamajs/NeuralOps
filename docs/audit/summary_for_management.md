# Management Summary — Inference Integration Audit

**Commit:** `215f8f893070317a9926af4925fcf24b337a0e11` · **Date:** 2026-10-21

NeuralOps has meaningful adapter and export code, but the evidence gathered does not certify that arbitrary surgically modified models can be served end-to-end in production engines. The GPU host refused SSH (`Connection refused`), so no package versions, runtime loads, generation, or compile matrix were observed.

- HF: partial; custom manifest/replay loader exists, ordinary HF reconstruction unverified.
- `torch.compile`: partial; custom ops/fake registrations exist, per-model graph tests not run.
- vLLM: plugin/backend scaffolding exists, real checkpoint serving unverified.
- SGLang: radix-cache helper exists; actual SGLang engine integration unverified.
- TensorRT-LLM: scaffold; no-SDK route does not create a verified engine.
- llama.cpp: inspected GGUF option emits metadata + weights, not GGUF.
- ONNX: partial; export path exists, and failure can create a text marker with an .onnx suffix.

**Single biggest blocker:** no proven reconstruction-and-serving contract that guarantees an external engine loads the exact surgically modified architecture and its state/cache semantics.

All 131 × 7 cells in the CSV are marked ⏳ UNTESTED, not inferred. No source code changed; reports are local and were not committed.

