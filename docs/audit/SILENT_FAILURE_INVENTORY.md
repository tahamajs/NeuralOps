# Silent Failure Inventory — v1.5.0 Audit

Date: 2026-10-21  
Scope: `neuralops/export/` and `neuralops/integrations/`; source inspection only. Python and pytest were not executed.

| Location | Previous behavior | Current disposition |
|---|---|---|
| `neuralops/export/exporter.py` ONNX branch (before this audit correction, line 209) | If both ONNX export attempts failed, wrote a plain-text exception marker to `model.onnx`. | Changed locally: exports to a temporary .onnx path, atomically publishes only after exporter success, removes the temp artifact on failure, and raises a RuntimeError naming the requested attention, underlying exception, and safetensors workaround. Not runtime-tested. |
| `neuralops/export/exporter.py` GGUF branch (before this audit correction, lines 238–270) | Wrote `gguf_metadata.json` and safetensors/bin weights while accepting `format="gguf"`; no GGUF container was generated. | Changed locally: `gguf`/`ggml` now raises NotImplementedError before creating the output directory, names the attention if available, and suggests a concrete safetensors + external converter workaround. Not runtime-tested. |
| `neuralops/integrations/tensorrt_llm.py:455–489` | When TensorRT is unavailable, writes a manifest and a file beginning with placeholder bytes `NEURALOPS_TRT_ENGINE_V1`. | **Unresolved / deliberately untouched.** The current mission explicitly forbids modifying `tensorrt_llm.py` without GPU validation. This means the silent-artifact inventory is not fully cleared and Phase 0 cannot pass its zero-match gate. |
| `neuralops/integrations/tensorrt_llm.py:523–566` | If engine deserialization is unavailable/failed, `TensorRTRuntimeRunner.forward()` returns random tensors. | **Unresolved / deliberately untouched** under the same no-touch constraint. This is not an engine result and must not be represented as successful inference. |

## Search evidence

Static search after the local exporter edits:

- ONNX text-marker writer: removed from `neuralops/export/exporter.py`.
- GGUF metadata-only writer: removed from `neuralops/export/exporter.py`; request now fails before output-directory creation.
- TensorRT sentinel and random-output path: still present in `neuralops/integrations/tensorrt_llm.py`, which this mission forbids editing.

The exact grep is recorded in the completion report. Do not describe the codebase as free of all silent failure paths until the TensorRT file is safely changed and tests pass in an allowed environment.
