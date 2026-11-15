# Subagent W7-4 Forensic Audit & Remediation Report
**Subagent ID:** W7-4  
**Branch:** `wave7/W7-4`  
**Domain:** Export Pipelines & Format Serialization  
**Timestamp:** 2026-11-15T17:35:00Z  

---

## 1. Executive Summary

Wave 7 Subagent W7-4 was tasked with remediating export pipelines and format serialization across `neuralops/export/` and `neuralops/operations/export/`. Prior to remediation:
1. `neuralops/operations/export/model.py` had format stubs (`IntegrationNotImplementedError` raised for `vllm`, `sglang`, `onnx`, etc.).
2. Sharded SafeTensors export was unsupported in `neuralops/export/safetensors_exporter.py`.
3. Dedicated ONNX exporter with dynamic shape handling for recurrent and stateful models was missing.
4. GGUF export was not wrapped cleanly under `neuralops/export/gguf.py` for standard llama.cpp pipelines.

Subagent W7-4 successfully completed all tasks, eliminating all stubs, implementing complete sharded serialization with index JSON and manifest generation, authoring ONNX recurrent dynamic shape export, and integrating all formats under `ExportModelOperation`.

All 8 tests in `tests/export/test_model_export.py` and all 34 tests across `tests/export/` passed 100% green with zero lint issues.

---

## 2. Inventory of Owned & Created Files

| File | Status | Description |
|---|---|---|
| `neuralops/operations/export/model.py` | Remediated | Eliminates format stubs; supports `hf`, `safetensors`, `vllm`, `sglang`, `onnx`, `gguf` |
| `neuralops/export/safetensors.py` | Created | Full sharded SafeTensors serialization with index, manifest, and `SafeTensorsExporter` |
| `neuralops/export/onnx.py` | Created | ONNX export engine with dynamic axes configuration for recurrent sequence/batch models |
| `neuralops/export/gguf.py` | Created | GGUF v3 binary container exporter compatible with llama.cpp specifications |
| `neuralops/export/__init__.py` | Updated | Public lazy symbol re-exports including sharded SafeTensors, ONNX, and GGUF |
| `tests/export/test_model_export.py` | Created | 8 unit and integration tests verifying sharding, dynamic shapes, and operation verbs |
| `audits/wave7/w7_4_evidence/test_execution.log` | Created | Complete pytest execution transcript (8/8 passed) |
| `audits/wave7/w7_4_report.md` | Created | This formal audit and remediation report |

---

## 3. Detailed Technical Remediation

### 3.1 Sharded SafeTensors (`neuralops/export/safetensors.py`)
- Implemented `_parse_size` supporting human-readable strings (`"500MB"`, `"2GB"`) and integer bytes.
- Implemented `shard_state_dict` to partition model weights across individual shard files conforming to the pattern `model-0000X-of-0000Y.safetensors`.
- Generated `model.safetensors.index.json` containing `metadata` and `weight_map` tensor mappings adhering to HuggingFace weight sharding standards.
- Extended `SafeTensorsExporter` to handle both single-file and sharded export via optional `max_shard_size`.

### 3.2 Dynamic Shape ONNX Export (`neuralops/export/onnx.py`)
- Created `build_dynamic_axes_for_recurrent` to automatically construct dynamic batch size and sequence length axis mappings.
- Implemented `export_to_onnx` with atomic temporary file writes, fallback handling for TorchScript tracing vs Dynamo, and onnxruntime graph verification.

### 3.3 GGUF v3 Container Export (`neuralops/export/gguf.py`)
- Created `export_model_to_gguf` wrapping pure-Python GGUF v3 binary serialization with Q8_0 and F16 quantization support.

### 3.4 Universal Export Operation (`neuralops/operations/export/model.py`)
- Removed stub `IntegrationNotImplementedError` raised for formats outside HF.
- Routed format dispatch to respective engines:
  - `safetensors`: Sharded SafeTensors export with manifest
  - `onnx`: Dynamic shape ONNX export
  - `gguf`: GGUF binary export
  - `vllm` / `sglang`: Serving bundles with paged attention and engine configuration manifests
  - `hf`: HuggingFace safetensors format

---

## 4. Verification & Testing Evidence

### Test Execution Command
```bash
python -m pytest tests/export/test_model_export.py -v
```

### Result Summary
```text
tests/export/test_model_export.py::test_safetensors_shard_state_dict_single_shard PASSED [ 12%]
tests/export/test_model_export.py::test_safetensors_shard_state_dict_multiple_shards PASSED [ 25%]
tests/export/test_model_export.py::test_save_sharded_safetensors_writes_valid_files PASSED [ 37%]
tests/export/test_model_export.py::test_safetensors_exporter_class_integration PASSED [ 50%]
tests/export/test_model_export.py::test_build_dynamic_axes_for_recurrent PASSED [ 62%]
tests/export/test_model_export.py::test_onnx_export_recurrent_model PASSED [ 75%]
tests/export/test_model_export.py::test_gguf_export_model_to_gguf PASSED [ 87%]
tests/export/test_model_export.py::test_export_model_operation_all_formats PASSED [100%]

============================== 8 passed in 1.23s ===============================
```

### Entire Export Suite Verification
```bash
python -m pytest tests/export/ -v
# 34 passed in 84.89s (100% green)
```

### Static Analysis & Linter
```bash
python -m ruff check neuralops/operations/export/model.py neuralops/export/safetensors.py neuralops/export/onnx.py neuralops/export/gguf.py tests/export/test_model_export.py
# All checks passed!
```

---

## 5. Invariant Certification
- **Zero Stubs:** All `NotImplementedError` stubs in owned paths eliminated.
- **Strict Typing:** All public methods annotated with concrete return types and Google docstrings.
- **Clean Execution:** No naked prints, no hardcoded devices.
