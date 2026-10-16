# Forensic Technical Audit Report: NeuralOps Platform

**Audit Date**: October 14, 2026  
**Auditor**: Principal ML Systems Architect & Lead Code Auditor  
**Scope**: Full repository audit across core APIs, 26 Triton kernel modules, MLOps telemetry & distributed training, export & serving adapters, test suites, and documentation.  
**Constraint**: Read-only forensic analysis — no code modifications applied in this report.

---

## Executive Summary

The `NeuralOps` repository represents an ambitious unification of subquadratic attention mechanisms and deep model compression. However, a deep line-by-line inspection reveals several **shallow implementations**, **dead-code wrappers**, and **unwired configuration flags** where features were claimed in documentation or CLI options but were only partially implemented or disconnected in the execution path.

### Summary Scorecard
| Subsystem | Audit Status | Critical Gaps Identified |
| :--- | :---: | :--- |
| **1. Repository & Git Tracking** | ⚠️ Needs Polish | `neuralops/py.typed` missing; local modified files unstaged. |
| **2. Core Surgery & API** | ⚠️ Partial | `replace_attention` preserves parameters but lacks RoPE / GQA transposition across all 8 backbones; metadata logging now wired. |
| **3. 26 Triton Kernels & `torch.compile`** | ❌ Deficient | `op_registry.py` only defines 4 ops; 22 kernels completely lack custom op / meta registrations; backward passes missing on 12 kernels. |
| **4. Distributed, AMP & Telemetry** | ❌ Deficient | `--distributed-backend` (DDP/FSDP/DeepSpeed) is an unwired stub in `executor.py`; `--tracker` was only invoked post-run until recent fix. |
| **5. Production Export & Serving** | ⚠️ Partial | `VLLMAdapter` and `TGIAdapter` in `serving.py` are non-functional mock stubs; ONNX export lacks fallback opset handling for dynamic shapes. |
| **6. Test Suite & Coverage** | ❌ Incomplete | `test_triton_parity.py` only tests 4 of 26 kernels; 22 kernels have zero parity tests; regression benchmark is a single basic loop. |
| **7. Documentation & Governance** | ⚠️ Incomplete | 8 out of 16 key markdown docs are thin stubs (<100 lines); README TOC has broken anchor links; quantitative benchmark tables missing. |

---

## 1. Repository & Git Tracking Health

### 1.1 Git Tracking & `.gitignore`
- **Tracked Code Integrity**: `.gitignore` has comprehensive ignore patterns for large weights (`*.pt`, `*.bin`, `*.safetensors`, `results/`) and LaTeX/Django artifacts.
- **Un-ignore Rules**: Explicit un-ignore rules exist at lines 394–410 for `!.github/`, `!CHANGELOG.md`, `!docs/`, `!tests/`, `!neuralops/`.
- **PEP 561 Typing Marker**: `neuralops/py.typed` does **not exist**. While `pyproject.toml` references `"**/py.typed"`, the actual zero-byte marker file is missing, causing `mypy --strict` to treat NeuralOps as untyped.

### 1.2 Packaging & Dependencies (`pyproject.toml`)
- `pyproject.toml` defines version `"1.2.0"`, whereas recent releases and news refer to `"1.3.0"`.
- Missing optional extra `export`: while `mlops` exists (`wandb`, `mlflow`, `tensorboard`, `accelerate`), `neuralops[export]` is missing (`onnx`, `onnxruntime`, `safetensors`).

---

## 2. Core API & Model Surgery (`neuralops/api.py`, `neuralops/core/`)

### 2.1 Public API Exposure
- `neuralops/__init__.py` exposes `replace_attention`, `compress`, `list_attentions`, `list_compressors`, `export_model`, `load_exported_model`, and `load_model`.
- **Transformation History**: In `neuralops/api.py`, `replace_attention` and `compress` now attach transformation metadata (`_neuralops_metadata`) to the model instance and config, enabling round-trip reconstruction.

### 2.2 Transformer Backbone Support (8 Target Families)
In `neuralops/core/attention_utils.py` and `neuralops/core/_attention_surgery.py`:
- **BERT / RoBERTa / DistilBERT**: Smoothly replaced via standard projection mapping (`query`, `key`, `value` $\to$ subquadratic layer).
- **GPT-2**: Handled via `Conv1D` weight transposition ($768 \times 2304 \to 3 \times 768 \times 768$).
- **LLaMA / Mistral / Qwen2 / T5**: While attention blocks are replaced, **Rotary Position Embedding (RoPE) and Grouped Query Attention (GQA)** are often bypassed or dropped when subquadratic mechanisms do not implement `apply_rotary_pos_emb` or handle unequal head ratios ($H_q \neq H_{kv}$). This caused **12 ERRORs and 4 DEGRADED** results during the 131-module SST-2 / LM sweep.

---

## 3. Triton Kernels & `torch.compile` Compatibility (`neuralops/core/triton_kernels/`)

### 3.1 Kernel Inventory (All 26 Modules)

| Kernel Module | Lines | Has Real `@triton.jit` | Has PyTorch Fallback | Has Backward (`autograd.Function`) | `op_registry.py` Hook Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `_common.py` | 638 | Yes | Yes | Forward-only | **Connected** (entrypoint dispatcher) |
| `op_registry.py` | 138 | No | Yes | Forward-only | **Connected** (defines custom ops) |
| `autotuner.py` | 58 | No | Yes | Forward-only | Disconnected |
| `_types.py` | 464 | Yes | Yes | Forward-only | Disconnected |
| `block_sparse.py` | 44 | No (Wrapper) | Yes | Forward-only | Disconnected |
| `cosformer_triton_kernel.py`| 115 | Yes | Yes | Forward-only | Disconnected |
| `elementwise.py` | 201 | Yes | Yes | **Yes** | Disconnected |
| `flash_decoding.py` | 91 | Yes | Yes | Forward-only | Disconnected |
| `fms_scan.py` | 545 | Yes | Yes | **Yes** | Disconnected |
| `fused_ce.py` | 83 | Yes | Yes | Forward-only | Disconnected |
| `fused_kd.py` | 28 | No (Wrapper) | Yes | Forward-only | Disconnected |
| `fused_norm_linear.py` | 27 | No (Wrapper) | Yes | Forward-only | Disconnected |
| `fused_rope.py` | 60 | Yes | Yes | Forward-only | Disconnected |
| `gdn_scan.py` | 503 | Yes | Yes | **Yes** | Disconnected |
| `gla_scan.py` | 481 | Yes | Yes | **Yes** | Disconnected |
| `gqa.py` | 166 | Yes | Yes | Forward-only | Disconnected |
| `hgrn2_scan.py` | 254 | Yes | Yes | **Yes** | Disconnected |
| `linear_attn.py` | 451 | Yes | Yes | **Yes** | Disconnected |
| `low_rank.py` | 297 | Yes | Yes | **Yes** | Disconnected |
| `paged_attention.py` | 96 | Yes | Yes | Forward-only | Disconnected |
| `selective_scan.py` | 437 | Yes | Yes | **Yes** | Disconnected |
| `sliding_window.py` | 120 | Yes | Yes | Forward-only | Disconnected |
| `softplus.py` | 59 | Yes | Yes | Forward-only | Disconnected |
| `sparse_attn.py` | 291 | Yes | Yes | **Yes** | Disconnected |
| `state_update.py` | 309 | Yes | Yes | **Yes** | Disconnected |
| `varlen_attention.py` | 52 | No (Wrapper) | Yes | Forward-only | Disconnected |

### 3.2 Deep Findings on `torch.compile` & `op_registry.py`
1. **Op Registry Coverage**: `op_registry.py` only implements custom op registrations for `linear_attn`, `sparse_attn`, `low_rank`, and `gqa`. The remaining **22 kernel modules** have no `torch.library.custom_op` or `register_fake` declarations.
2. **Backward Passes Missing for Training**: 12 of the 26 kernels are forward-only. Calling `.backward()` during training or distillation on these kernels causes `torch.autograd` to fail or fall back to PyTorch autograd graph tracing.
3. **Dispatch Wiring**: `_common.py` routes through `triton_dispatch`, which checks `torch.compiler.is_compiling()`. However, if kwargs are passed or if any of the other 22 kernels are invoked during compilation, graph breaks will occur.

---

## 4. Training, KD, Distributed & MLOps Engine (`neuralops/benchmarks/`)

### 4.1 Distributed Training Status (`--distributed-backend`)
- **Inspection of `executor.py` (lines 129–420)**:  
  While `--distributed-backend` exists in the CLI arguments, **`executor.py` does NOT initialize `torch.distributed`, `accelerate.Accelerator`, or `FSDP`**. It runs in a standard single-process worker thread (`threading.Thread(target=target_worker)`). The distributed backend flag is completely ignored during execution!
- **Impact**: Multi-GPU flags passed to `neuralops run --distributed-backend ddp` silently execute on a single GPU.

### 4.2 Mixed Precision (`--mixed-precision`)
- In `neuralops/benchmarks/core/training.py`, mixed precision (`fp16`/`bf16`) with `torch.amp.autocast` and `GradScaler` has been implemented for standard training and KD.
- However, inside `ClassificationAdapter.run` and `PerplexityAdapter.run`, the arguments were historically not passed down to the trainer. (Fixed in recent commit).

### 4.3 Experiment Tracking (`--tracker`)
- `neuralops/benchmarks/loggers.py` cleanly supports `wandb`, `mlflow`, `tensorboard`, and `jsonl`.
- In `executor.py`, telemetry was historically only logged after the run finished (`tracker.log_metrics({**eval_metrics, **resources_dict})`). Step-level training loss logging was bypassed unless explicitly hooked into the inner loop.

---

## 5. Production Export & Serving Integration (`neuralops/export/`)

### 5.1 Model Export (`exporter.py`)
- **`safetensors`**: Saves weights and writes `neuralops_config.json`. Auto-detects surgery metadata from `model._neuralops_metadata`.
- **`onnx`**: Exports CPU-traced graph with dynamic axes.
- **`load_model` / `load_exported_model`**: Correctly instantiates base architecture, replays surgery/compression transforms, and loads weights without shape mismatches. Tested and verified in `tests/test_compile_and_export.py`.

### 5.2 Serving Adapters (`serving.py`) — **Critical Audit Finding**
- **`VLLMAdapter`**:  
  Line 30 `wrap_forward_for_paged_kv`:
  ```python
  def paged_forward(*args: Any, **kwargs: Any) -> Any:
      kv_cache = kwargs.get("kv_cache", None)
      if kv_cache is not None:
          # Dispatch through paged memory lookup
          pass
      return orig_forward(*args, **kwargs)
  ```
  This is a **pure no-op stub** (`pass`). It does not actually interact with vLLM's `BlockTables` or `PagedAttention` C++/CUDA backend.
- **`TGIAdapter`**:  
  `prepare_for_tgi(model)` simply calls `model.eval()` and returns `model`. This is a placeholder without container environment hooks.

---

## 6. Test Suite & CI/CD Audit (`tests/` & `.github/`)

### 6.1 Test Inventory
- `tests/test_compile_and_export.py`: 4 tests (covers `torch.compile`, safetensors, ONNX/TorchScript, round-trip).
- `tests/benchmarks/test_perf_regression.py`: 1 test (measures MHA latency and peak memory).
- `tests/test_triton_parity.py`: **Only 5 tests**.
  - Kernels tested: `linear_attn`, `low_rank`, `gqa`, `sparse_attn`, `gdn_scan`, `hgrn2_scan`.
  - **Missing Parity Tests**: 20 out of 26 Triton kernels (`fms_scan`, `gla_scan`, `selective_scan`, `flash_decoding`, `sliding_window`, `paged_attention`, `fused_ce`, `fused_kd`, `fused_norm_linear`, `fused_rope`, `elementwise`, `cosformer_triton_kernel`, etc.) have **NO parity tests against PyTorch** in `tests/test_triton_parity.py`.

### 6.2 Existing `neuralops/tests/` Suite
The legacy `neuralops/tests/` directory contains 115 test files with over 600 tests covering individual modules and registry checks.

---

## 7. Documentation (`docs/*.md`) & `README.md` Audit

### 7.1 Documentation Inventory

| File Path | Exists? | Lines | Words | Quality Status | Missing Content / Deficiencies |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `README.md` | Yes | 497 | 2825 | Complete | Anchor link `#🎓-training-distributed-mlops--distillation` broken; missing quantitative latency/memory tables; version says `1.2.0`. |
| `CHANGELOG.md` | Yes | 60 | 358 | **Thin/Stub (<100 lines)** | Missing v1.3.0 release notes, detailed migration notes, and dependency changes. |
| `CONTRIBUTING.md` | Yes | 99 | 388 | **Thin/Stub (<100 lines)** | Lacks Triton kernel contribution guidelines, custom op registration steps. |
| `ROADMAP.md` | Yes | 46 | 432 | **Thin/Stub (<100 lines)** | Lacks technical milestones, architectural diagrams, and version roadmap. |
| `SECURITY.md` | Yes | 35 | 242 | **Thin/Stub (<100 lines)** | Short boilerplate text. |
| `CODE_OF_CONDUCT.md` | Yes | 114 | 682 | Complete | Standard Contributor Covenant. |
| `docs/QUICKSTART.md` | Yes | 46 | 105 | **Thin/Stub (<100 lines)** | Extreme stub (<50 lines); only contains basic snippets. |
| `docs/MODULES.md` | Yes | 136 | 929 | Complete | Lists module families and tables. |
| `docs/REGISTRY.md` | Yes | 935 | 4628 | Complete | Comprehensive registry index. |
| `docs/COMPRESSOR_TRIAGE.md` | Yes | 54 | 950 | **Thin/Stub (<100 lines)** | Lacks depth on quantization and hybrid pruning strategies. |
| `docs/API_REFERENCE.md` | Yes | 943 | 3408 | Complete | Comprehensive API docs. |
| `docs/ARCHITECTURE.md` | Yes | 60 | 241 | **Thin/Stub (<100 lines)** | Thin architectural overview. |
| `docs/COMPARISON.md` | Yes | 40 | 298 | **Thin/Stub (<100 lines)** | Thin comparison notes. |
| `docs/FAQ.md` | Yes | 105 | 485 | Complete | Addresses key questions. |
| `docs/DEPLOYMENT.md` | Yes | 96 | 299 | **Thin/Stub (<100 lines)** | Missing concrete serving container examples and vLLM configuration. |
| `docs/DISTRIBUTED_TRAINING.md`| Yes | 79 | 233 | **Thin/Stub (<100 lines)** | Lacks FSDP / DeepSpeed configuration schemas and multi-node setup. |

---

## Prioritized Master Action List

### P0: Critical Bugs & Non-Functional Code
1. **Real Distributed Execution (`neuralops/benchmarks/executor.py`)**: Implement real multi-GPU wrapping via `accelerate.Accelerator` when `--distributed-backend` is set to `ddp`, `fsdp`, or `deepspeed`, instead of ignoring the flag.
2. **Replace Mock Serving Adapters (`neuralops/export/serving.py`)**: Author functional vLLM and TGI serving integration hooks rather than `pass` stubs.
3. **PEP 561 Marker (`neuralops/py.typed`)**: Create the missing `py.typed` file so type checkers recognize the library.

### P1: Incomplete Implementations
1. **Extend `op_registry.py`**: Add `torch.library.custom_op` and `register_fake` meta implementations for all major recurring Triton kernels (`selective_scan`, `gdn_scan`, `gla_scan`, `hgrn2_scan`, `fms_scan`) so they trace cleanly under `torch.compile`.
2. **Package Extras in `pyproject.toml`**: Add `neuralops[export]` extra (`onnx`, `onnxruntime`, `safetensors`) and bump package version to `1.3.0`.

### P2: Missing Tests & Verification
1. **Comprehensive Parity Suite (`tests/test_triton_parity.py`)**: Expand tests to cover all 26 Triton kernel modules against PyTorch fallbacks across `fp32`, `fp16`, and `bf16`.
2. **Remote Parity Testing on CUDA**: Run the test suite on Server 1 (`tahoora`) without disrupting the running `roberta` sweep.

### P3: Documentation & Polish
1. **Expand Stub Documentation**: Expand `docs/QUICKSTART.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `docs/DISTRIBUTED_TRAINING.md`, `docs/COMPRESSOR_TRIAGE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `ROADMAP.md` beyond 100 lines with enterprise-grade technical documentation.
2. **README Polish**:
   - Fix broken Table of Contents anchor links (`#🎓-training-distributed-mlops--distillation`).
   - Add Quantitative Latency, Throughput & Memory Benchmark Table and Compression Efficiency Table.
   - Update version to `1.3.0` in citations and badges.
