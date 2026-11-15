# NEURALOPS WAVE 7 — DEEP SUBSYSTEM REMEDIATION & COMPLETION ROADMAP

**Document version:** 1.0  
**Effective date:** 2026-11-15  
**Mission owner:** Human Coordinator (Taha Majlesi)  
**Status:** Canonical Wave 7 Action Plan (All directories audited, zero shortcuts)

---

## 0. EXECUTIVE SUMMARY & WAVE 7 MISSION

While Wave 5 focused on **Release Verification & Integration (v2.4.0)** and Wave 6 focused on **Examples Remediation**, **Wave 7** is dedicated to **Deep Subsystem Completeness & Deferred API Hardening**.

An exhaustive scan across all 35 subsystems under `neuralops/` revealed key areas where operations were deferred (`"deferred to Phase 2"`), where optional integrations fail closed with `IntegrationNotImplementedError`, and where adapter interfaces require complete native implementations.

---

## 1. COMPREHENSIVE DIRECTORY-BY-DIRECTORY AUDIT & TODO CHECKLIST

### 📂 `neuralops/operations/` (28 files, 27 deferred operations)
The `neuralops/operations/` package exposes high-level surgical and transform verbs. Many currently fail closed with `IntegrationNotImplementedError("... is deferred to Phase 2.")`.
- [ ] **`neuralops/operations/attention/configure.py`**:
  - Replace `IntegrationNotImplementedError("attention.configure runtime tuning is deferred to Phase 2.")` with dynamic kernel selection and block-size auto-tuning based on device properties.
- [ ] **`neuralops/operations/attention/replace.py`**:
  - Implement full in-place attention swap routing through `UniversalSurgeryDirector` without hard fail.
- [ ] **`neuralops/operations/weights/decompose.py`**:
  - Wire directly to `neuralops.core.low_rank.svd_linear` for SVD decomposition.
- [ ] **`neuralops/operations/weights/prune.py`**:
  - Connect to `neuralops.compression.pruning` structured 2:4 and group L1/L2 pruning.
- [ ] **`neuralops/operations/weights/compress.py`**:
  - Implement ASVD, FWSVD, and HNC-SVD dispatch on target linear/conv weights.
- [ ] **`neuralops/operations/architecture/modify.py`**:
  - Replace stub with FX-graph architecture mutation engine (`neuralops.surgery.fx_surgery`).
- [ ] **`neuralops/operations/adaptation/heal.py`**:
  - Wire to DoRA / directional adaptation healing (`neuralops.trainer.methods.lora`).
- [ ] **`neuralops/operations/adaptation/lora.py`**:
  - Wire parameter injection to `neuralops.core.peft.lora`.
- [ ] **`neuralops/operations/export/model.py`**:
  - Implement SafeTensors, ONNX, and GGUF serialization pipelines without stubs.

---

### 📂 `neuralops/integrations/` (32 files, 54 runtime hooks)
Integrations connect NeuralOps to external serving engines. If the external runtime is missing, they fail closed gracefully. In Wave 7, they must support mock/stub execution for headless CI:
- [ ] **`neuralops/integrations/vllm/` & `vllm_plugin.py`**:
  - Implement offline dummy vLLM worker runner so custom attention layers can be validated in CI without requiring physical CUDA vLLM binary.
  - Implement custom PagedAttention v2 kernel bridge.
- [ ] **`neuralops/integrations/sglang/`**:
  - Implement RadixAttention prefix caching bridge with `UnifiedHybridCache`.
- [ ] **`neuralops/integrations/tensorrt_llm.py`**:
  - Finalize TensorRT-LLM plugin registration bindings.
- [ ] **`neuralops/integrations/fla/adapter.py`**:
  - Add native Triton bindings for Flash-Linear-Attention state passing.

---

### 📂 `neuralops/layers/` (289 files, 29 cross-attention stubs)
Cross-attention is currently rejected with `NotImplementedError` across 10 subquadratic recurrent/SSM attention models:
- [ ] **`neuralops/layers/ssm/vspec.py`**:
  - Implement cross-attention KV projection projection adaptation or document mathematically why SSMs reject asymmetric query/context shapes.
- [ ] **`neuralops/layers/ssm/selective_ssm.py`**:
  - Implement cross-context state injection for encoder-decoder models.
- [ ] **`neuralops/layers/ssm/mamba_block.py` & `mamba_svd_flash.py`**:
  - Add cross-sequence conditioning layer.
- [ ] **`neuralops/layers/ssm/samba_moe.py`**:
  - Validate router gating when cross-sequence tokens are provided.

---

### 📂 `neuralops/adapters/` (8 files, 4 incomplete adapters)
- [ ] **`neuralops/adapters/hf_transformers.py`**:
  - Support automatic hook injection for HuggingFace `PreTrainedModel` without monkey-patching `__call__`.
- [ ] **`neuralops/adapters/peft_adapter.py`**:
  - Ensure 100% round-trip conversion between HuggingFace PEFT `LoraConfig` and NeuralOps `LoRAAdapter`.

---

### 📂 `neuralops/export/` (9 files, 7 export formats)
- [ ] **`neuralops/export/safetensors.py`**:
  - Support sharded SafeTensors exporting for models > 5GB.
- [ ] **`neuralops/export/onnx.py`**:
  - Fix dynamic shape export for subquadratic recurrent layers with state carryover.
- [ ] **`neuralops/export/gguf.py`**:
  - Implement quantization matrix mapping for llama.cpp compatible Q4_K_M and Q8_0.

---

### 📂 `neuralops/core/` (259 files, 14 low-level routines)
- [ ] **`neuralops/core/exceptions.py`**:
  - Audit and certify that all newly introduced exception classes inherit from `NeuralOpsError`.
- [ ] **`neuralops/core/autograd/`**:
  - Verify double-backward Hessian-Vector Products (HVP) autograd graph cleanliness without memory leak.

---

### 📂 `neuralops/benchmarks/` (278 files)
- [ ] **`neuralops/benchmarks/suites/`**:
  - Migrate legacy suites completely to `suites_v2/`.
- [ ] **`neuralops/benchmarks/cli.py`**:
  - Complete full deprecation cycle and delegate 100% to `neuralops/benchmarks/cli_v2/main.py`.

---

### 📂 `neuralops/ui/` (19 files)
- [ ] **`neuralops/ui/routes/`**:
  - Add WebSocket streaming endpoint for live benchmark telemetry visualizations.
- [ ] **`neuralops/ui/templates/`**:
  - Update benchmark comparison tables with KaTeX rendering for mathematical parity metrics.

---

## 2. WAVE 7 EXECUTION PHASES & AGENT ROSTER

Wave 7 will follow the strict worktree and branch isolation protocol established in `PROJECT_MASTER_TRUTH.md`:

| Subagent ID | Domain | Owned Directories / Files | Objective |
|---|---|---|---|
| **W7-1** | Operations: Weights & Surgery | `neuralops/operations/weights/`, `neuralops/operations/attention/` | Replace all `deferred to Phase 2` stubs with working implementations |
| **W7-2** | Operations: Arch & Adapt | `neuralops/operations/architecture/`, `neuralops/operations/adaptation/` | Complete FX-graph surgery and healing verbs |
| **W7-3** | Export Pipelines | `neuralops/export/`, `neuralops/operations/export/` | Sharded SafeTensors, ONNX dynamic shape, and GGUF export |
| **W7-4** | Integrations: Serving Mocks | `neuralops/integrations/vllm/`, `neuralops/integrations/sglang/` | Mock headless runtime execution for CI validation |
| **W7-5** | Layers: SSM Cross-Attention | `neuralops/layers/ssm/` | Audit and implement/certify cross-attention contracts |
| **W7-6** | Adapters & HF Bridge | `neuralops/adapters/` | PEFT round-trip and PreTrainedModel non-invasive hook injection |
| **W7-7** | Benchmarks Legacy Sunset | `neuralops/benchmarks/cli.py`, `neuralops/benchmarks/suites/` | Complete v2 migration and clean up legacy shims |
| **W7-8** | UI & Live Telemetry | `neuralops/ui/` | Live WebSockets and KaTeX telemetry comparison dashboard |

---

## 3. VERIFICATION & ACCEPTANCE CRITERIA
- [ ] Zero unhandled `IntegrationNotImplementedError` on supported internal formats.
- [ ] All 35 subpackages under `neuralops/` have 100% runnable unit and contract tests.
- [ ] `pytest tests/unit/ tests/contract/ tests/integration/` passes 100% green.
- [ ] `ruff check neuralops/` passes with 0 lint violations.
- [ ] All commits isolated to `wave7/<AGENT_ID>` branches and merged cleanly by the orchestrator.
