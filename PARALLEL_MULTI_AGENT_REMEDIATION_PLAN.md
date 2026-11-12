# NeuralOps 10-Agent Parallel Remediation & Execution Blueprint

**Authority:** [AGENTS.md](file:///Users/tahamajs/dev/Project/AGENTS.md)  
**Forensic Reference Dossier:** [FINAL_BRANCH_AUDIT_AND_REMEDIATION_REPORT.md](file:///Users/tahamajs/dev/Project/FINAL_BRANCH_AUDIT_AND_REMEDIATION_REPORT.md)  
**Package Alignment:** `v2.1.0`  
**Current Branch:** `feat/zoo-forensic-remediation`  
**Target Branch:** `main`  
**Execution Architecture:** 10 Dedicated Autonomous Subagents Running in Isolated Workspaces / Parallel Waves

---

## 1. Executive Multi-Agent Architecture

To systematically remediate all 324 files and 236 cataloged forensic defects identified in the 8,656-line master audit dossier without causing git merge conflicts or disrupting runtime contracts, execution is orchestrated across **10 Specialized Parallel Agents** divided into **4 Sequential Phased Waves**.

```mermaid
flowchart TD
    subgraph Wave 1: Core Layers & Mathematical Integrity
        A1["Agent 1: Attention Layers Specialist"]
        A2["Agent 2: Compressor Zoo Specialist"]
        A3["Agent 3: Core Surgery Architect"]
        A4["Agent 4: Distillation Engine & Cache Specialist"]
    end
    subgraph Wave 2: Systems, Kernels & Hardware Optimization
        A5["Agent 5: CUDA & Triton Systems Engineer"]
        A6["Agent 6: Hugging Face Hub Daemon Specialist"]
        A7["Agent 7: Benchmark 5-Format Persistence Auditor"]
    end
    subgraph Wave 3: Cluster Automation & Test Remediation
        A8["Agent 8: HPC Cluster Automation Specialist"]
        A9["Agent 9: Compression Pareto Test Engineer"]
        A10["Agent 10: Invariant Gatekeeper & Hygiene Auditor"]
    end
    subgraph Wave 4: Unified Multi-Zoo Reconciliation
        A1 --> G["Master Reconciliation & Invariant Verification"]
        A2 --> G
        A3 --> G
        A4 --> G
        A5 --> G
        A6 --> G
        A7 --> G
        A8 --> G
        A9 --> G
        A10 --> G
        G --> M["Production Merge into main"]
    end
```

---

## 2. Global Agent Operational Principles (Mandatory for All 10 Agents)

1. **Git Commit Format (Strict AGENTS.md §2):**
   ```bash
   git commit -m "<type>(<scope>): <message>" --author="taha <tahamaj4@gmail.com>"
   ```
2. **File Header Compliance (AGENTS.md §3):**
   Every modified or new Python or Shell file must begin with the repository MIT header:
   ```python
   #!/usr/bin/env python3
   # Copyright (c) 2024-2026, Taha Majlesi
   #
   # This source code is licensed under the MIT license found in the
   # LICENSE file in the root directory of this source tree.
   # For a list of all contributors, visit:
   #   https://github.com/tahamajs/NeuralOps/graphs/contributors
   ```
3. **Zero Silent Swallow Policy (AGENTS.md §0.2):**
   Never use `except Exception: pass`. Replace with structured domain logging (`logger.debug(...)`) or raise explicit domain exceptions (`SingularValueDecompositionError`, `LinearDecompositionError`).
4. **Zero Fake Metrics (AGENTS.md §0.2):**
   Never hardcode throughput, accuracy, perplexity, or singular values.
5. **No Broken Invariants (AGENTS.md §6):**
   115 Attention Keys, 82 Compressor Keys, 58 Triton Modules, 140 CUDA Packages, 94 Notebooks must remain 100% verified via `python3 scripts/verifiers/verify_invariants.py`.

---

## 3. Dedicated Agent Mandates & Phase-by-Phase Directives

---

### Agent 1: Attention Mechanisms & Architectural Layer Taxonomy Specialist
- **Workspace Scope:** `neuralops/layers/` (68 files)
- **Primary Reference:** Section 1 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-009`, `DEF-010`, `DEF-011`, `DEF-012`, `DEF-013`, `DEF-019`, `DEF-020`, `DEF-023`, `DEF-024`
- **Execution Phases:**
  - **Phase 1.1 (Causal Masking & Residual Padding):** Inspect `neuralops/layers/gated/gated_delta_net.py` and `neuralops/layers/linear/gla_attention.py`. Implement dynamic chunk padding for sequence lengths not divisible by 64 ($L \pmod{64} 
e 0$) and ensure causal masking holds across arbitrary lengths.
  - **Phase 1.2 (FP16 Clamping & Logit Overflows):** Inspect `neuralops/layers/fast_kernels/xattention.py` and `soa_attn.py`. Replace `-1e9` masks with `-65500.0` or `.clamp(min=-65500.0, max=65500.0)` to eliminate `-inf`/`NaN` under FP16 autocast.
  - **Phase 1.3 (Division-by-Zero Elimination):** Add `.clamp(min=1e-8)` to normalization terms in `vspec.py`, `block_transformer_hiba.py`, `bam_umoe.py`, and `reap_moe.py`.
  - **Phase 1.4 (In-Place Mutation Remediation):** Wrap parameter updates in `eat.py:545` and `soa_attn.py:567` with `with torch.no_grad():`.
  - **Phase 1.5 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/test_attention_numerics_and_gradients.py -v --tb=short
    git commit -m "fix(layers): enforce causal chunk padding, FP16 logit clamp, and zero-grad mutation guards" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 2: 82-Compressor Zoo Algorithms & Quantization Specialist
- **Workspace Scope:** `neuralops/compression/` (65 files)
- **Primary Reference:** Section 2 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-001`, `DEF-002`, `DEF-003`, `DEF-004`, `DEF-007`, `DEF-008`, `DEF-014`, `DEF-015`, `DEF-016`, `DEF-017`, `DEF-018`, `DEF-021`, `DEF-022`
- **Execution Phases:**
  - **Phase 2.1 (Silent Swallow Elimination):** Remove bare `except Exception: pass` from `grecodec_compressor.py:130`, `mlorq_compressor.py`, `modegpt_compressor.py`, and `optimal_subspace_projection.py`. Raise explicit `CompressionVerificationError`.
  - **Phase 2.2 (Division-by-Zero in Pruners):** Inspect `wanda.py` and `smoothquant.py`. Guard activation norm reductions $\|X_j\|_2$ with `.clamp(min=1e-8)`.
  - **Phase 2.3 (In-Place Mutation Remediation):** Wrap `.zero_()` and `.mul_()` calls in `frandi_compressor.py:191`, `svd_llm_compressor.py:136`, and `uniql_compressor.py:410` inside `with torch.no_grad():`.
  - **Phase 2.4 (Physical Reduction Contract):** Assert $R > 1.05	imes$ or non-zero sparsity across all 82 compressors.
  - **Phase 2.5 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/compression/group1/ -v --tb=short
    git commit -m "fix(compression): eliminate silent swallows, add norm division guards, and wrap parameter mutations in no_grad" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 3: Core Surgery Protocols & Model Transformation Architect
- **Workspace Scope:** `neuralops/models/`, `neuralops/core/`, `neuralops/pipeline_transform/` (9 files)
- **Primary Reference:** Section 3 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-025`, `DEF-026`, `DEF-031`
- **Execution Phases:**
  - **Phase 3.1 (Atomic Rollback Engine):** In `neuralops/models/surgery.py`, wrap `transplant_model_attention` with pre-surgery state capture. On forward/backward gate failure, restore state and raise `SurgeryTransactionError`.
  - **Phase 3.2 (FP32 Softmax Accumulation):** In `neuralops/models/surgical_io.py:145`, compute attention weights using `torch.softmax(logits.float(), dim=-1).to(logits.dtype)` to prevent half-precision underflow.
  - **Phase 3.3 (Device & Dtype Preservation):** Fix `.to(device)` calls in `pipeline_transform/snapshot.py:278` to specify `.to(device=..., dtype=...)`.
  - **Phase 3.4 (Legacy File Deprecation):** Convert `neuralops/core/_attention_surgery.py` into a thin wrapper delegating to `neuralops.models.surgery`.
  - **Phase 3.5 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/test_surgical_delta_and_hub_loader.py -v --tb=short
    git commit -m "refactor(surgery): transactional rollback safety, FP32 softmax headroom, and legacy shim unification" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 4: Six-Pillar Knowledge Distillation & Teacher Cache Specialist
- **Workspace Scope:** `neuralops/benchmarks/kd/` (5 files)
- **Primary Reference:** Section 4 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-005`, `DEF-006`, `DEF-028`, `DEF-029`, `DEF-032`
- **Execution Phases:**
  - **Phase 4.1 (Spin Lock Elimination):** In `neuralops/benchmarks/kd/teacher_cache.py:401, 610`, insert `time.sleep(0.05)` inside `while True:` polling loops to prevent 100% CPU starvation.
  - **Phase 4.2 (GPU Synchronization Stall Removal):** Remove `torch.cuda.empty_cache()` from inside batch iteration loops in `distill_engine.py:342, 481, 533` and `runner.py:153`. Move cleanup strictly to inter-key boundaries.
  - **Phase 4.3 (Masked Representation Loss):** Ensure `distill_engine.py` applies `attention_mask` when computing token-level MSE to prevent learning padding geometry.
  - **Phase 4.4 (Hardcoded Path Parameterization):** Replace `/home/tahamajlesi.ut/NeuralOps/...` fallbacks with `os.environ.get("NEURALOPS_CACHE_DIR", ...)`.
  - **Phase 4.5 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/test_teacher_cache_utils.py -v --tb=short
    git commit -m "perf(kd): eliminate CPU spin-locks, remove intra-epoch empty_cache stalls, and parameterize cache paths" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 5: Native CUDA Autograd Extensions & Triton JIT Systems Engineer
- **Workspace Scope:** `neuralops/ops/` (12 files)
- **Primary Reference:** Section 5 of Master Audit Dossier
- **Defect IDs Assigned:** Autograd contiguity, Triton 3.0 block pointers, 128-bit tile alignments
- **Execution Phases:**
  - **Phase 5.1 (Tensor Contiguity Assertions):** In `neuralops/ops/cuda/direct_svd/autograd.py` and `sparsegpt/autograd.py`, enforce `x = x.contiguous()` prior to launching CUDA kernels.
  - **Phase 5.2 (Triton SRAM Tile Alignments):** Inspect `neuralops/ops/triton/low_rank.py` and `fused_svd_projection_triton.py`. Ensure block sizes (`BLOCK_M`, `BLOCK_N`) adhere to 128-bit memory coalescing guidelines.
  - **Phase 5.3 (Three-Tier Fallback Validation):** Validate that CUDA -> Triton -> PyTorch fallback logs exact error reasons and records `backend_used` truthfully.
  - **Phase 5.4 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/test_all_cuda_kernels_comprehensive.py -v --tb=short
    git commit -m "fix(ops): assert tensor contiguity in autograd functions and verify 128-bit Triton tile alignments" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 6: Hugging Face Hub Integration & Model Card Specialist
- **Workspace Scope:** `neuralops/hub/` (6 files)
- **Primary Reference:** Section 6 of Master Audit Dossier
- **Defect IDs Assigned:** HTTP 429 rate limit backoff, standalone surgical delta metadata, YAML tags
- **Execution Phases:**
  - **Phase 6.1 (Exponential Retry Loop):** In `neuralops/hub/uploader.py`, implement an exponential backoff loop with jitter to handle HTTP 429 rate-limiting from the Hugging Face API.
  - **Phase 6.2 (Strict Token Security):** Ensure `HF_TOKEN` is strictly read from environment variables; reject hardcoded credentials.
  - **Phase 6.3 (Surgical Delta Packaging):** Verify `model_card.py` embeds accurate base model references, parameter reduction ratios, and SST-2 / WikiText-2 benchmark metrics.
  - **Phase 6.4 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/test_upgrade_hf_model_cards.py -v --tb=short
    git commit -m "feat(hub): add exponential backoff on HTTP 429 and enforce strict credential extraction" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 7: Benchmark Runners & 5-Format Persistence Auditor
- **Workspace Scope:** `scripts/benchmarks/` (23 files)
- **Primary Reference:** Section 7 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-027`, `DEF-030`
- **Execution Phases:**
  - **Phase 7.1 (Parquet Type Coercion):** In `scripts/benchmarks/export_dual_zoo_5formats.py`, sanitize all numerical fields (`accuracy`, `perplexity`, `effective_ratio`) to float with `np.nan` fallback to prevent `ArrowTypeError`.
  - **Phase 7.2 (Atomic Ledger Writes):** Ensure all ledger modifications write to a `.tmp` file before atomic POSIX rename (`tmp_path.replace(dest_path)`).
  - **Phase 7.3 (DataLoader Pin Memory):** In `scripts/benchmarks/bert_zoo_train_cached.py:79` and `bert_zoo_train_wiki.py:686`, add `pin_memory=True` and `non_blocking=True`.
  - **Phase 7.4 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/benchmarks/test_export_dual_zoo_5formats.py -v --tb=short
    git commit -m "fix(benchmarks): sanitize columnar parquet schemas, enforce atomic ledger writes, and enable pinned DMA transfers" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 8: HPC Cluster Automation & PBS Job Specialist
- **Workspace Scope:** `scripts/hpc/` (7 files)
- **Primary Reference:** Section 8 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-033`, `DEF-034`, `DEF-035`, `DEF-036`
- **Execution Phases:**
  - **Phase 8.1 (MIT License Compliance):** Prepend standard MIT license header comments to:
    - `scripts/hpc/hf_sync_upload_daemon.sh`
    - `scripts/hpc/launch_full_dual_zoo_pipeline.sh`
    - `scripts/hpc/launch_qwen_attention_zoo.sh`
    - `scripts/hpc/launch_qwen_then_compressors.sh`
  - **Phase 8.2 (Cluster Disk Quota Management):** In `bert_zoo_upload_hf.py`, add automated pruning of local uncompressed weights upon verified upload to Hugging Face Hub.
  - **Phase 8.3 (Process Lock Hygiene):** Ensure PBS launchers clean orphaned lock files on `/dev/shm` before starting new training chains.
  - **Phase 8.4 (Verification & Commit):**
    ```bash
    bash -n scripts/hpc/hf_sync_upload_daemon.sh
    bash -n scripts/hpc/launch_full_dual_zoo_pipeline.sh
    bash -n scripts/hpc/launch_qwen_attention_zoo.sh
    bash -n scripts/hpc/launch_qwen_then_compressors.sh
    git commit -m "chore(hpc): add MIT license headers, cluster quota cleanup, and process lock hygiene" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 9: Dedicated Compression Pareto Test Engineer
- **Workspace Scope:** `tests/compression/` (92 files)
- **Primary Reference:** Section 9 of Master Audit Dossier
- **Defect IDs Assigned:** Multi-rank Pareto monotonicity, autograd backward pass assertion, assert swallowing
- **Execution Phases:**
  - **Phase 9.1 (Assert Swallowing Remediation):** In `tests/compression/group1/test_group1_suite.py:165` and `group2_suite.py:157`, remove `try ... except: pass` around `effective_ratio` assertions.
  - **Phase 9.2 (Monotonicity Across Ranks):** Verify that decreasing ranks `[256, 128, 64, 32]` yield strictly decreasing parameter counts and increasing reduction ratios.
  - **Phase 9.3 (Autograd Gradient Gate):** Add backward pass checks (`loss.backward()`) on compressed layers to confirm gradient flow to remaining trainable parameters.
  - **Phase 9.4 (Verification & Commit):**
    ```bash
    python3 -m pytest tests/compression/group1/ tests/compression/group2/ -v --tb=short
    git commit -m "test(compression): enforce strict Pareto monotonicity assertions and autograd backward verification" --author="taha <tahamaj4@gmail.com>"
    ```

---

### Agent 10: System Integration, Invariant Gatekeeper & Repository Hygiene Auditor
- **Workspace Scope:** `tests/*` (non-compression), `scripts/verifiers/`, scratch scripts (31 files)
- **Primary Reference:** Section 10 of Master Audit Dossier
- **Defect IDs Assigned:** `DEF-037`, `DEF-038`, `DEF-039`, `DEF-040`
- **Execution Phases:**
  - **Phase 10.1 (Scratch Script Purging):** Permanently remove temporary debug scripts:
    ```bash
    git rm scripts/test_cuda_ld.py scripts/test_qwen_gpu.py
    ```
  - **Phase 10.2 (Diagnostic Probe Relocation):** Move diagnostic scripts to `tests/diagnostics/`:
    ```bash
    mkdir -p tests/diagnostics
    git mv tests/probe_group1.py tests/probe_group2.py tests/probe_group3.py tests/probe_group4.py tests/probe_group5.py tests/diagnose_group3_individual.py tests/diagnose_gla_magnitude.py tests/diagnostics/
    ```
  - **Phase 10.3 (Changelog Alignment):** Update `CHANGELOG.md` under `[Unreleased]` documenting:
    - In-memory fast deep-copy runner for TinyLlama.
    - Six-zoo canonical 5-format persistence engine.
    - Asynchronous background Hugging Face uploader with retry logic.
    - PyTorch version-tolerant `GradScaler` fallback wrapper.
  - **Phase 10.4 (Invariant & Version Gate):**
    ```bash
    python3 scripts/verifiers/verify_invariants.py
    python3 scripts/utils/sync_version.py --check
    git commit -m "chore(hygiene): purge scratch scripts, quarantine diagnostic probes, and align changelog with v2.1.0" --author="taha <tahamaj4@gmail.com>"
    ```

---

## 4. Multi-Agent Conflict Prevention Matrix

To eliminate git merge conflicts during parallel execution:
| Agent | Exclusive Directory Path | Shared Files / Interfaces | Conflict Resolution Rule |
| :---: | :--- | :--- | :--- |
| **Agent 1** | `neuralops/layers/` | `ATTENTION_REGISTRY` | Read-only access to registry; only mutate layer subclasses. |
| **Agent 2** | `neuralops/compression/` | `COMPRESSOR_REGISTRY` | Read-only access to registry; only mutate compressor subclasses. |
| **Agent 3** | `neuralops/models/` & `core/` | Model Surgery Facade | Sole writer to `surgery.py` and `snapshot.py`. |
| **Agent 4** | `neuralops/benchmarks/kd/` | Distillation Engine | Sole writer to `distill_engine.py` and `teacher_cache.py`. |
| **Agent 5** | `neuralops/ops/` | CUDA & Triton Kernels | Sole writer to `ops/cuda/` and `ops/triton/`. |
| **Agent 6** | `neuralops/hub/` | Hub Uploaders & Cards | Sole writer to `hub/uploader.py` and `hub/model_card.py`. |
| **Agent 7** | `scripts/benchmarks/` | Benchmark Runners | Sole writer to `export_dual_zoo_5formats.py`. |
| **Agent 8** | `scripts/hpc/` | Cluster Launchers | Sole writer to shell scripts under `scripts/hpc/`. |
| **Agent 9** | `tests/compression/` | Compression Test Suites | Sole writer to test files under `tests/compression/`. |
| **Agent 10** | `tests/` root & `scripts/verifiers/` | Repository Invariants | Sole agent authorized to execute `git rm` on scratch scripts. |

---

## 5. Master Convergence & Production Verification Protocol

Once all 10 agents finish their wave phases, the final reconciliation sequence is physically executed on the unified branch:

```bash
# 1. Authoritative Invariants Audit (Non-Negotiable)
python3 scripts/verifiers/verify_invariants.py

# 2. Package Version Synchronization Check
python3 scripts/utils/sync_version.py --check

# 3. Canonical 5-Format Persistence Test Gate
python3 -m pytest tests/benchmarks/test_export_dual_zoo_5formats.py -v --tb=short

# 4. Attention Numerics & Autograd Backward Gate
python3 -m pytest tests/test_attention_numerics_and_gradients.py -v --tb=short

# 5. Full Repository Status & Git Hygiene
git status --short
git diff --check
```

---
*Blueprint certified by NeuralOps Autonomous Multi-Agent Orchestrator — 2026-11-12.*
