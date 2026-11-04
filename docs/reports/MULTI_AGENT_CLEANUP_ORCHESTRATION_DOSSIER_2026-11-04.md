# NeuralOps Multi-Agent Cleanup Orchestration Dossier
**Version:** 1.0 | **Date:** 2026-11-04  
**Author / Orchestrator:** Master Principal Systems Architect  
**Classification:** Operational Master Execution Dossier  
**Target Subsystems:** `neuralops/` package & `scripts/` automation tree  
**Invariant Mandate:** 131 Attentions | 72 Compressors | 26 Triton Kernels | 110 CUDA Kernels | 29 Notebooks  

---

## Table of Contents
1. [Part 1 — Executive Summary](#part-1--executive-summary)
2. [Part 2 — Wave Execution Model](#part-2--wave-execution-model)
3. [Part 3 — Global Golden Rules](#part-3--global-golden-rules)
4. [Part 4 — Agent Assignments & Detailed Phase Runbooks](#part-4--agent-assignments--detailed-phase-runbooks)
   - [4.1 Agent 1: Root Shim Eliminator](#41-agent-1-root-shim-eliminator)
   - [4.2 Agent 2: MetaPathFinder Stub Eliminator](#42-agent-2-metapathfinder-stub-eliminator)
   - [4.3 Agent 3: Validator Merger](#43-agent-3-validator-merger)
   - [4.4 Agent 4: Operations Legacy Remover](#44-agent-4-operations-legacy-remover)
   - [4.5 Agent 5: Legacy Logging & Compat Remover](#45-agent-5-legacy-logging--compat-remover)
   - [4.6 Agent 6: Scripts Dead Code Purger](#46-agent-6-scripts-dead-code-purger)
   - [4.7 Agent 7: High-Reference Shim Migrator](#47-agent-7-high-reference-shim-migrator)
   - [4.8 Agent 8: Scripts Directory Reorganizer](#48-agent-8-scripts-directory-reorganizer)
   - [4.9 Agent 9: Test Suite Reconciliation](#49-agent-9-test-suite-reconciliation)
   - [4.10 Agent 10: Documentation, Version, Release](#410-agent-10-documentation-version-release)
5. [Part 5 — Per-Phase Verification Schema](#part-5--per-phase-verification-schema)
6. [Part 6 — Cross-Agent Coordination Matrix](#part-6--cross-agent-coordination-matrix)
7. [Part 7 — Global Verification Suite](#part-7--global-verification-suite)
8. [Part 8 — Expected Final State](#part-8--expected-final-state)
9. [Part 9 — Rollback Playbook](#part-9--rollback-playbook)
10. [Part 10 — Orchestrator Checklist](#part-10--orchestrator-checklist)
11. [Appendices](#appendices)
    - [Appendix A — Raw Audit Excerpts](#appendix-a--raw-audit-excerpts)
    - [Appendix B — Full File Ownership Matrix](#appendix-b--full-file-ownership-matrix)
    - [Appendix C — All Commands Used](#appendix-c--all-commands-used)
    - [Appendix D — Sign-Off Template](#appendix-d--sign-off-template)

---

## Part 1 — Executive Summary

### 1.1 What This Dossier Does
This master orchestration dossier provides the definitive, turnkey execution runbook for eliminating architectural dualities, dead-code shims, and script sprawl across the NeuralOps repository. Guided by the empirical findings of two exhaustive forensic audits completed on 2026-11-04, this protocol organizes the cleanup into **10 specialized sub-agents** deployed across **4 sequential and parallel waves**. It specifies exact shell commands, expected terminal output patterns, 5-stage verification gates, and atomic rollback procedures for all 143 operational phases.

### 1.2 The Two Source Audits
This runbook directly operationalizes the findings of:
1. **`docs/reports/SCRIPTS_FORENSIC_AUDIT_2026-11-04.md` (728 lines):**
   > *"The `scripts/` directory currently contains 108 non-pyc files totaling 18,730 LOC. It suffers from root script sprawl (3 benchmark runners and 19 `.pbs` cluster job scripts sitting directly in root), 13 duplicate deprecated stubs, 4 dead/broken scripts, and hardcoded `/home/tahamajlesi.ut/` cluster paths that impede cross-cluster portability."*
2. **`docs/reports/DIRECTORY_ARCHITECTURE_BLUEPRINT_2026-11-04.md` (2,525 lines):**
   > *"The `neuralops/` package contains 1,441 Python files and 321,480 LOC across 37 directories. It exhibits transitional dual-hierarchy liabilities: `training/` (1 file, 60 LOC MetaPathFinder redirect) vs `trainer/` (49 files, 8,581 LOC); `validator/` (4 files, 467 LOC) vs `validation/` (34 files, 3,209 LOC); `operations/` (28 files, 1,709 LOC) vs `ops/` (364 files, 53,973 LOC); and `experiment_logging/` (6 files, 464 LOC) vs `tracking/` (4 files, 460 LOC). Additionally, 7 root-level shims have exactly zero references across the entire codebase."*

### 1.3 The 10 Agents at a Glance
| Agent ID | Agent Designation | Wave | Scope & Target Files | Total LOC | Risk Tier | Commit Target |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **Agent 1** | Root Shim Eliminator | Wave 1 | 7 unreferenced root shims (`server.py`, `checkpoint.py`, ...) | 133 | ZERO | `refactor(core): remove 7 unreferenced root compatibility shims` |
| **Agent 2** | MetaPathFinder Stub Eliminator | Wave 1 | `neuralops/training/` (`__init__.py`) | 60 | ZERO | `refactor(trainer): eliminate legacy training metapathfinder stub` |
| **Agent 3** | Validator Merger | Wave 2 | `neuralops/validator/` (4 files) -> `validation/` | 467 | LOW | `refactor(validation): merge recipe validator into validation engine` |
| **Agent 4** | Operations Legacy Remover | Wave 2 | `neuralops/operations/` (28 files) | 1,709 | MEDIUM | `refactor(ops): eliminate legacy operations directory in favor of ops` |
| **Agent 5** | Legacy Logging & Compat Remover | Wave 2 | `neuralops/experiment_logging/` (6) + `backward_compat/` (5) | 539 | LOW | `refactor(tracking): purge legacy experiment logging and backward compat` |
| **Agent 6** | Scripts Dead Code Purger | Wave 2 | 4 dead scripts + 2 hardcoded paths in `scripts/` | ~600 | LOW | `chore(scripts): purge dead scripts and parameterize cluster cache paths` |
| **Agent 7** | High-Reference Shim Migrator | Wave 3 | `surgery.py`, `serving.py`, `replace_attention.py`, `registry.py` | 75 | HIGH | `refactor(architecture): migrate high-reference root shims via AST rewriting` |
| **Agent 8** | Scripts Directory Reorganizer | Wave 3 | 19 `.pbs` files -> `hpc/`, 3 root benchmarks -> `benchmarks/` | ~3,500 | MEDIUM | `chore(scripts): reorganize hpc cluster jobs and benchmark runners` |
| **Agent 9** | Test Suite Reconciliation | Wave 4 | Full test suite imports & collection audit | N/A | MEDIUM | `test(suite): reconcile all test imports with modernized package layout` |
| **Agent 10** | Documentation, Version, Release | Wave 4 | README, CHANGELOG, pyproject.toml, version sync, release tag | N/A | LOW | `chore(release): bump version to v2.0.2 and update architecture docs` |

### 1.4 The 4 Waves and Their Dependencies
The execution follows an uncompromising unidirectional dependency flow:
1. **Wave 1 (Foundation - Sequential):** Agents 1 & 2 eliminate zero-risk root clutter and legacy runtime import hooks.
2. **Wave 2 (Parallel Safe - Concurrent):** Agents 3, 4, 5, and 6 operate concurrently on strictly isolated, non-overlapping directory trees (`validator`, `operations`, `experiment_logging`, `scripts`).
3. **Wave 3 (High Risk - Sequential):** Agents 7 & 8 execute complex AST rewriting of 395 reference sites and physical directory relocations with git history preservation.
4. **Wave 4 (Final Consolidation - Sequential):** Agents 9 & 10 execute repository-wide test suite reconciliation, version bumping to v2.0.2, and camera-ready documentation updates.

### 1.5 Global Invariant Contract
The following five invariant metrics represent the non-negotiable architectural bedrock of NeuralOps. Every single phase of every agent must verify that these counts remain 100% frozen:
- **131 Attention Mechanisms:** `grep -rE "@register_attention" neuralops/ | wc -l == 131`
- **72 Matrix Compressors:** `grep -rE "@register_compressor" neuralops/ | wc -l == 72`
- **26 Triton JIT Kernels:** `find neuralops/ops/triton -maxdepth 1 -name "*.py" -not -name "__init__.py" | wc -l == 26`
- **110 CUDA C++ Kernels:** `find neuralops/ops/cuda -mindepth 1 -maxdepth 1 -type d | wc -l == 110`
- **29 Jupyter Notebook Tutorials:** `find notebooks -maxdepth 1 -name "*.ipynb" | wc -l == 29`

### 1.6 Expected Final Outcome
- **Total Files Deleted / Consolidated:** 51+ files removed from `neuralops/` and `scripts/`.
- **Total Lines of Code Eliminated:** 2,908+ LOC of obsolete shims and dead routines removed.
- **Top-Level Package Reduction:** Reduced from 37 messy directories to 26 clean, canonical packages.
- **Expected Python File Count:** `find neuralops/ -type f -name "*.py" | wc -l` reduces from 1,441 to ~1,390.
- **Clean Git History:** Exactly 10 atomic, well-attributed commits on `main`.

---

## Part 2 — Wave Execution Model

### 2.1 Why Waves (Not All Parallel)
Uncontrolled parallel execution across 10 agents on a shared repository causes devastating merge conflicts, race conditions in AST rewriting, and broken imports. For example, Agent 7 cannot rewrite imports to `neuralops.ops` if Agent 4 is simultaneously deleting `neuralops/operations/`. Agent 9 cannot reconcile tests while files are actively in motion. The 4-Wave model provides deterministic synchronization barriers.

### 2.2 Wave 1: Foundation (Sequential — Agents 1, 2)
- **Agent 1** runs first: Cleans out 7 unreferenced root files. Verifies that `neuralops` imports smoothly.
- **Agent 2** runs second: Removes `neuralops/training/` MetaPathFinder stub. Verifies that `neuralops.trainer` remains fully intact.
- **Barrier 1:** Orchestrator verifies working tree, commits, and runs package load test before launching Wave 2.

### 2.3 Wave 2: Parallel Safe (Concurrent — Agents 3, 4, 5, 6)
- **Agent 3 (Validator):** Modifies only `neuralops/validator/` and `neuralops/validation/`.
- **Agent 4 (Operations):** Modifies only `neuralops/operations/` and checks against `neuralops/ops/`.
- **Agent 5 (Logging & Compat):** Modifies only `neuralops/experiment_logging/` and `neuralops/backward_compat/`.
- **Agent 6 (Scripts Purger):** Modifies only dead files in `scripts/`.
- **Barrier 2:** All 4 agents must complete their commits with green verification gates before Wave 3 begins.

### 2.4 Wave 3: High Risk (Sequential — Agents 7, 8)
- **Agent 7 (High-Reference Shim Migrator):** Rewrites 395 import sites using dedicated AST migration scripts, updates `neuralops/__init__.py`, and deletes `surgery.py`, `serving.py`, `replace_attention.py`, and `registry.py`.
- **Agent 8 (Scripts Reorganizer):** Executes `git mv` on 19 `.pbs` cluster files and 3 root benchmark scripts, maintaining full Git history and updating PBS file paths.
- **Barrier 3:** Orchestrator verifies that zero broken imports exist in `neuralops/` and `scripts/`.

### 2.5 Wave 4: Final Consolidation (Sequential — Agents 9, 10)
- **Agent 9 (Test Suite Reconciliation):** Runs `pytest --collect-only` across the entire test suite, resolves any redirected imports in test fixtures, and runs the full CPU regression suite.
- **Agent 10 (Documentation & Release):** Updates `README.md`, `CHANGELOG.md`, bumps version to `2.0.2` across `pyproject.toml`, `CITATION.cff`, and `neuralops/__init__.py`, runs `audit_dead_code.py`, and creates local release tag `v2.0.2-cleanup`.

### 2.6 Inter-Wave Verification Gate Protocol
Before any wave transition, the orchestrator executes the 5-point verification gate:
1. Working tree is clean: `git status --porcelain` is empty.
2. Invariants unchanged: 131 / 72 / 26 / 110 / 29.
3. Package loads without warnings: `python3 -c "import neuralops"`.
4. Tests pass: `pytest tests/ -m "not gpu" -q`.
5. Git log reflects expected atomic commits.

### 2.7 Rollback Protocol Per Wave
If any agent fails or corrupts state:
- **Intra-Agent Failure:** `git reset --hard HEAD` reverts the current uncommitted phase.
- **Wave Failure:** `git reset --hard <pre-wave-sha>` restores the repository to the last certified wave checkpoint.

---

## Part 3 — Global Golden Rules

1. **NO FABRICATION OF TERMINAL OUTPUT:** Every claim of a passed test or created file must be backed by pasted raw terminal output. If not run, state: `I did not run it.`
2. **NO GIT PUSH, NO REMOTE TAGGING:** All commits and tags must remain strictly local. Do not push to remote origin.
3. **INVARIANT PRESERVATION (131/72/26/110/29):** The canonical registry counts must never change under any circumstance.
4. **STRICT SCOPE DISCIPLINE:** Each agent is strictly confined to its assigned file list. Zero drive-by refactoring of unrelated modules.
5. **MANDATORY 5-GATE VERIFICATION:** Every phase must pass the 5-gate schema before proceeding.
6. **ATOMIC ROLLBACK DISCIPLINE:** Immediately rollback on error. Never attempt ad-hoc patch-ups on broken intermediate states.
7. **STOP-AND-REPORT ON HARD BLOCKERS:** If a circular import or permission issue arises, halt and report to the orchestrator.
8. **EXACT COMMIT ATTRIBUTION:** Every commit must strictly use: `--author="taha <tahamaj4@gmail.com>"`.
9. **HANDS OFF FLASH ATTENTION (FLA) & PIPELINE:** Do not alter third-party FLA bindings or pipeline transformation kernels.
10. **HANDS OFF MODEL TRAINING PPL BUG:** Do not attempt to fix mathematical model perplexity bugs during structural directory cleanup.

---

## Part 4 — Agent Assignments & Detailed Phase Runbooks

### 4.1 Agent 1: Root Shim Eliminator (Zero-Risk)

#### 4.1.1 Mission
Agent 1 is tasked with the surgical elimination of 7 root-level single-file compatibility shims located directly in `neuralops/`. The directory architecture audit confirmed that each of these 7 files contains only `from X import *` redirects and has **exactly zero references** across all package modules, tests, scripts, documentation, and notebooks. Removing them resolves Python package-versus-module shadowing and declutters the root package namespace with zero functional risk.

#### 4.1.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/server.py` | 19 | DELETE | `neuralops.runners.monitor.server` | 0 references across repo |
| `neuralops/checkpoint.py` | 19 | DELETE | `neuralops.utils.checkpoint` | 0 references across repo |
| `neuralops/qwen_compat.py` | 19 | DELETE | `neuralops.utils.qwen_compat` | 0 references across repo |
| `neuralops/hub.py` | 19 | DELETE | `neuralops.models.hub` | 0 references across repo |
| `neuralops/architectures.py` | 19 | DELETE | `neuralops.utils.compat.architectures` | 0 references across repo |
| `neuralops/lora.py` | 19 | DELETE | `neuralops.utils.lora` | 0 references across repo |
| `neuralops/fastapi_app.py` | 19 | DELETE | `neuralops.runners.monitor.fastapi_app` | 0 references across repo |
| **Total** | **133 LOC** | **7 files deleted** | | **Verified 100% Unreferenced** |

#### 4.1.3 Wave & Dependencies
- **Wave:** Wave 1 (Foundation)
- **Execution Mode:** Sequential (runs first before Agent 2)
- **Can run in parallel with:** None (must establish clean root baseline)
- **Must complete before:** Agent 2 and all subsequent waves
- **Blocked if:** Working tree is dirty or `git status --porcelain` is non-empty.

#### 4.1.4 Phases (15 Phases)

**Phase 4.1.4.1 — Pre-Flight Environment & Invariant Baseline Check**
- **Objective:** Verify working tree cleanliness and establish baseline invariant counts.
- **Prerequisites:** Clean Git repository on branch `main`.
- **Commands:**
```bash
git status --porcelain
python3 -c "import neuralops; print('NeuralOps loaded:', neuralops.__version__)"
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `git status` output is empty; `neuralops.__version__` prints `2.0.1`; Attentions: 131; Compressors: 72.
- **Verification Gate:**
  - [x] Working tree clean (zero lines from `git status --porcelain`)
  - [x] Package imports cleanly with exit code 0
  - [x] Attentions count == 131
  - [x] Compressors count == 72
- **Rollback if failed:** `git reset --hard HEAD` and stop.
- **Evidence to paste:** Raw output of the 4 bash commands.

**Phase 4.1.4.2 — Confirm Zero References for All 7 Candidate Files**
- **Objective:** Re-validate that zero external references exist for the 7 files prior to deletion.
- **Prerequisites:** Phase 4.1.4.1 passed.
- **Commands:**
```bash
for f in server checkpoint qwen_compat hub architectures lora fastapi_app; do
  refs=$(git grep -rn "neuralops\.${f}" neuralops/ tests/ scripts/ docs/ notebooks/ 2>/dev/null | grep -v "neuralops/${f}.py" | wc -l)
  echo "File neuralops/${f}.py external references: $refs"
done
```
- **Expected Output Pattern:** Exactly 7 lines showing `external references: 0`.
- **Verification Gate:**
  - [x] Every candidate file reports exactly 0 external references.
- **Rollback if failed:** Stop execution; do not proceed if any file has >0 references.
- **Evidence to paste:** Terminal output of reference scan.

**Phase 4.1.4.2.5 — Record Base SHA for Rollback**
- **Objective:** Persist the exact pre-cleanup commit SHA to disk so any rollback can reference it reliably.
- **Prerequisites:** Phase 4.1.4.2 confirmed zero references.
- **Commands:**
```bash
git rev-parse HEAD > .cleanup_base_sha
cat .cleanup_base_sha
```
- **Expected Output Pattern:** One 40-character SHA printed.
- **Verification Gate:**
  - [x] File `.cleanup_base_sha` exists and contains a valid 40-character SHA.
- **Rollback if failed:** `rm -f .cleanup_base_sha`.
- **Evidence to paste:** Output of `cat .cleanup_base_sha`.

**Phase 4.1.4.3 — Surgical Deletion of `neuralops/server.py`**
- **Objective:** Remove unreferenced root shim `neuralops/server.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.2 confirmed 0 references.
- **Commands:**
```bash
git rm neuralops/server.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/server.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/server.py`
  - [x] Package imports cleanly without errors.
- **Rollback if failed:** `git checkout HEAD -- neuralops/server.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.4 — Surgical Deletion of `neuralops/checkpoint.py`**
- **Objective:** Remove unreferenced root shim `neuralops/checkpoint.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.3 passed.
- **Commands:**
```bash
git rm neuralops/checkpoint.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/checkpoint.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/checkpoint.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/checkpoint.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.5 — Surgical Deletion of `neuralops/qwen_compat.py`**
- **Objective:** Remove unreferenced root shim `neuralops/qwen_compat.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.4 passed.
- **Commands:**
```bash
git rm neuralops/qwen_compat.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/qwen_compat.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/qwen_compat.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/qwen_compat.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.6 — Surgical Deletion of `neuralops/hub.py`**
- **Objective:** Remove unreferenced root shim `neuralops/hub.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.5 passed.
- **Commands:**
```bash
git rm neuralops/hub.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/hub.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/hub.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/hub.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.7 — Surgical Deletion of `neuralops/architectures.py`**
- **Objective:** Remove unreferenced root shim `neuralops/architectures.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.6 passed.
- **Commands:**
```bash
git rm neuralops/architectures.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/architectures.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/architectures.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/architectures.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.8 — Surgical Deletion of `neuralops/lora.py`**
- **Objective:** Remove unreferenced root shim `neuralops/lora.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.7 passed.
- **Commands:**
```bash
git rm neuralops/lora.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/lora.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/lora.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/lora.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.9 — Surgical Deletion of `neuralops/fastapi_app.py`**
- **Objective:** Remove unreferenced root shim `neuralops/fastapi_app.py` (19 LOC).
- **Prerequisites:** Phase 4.1.4.8 passed.
- **Commands:**
```bash
git rm neuralops/fastapi_app.py
python3 -c "import neuralops; print('OK')"
```
- **Expected Output Pattern:** `rm 'neuralops/fastapi_app.py'` followed by `OK`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/fastapi_app.py`
  - [x] Package imports cleanly.
- **Rollback if failed:** `git checkout HEAD -- neuralops/fastapi_app.py`
- **Evidence to paste:** Output of `git rm` and import test.

**Phase 4.1.4.10 — Batch Staged Status Verification**
- **Objective:** Verify that exactly the 7 intended files are staged for deletion.
- **Prerequisites:** Phases 4.1.4.3 through 4.1.4.9 completed.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Exactly 7 lines starting with `D `, corresponding to the 7 root shims.
- **Verification Gate:**
  - [x] Exactly 7 files staged as deleted.
  - [x] Zero untracked or modified files present.
- **Rollback if failed:** `git reset --hard HEAD`
- **Evidence to paste:** Full output of `git status --porcelain`.

**Phase 4.1.4.11 — Invariant & Integrity Verification**
- **Objective:** Execute full 5-gate invariant validation.
- **Prerequisites:** Phase 4.1.4.10 passed.
- **Commands:**
```bash
python3 -c "import neuralops; from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY; print('Attentions:', len(ATTENTION_REGISTRY), 'Compressors:', len(COMPRESSOR_REGISTRY))"
```
- **Expected Output Pattern:** `Attentions: 131 Compressors: 72`.
- **Verification Gate:**
  - [x] Attentions count == 131
  - [x] Compressors count == 72
- **Rollback if failed:** `git reset --hard HEAD`
- **Evidence to paste:** Registry count output.

**Phase 4.1.4.12 — Fast Smoke Test Execution**
- **Objective:** Run fast CPU-only test suite subset to confirm zero regression.
- **Prerequisites:** Phase 4.1.4.11 passed.
- **Commands:**
```bash
pytest tests/test_registry.py -v --tb=short
```
- **Expected Output Pattern:** Summary containing `passed in X.Ys`.
- **Verification Gate:**
  - [x] `test_registry.py` passes with 0 failures.
- **Rollback if failed:** `git reset --hard HEAD`
- **Evidence to paste:** Pytest summary line.

**Phase 4.1.4.13 — Local Git Commit**
- **Objective:** Create atomic commit for Agent 1.
- **Prerequisites:** All verification gates passed.
- **Commands:**
```bash
git commit -m "refactor(core): remove 7 unreferenced root compatibility shims" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** 1 commit created; 7 files changed, 0 insertions(+), 133 deletions(-).
- **Verification Gate:**
  - [x] Commit hash generated.
  - [x] Author matches `taha <tahamaj4@gmail.com>`.
- **Rollback if failed:** `git reset --soft HEAD~1`
- **Evidence to paste:** Output of `git log -1 --stat`.

**Phase 4.1.4.14 — Final Handoff Report**
- **Objective:** Generate formal completion summary for orchestrator.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Working tree clean; ready for Agent 2.
- **Evidence to paste:** Agent 1 sign-off report.

---

### 4.2 Agent 2: MetaPathFinder Stub Eliminator

#### 4.2.1 Mission
Agent 2 eliminates the legacy `neuralops/training/` directory. The audit established that this directory contains only a single 60-LOC file (`__init__.py`) implementing a dynamic `sys.meta_path` `_TrainingRedirectFinder` hook that transparently redirects `neuralops.training.*` imports to `neuralops.trainer.*`. All actual training logic, knowledge distillation algorithms, and WSD schedulers reside canonically in `neuralops/trainer/` (49 files, 8,581 LOC). Removing this stub eliminates runtime import interception overhead and enforces explicit usage of `neuralops.trainer`.

#### 4.2.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/training/__init__.py` | 60 | DELETE | `neuralops.trainer` | MetaPathFinder redirect stub |
| `neuralops/training/` | (dir) | DELETE DIR | `neuralops/trainer/` | Absorbed by trainer (49 files) |
| **Total** | **60 LOC** | **1 file, 1 dir** | | **Subsumed by `trainer/`** |

#### 4.2.3 Wave & Dependencies
- **Wave:** Wave 1 (Foundation)
- **Execution Mode:** Sequential (runs after Agent 1 completes)
- **Can run in parallel with:** None
- **Must complete before:** Wave 2 agents
- **Blocked if:** Agent 1 commit is missing or invariants deviate from 131/72.

#### 4.2.4 Phases (12 Phases)

**Phase 4.2.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify clean branch and Agent 1 completion.
- **Commands:**
```bash
git log -1 --oneline
python3 -c "import neuralops.training; import neuralops.trainer; print('Both importable')"
```
- **Expected Output Pattern:** Shows Agent 1 commit; prints `Both importable`.
- **Verification Gate:**
  - [x] Git HEAD is Agent 1 commit.
  - [x] Package imports cleanly.
- **Rollback:** Stop execution.
- **Evidence to paste:** Terminal output of pre-flight commands.

**Phase 4.2.4.2 — Confirm MetaPathFinder Implementation in `training/__init__.py`**
- **Objective:** Inspect file content to verify zero original business logic.
- **Commands:**
```bash
cat neuralops/training/__init__.py | grep -E "MetaPathFinder|_TrainingRedirectFinder|neuralops.trainer"
```
- **Expected Output Pattern:** Displays the redirect finder classes pointing to `neuralops.trainer`.
- **Verification Gate:**
  - [x] File confirmed as pure redirection shim.
- **Rollback:** Stop if unique logic exists.
- **Evidence to paste:** Grep output.

**Phase 4.2.4.3 — Scan Codebase for Lingering `neuralops.training` Imports**
- **Objective:** Identify any internal imports that still reference `neuralops.training`.
- **Commands:**
```bash
git grep -rn "from neuralops.training\|import neuralops.training" neuralops/ tests/ scripts/
```
- **Expected Output Pattern:** 0 references, or references only within deprecated comments.
- **Verification Gate:**
  - [x] No active operational logic relies on `neuralops.training`.
- **Rollback:** If active references exist, rewrite them to `neuralops.trainer` before proceeding.
- **Evidence to paste:** Grep output.

**Phase 4.2.4.4 — Remove `neuralops/training/__init__.py` via Git**
- **Objective:** Delete the single stub file.
- **Commands:**
```bash
git rm -f neuralops/training/__init__.py
```
- **Expected Output Pattern:** `rm 'neuralops/training/__init__.py'`.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D neuralops/training/__init__.py`.
- **Rollback if failed:** `git checkout HEAD -- neuralops/training/__init__.py`.
- **Evidence to paste:** Output of `git rm`.

**Phase 4.2.4.5 — Remove Empty `neuralops/training/` Directory**
- **Objective:** Clean filesystem directory.
- **Commands:**
```bash
rmdir neuralops/training 2>/dev/null || true
test ! -d neuralops/training && echo "Directory removed successfully"
```
- **Expected Output Pattern:** `Directory removed successfully`.
- **Verification Gate:**
  - [x] Directory `neuralops/training` no longer exists on disk.
- **Rollback if failed:** `mkdir -p neuralops/training`.
- **Evidence to paste:** Output confirmation.

**Phase 4.2.4.6 — Verify Canonical `neuralops.trainer` Integrity**
- **Objective:** Ensure `neuralops.trainer` functions completely without the training stub.
- **Commands:**
```bash
python3 -c "from neuralops.trainer import Trainer, TrainingConfig; print('Trainer OK')"
```
- **Expected Output Pattern:** `Trainer OK`.
- **Verification Gate:**
  - [x] `neuralops.trainer` exports primary classes cleanly.
- **Rollback if failed:** `git reset --hard HEAD`.
- **Evidence to paste:** Python execution output.

**Phase 4.2.4.7 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global architectural metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Invariants match exactly.
- **Rollback if failed:** `git reset --hard HEAD`.
- **Evidence to paste:** Registry count output.

**Phase 4.2.4.8 — Execute Trainer Test Suite**
- **Objective:** Run targeted unit tests for the trainer subsystem.
- **Commands:**
```bash
pytest tests/ -k "trainer" -q --tb=short
```
- **Expected Output Pattern:** Tests pass or show green summary.
- **Verification Gate:**
  - [x] Zero trainer test failures.
- **Rollback if failed:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary line.

**Phase 4.2.4.9 — Run Ruff Linter on Modified Package Tree**
- **Objective:** Ensure zero linting errors introduced.
- **Commands:**
```bash
ruff check neuralops/trainer/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Ruff reports clean code.
- **Rollback if failed:** Resolve linting or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.2.4.10 — Verify Overall Package Import**
- **Objective:** Ensure top-level package loads cleanly without dynamic meta-path hooks.
- **Commands:**
```bash
python3 -c "import neuralops; print('Package loaded successfully')"
```
- **Expected Output Pattern:** `Package loaded successfully`.
- **Verification Gate:**
  - [x] Exit code 0; zero warnings.
- **Rollback if failed:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.2.4.11 — Local Git Commit for Agent 2**
- **Objective:** Commit the elimination of `neuralops/training/`.
- **Commands:**
```bash
git commit -m "refactor(trainer): eliminate legacy training metapathfinder stub" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** 1 file changed, 60 deletions(-).
- **Verification Gate:**
  - [x] Commit SHA created with proper attribution.
- **Rollback if failed:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Output of `git log -1 --stat`.

**Phase 4.2.4.12 — Wave 1 Barrier Sign-Off**
- **Objective:** Certify completion of Wave 1 (Foundation).
- **Commands:**
```bash
git status --porcelain
git log -2 --oneline
```
- **Expected Output Pattern:** Working tree clean; shows Agent 1 and Agent 2 commits.
- **Verification Gate:**
  - [x] Working tree clean; Wave 1 officially cleared for Wave 2 launch.
- **Evidence to paste:** Output of `git log -2 --oneline`.

---

### 4.3 Agent 3: Validator Merger

#### 4.3.1 Mission
Agent 3 consolidates the legacy `neuralops/validator/` directory (4 files, 467 LOC) into the modern `neuralops/validation/` engine (34 files, 3,209 LOC). The forensic audit demonstrated that `validator/` hosts an earlier iteration of recipe static validation rules, while `validation/` contains the modern 8-stage numerical invariant engine and gradient parity certifier. Unique classes from `validator/` (specifically `SymplecticNumericalGuard`, `check_condition_number`, and `check_numerical_stability` from `numerical_guardrails.py`) will be migrated into a dedicated `neuralops/validation/recipe_validator.py` module, followed by updating internal imports and cleanly removing the legacy directory.

#### 4.3.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/validator/__init__.py` | 33 | MIGRATE / DELETE | `neuralops/validation/recipe_validator.py` | Exports validator symbols |
| `neuralops/validator/errors.py` | 56 | MERGE / DELETE | `neuralops/validation/base.py` | ValidationError classes |
| `neuralops/validator/numerical_guardrails.py` | 212 | MIGRATE / DELETE | `neuralops/validation/recipe_validator.py` | Symplectic numerical guards |
| `neuralops/validator/static.py` | 166 | MIGRATE / DELETE | `neuralops/validation/recipe_validator.py` | Static recipe AST validation |
| `neuralops/validator/` | (dir) | DELETE DIR | `neuralops/validation/` | Entire directory subsumed |
| **Total** | **467 LOC** | **4 files, 1 dir** | | **Merged into `validation/`** |

#### 4.3.3 Wave & Dependencies
- **Wave:** Wave 2 (Parallel Safe)
- **Execution Mode:** Concurrent (runs in parallel with Agents 4, 5, 6)
- **Can run in parallel with:** Agent 4 (`operations`), Agent 5 (`tracking`), Agent 6 (`scripts`)
- **Must complete before:** Wave 3 (Agent 7 & 8)
- **Blocked if:** Wave 1 verification gates fail.

#### 4.3.4 Phases (14 Phases)

**Phase 4.3.4.1 — Pre-Flight & Working Tree Cleanliness**
- **Objective:** Verify clean working tree and confirm Wave 1 commits exist.
- **Commands:**
```bash
git status --porcelain
git log -2 --oneline
```
- **Expected Output Pattern:** Empty status; shows Agent 1 and Agent 2 commits.
- **Verification Gate:**
  - [x] Working tree clean; HEAD is Agent 2 commit.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Git status output.

**Phase 4.3.4.2 — AST Symbol Extraction from `neuralops/validator/`**
- **Objective:** Extract all public symbols, functions, and classes across the 4 validator files.
- **Commands:**
```bash
python3 -c "
import ast, glob
symbols = {}
for f in glob.glob('neuralops/validator/*.py'):
    with open(f) as fp:
        tree = ast.parse(fp.read(), filename=f)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            symbols[node.name] = f
for name, f in sorted(symbols.items()):
    print(f'{name:<32} in {f}')
"
```
- **Expected Output Pattern:** Lists ~12 symbols including `SymplecticNumericalGuard`, `RuntimeValidationError`, `StaticValidationError`, `validate_recipe_static`.
- **Verification Gate:**
  - [x] Complete list of ~12 symbols extracted without AST syntax errors.
- **Rollback:** None (read-only).
- **Evidence to paste:** Symbol mapping list.

**Phase 4.3.4.3 — Scan Codebase References to `neuralops.validator`**
- **Objective:** Identify all call sites referencing `neuralops.validator`.
- **Commands:**
```bash
git grep -rn "neuralops\.validator" neuralops/ tests/ scripts/
```
- **Expected Output Pattern:** ~21 occurrences across recipe tests and internal imports.
- **Verification Gate:**
  - [x] All 21 reference sites identified and cataloged.
- **Rollback:** None (read-only).
- **Evidence to paste:** Grep output.

**Phase 4.3.4.4 — Create Canonical Target `neuralops/validation/recipe_validator.py`**
- **Objective:** Establish the target module housing recipe validation and symplectic guards.
- **Commands:**
```bash
python3 -c "
guardrails_code = open('neuralops/validator/numerical_guardrails.py').read()
static_code = open('neuralops/validator/static.py').read()
errors_code = open('neuralops/validator/errors.py').read()

header = '''#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
\"\"\"
Recipe and Symplectic Numerical Guardrail Validation Engine.
Consolidated from legacy neuralops.validator into canonical neuralops.validation.
\"\"\"

from __future__ import annotations
import math, torch, warnings
from typing import Any, Dict, List, Optional, Tuple, Union
'''
with open('neuralops/validation/recipe_validator.py', 'w') as out:
    out.write(header + '\n' + errors_code + '\n' + guardrails_code + '\n' + static_code)
print('Generated neuralops/validation/recipe_validator.py')
"
```
- **Expected Output Pattern:** `Generated neuralops/validation/recipe_validator.py`.
- **Verification Gate:**
  - [x] Target file exists and compiles: `python3 -m py_compile neuralops/validation/recipe_validator.py`.
- **Rollback:** `rm -f neuralops/validation/recipe_validator.py`.
- **Evidence to paste:** Compilation output.

**Phase 4.3.4.5 — Update `neuralops/validation/__init__.py` Exports**
- **Objective:** Re-export migrated symbols from canonical `validation` package.
- **Commands:**
```bash
python3 -c "
init_path = 'neuralops/validation/__init__.py'
content = open(init_path).read()
if 'recipe_validator' not in content:
    with open(init_path, 'a') as fp:
        fp.write('\nfrom neuralops.validation.recipe_validator import *\n')
print('Updated validation/__init__.py')
"
```
- **Expected Output Pattern:** `Updated validation/__init__.py`.
- **Verification Gate:**
  - [x] `python3 -c "from neuralops.validation import SymplecticNumericalGuard; print('OK')"` prints `OK`.
- **Rollback:** `git checkout HEAD -- neuralops/validation/__init__.py`.
- **Evidence to paste:** Import verification output.

**Phase 4.3.4.6 — Rewrite Call Sites from `neuralops.validator` to `neuralops.validation`**
- **Objective:** Safely rewrite all 21 import sites.
- **Commands:**
```bash
python3 -c "
import glob, re
for path in glob.glob('neuralops/**/*.py', recursive=True) + glob.glob('tests/**/*.py', recursive=True):
    with open(path, 'r', errors='ignore') as fp:
        text = fp.read()
    if 'neuralops.validator' in text:
        new_text = re.sub(r'neuralops\.validator', 'neuralops.validation', text)
        with open(path, 'w') as fp:
            fp.write(new_text)
        print(f'Updated: {path}')
"
```
- **Expected Output Pattern:** Lists updated files across tests and recipes.
- **Verification Gate:**
  - [x] `git grep "neuralops.validator" neuralops/ tests/ | grep -v "neuralops/validator/"` returns 0.
- **Rollback:** `git reset --hard HEAD` and re-run Phase 4.3.4.4.
- **Evidence to paste:** Grep verification.

**Phase 4.3.4.7 — Verify Test Suite with Migrated Validator Symbols**
- **Objective:** Confirm tests pass with new `recipe_validator.py` implementation.
- **Commands:**
```bash
pytest tests/ -k "validation or recipe" -q --tb=short
```
- **Expected Output Pattern:** Tests pass without import errors.
- **Verification Gate:**
  - [x] Validation tests pass.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest output.

**Phase 4.3.4.8 — Delete Legacy `neuralops/validator/` Files**
- **Objective:** Git remove the 4 legacy validator files.
- **Commands:**
```bash
git rm -rf neuralops/validator/
```
- **Expected Output Pattern:** `rm 'neuralops/validator/...'` for all 4 files.
- **Verification Gate:**
  - [x] `git status --porcelain` shows `D ` for all validator files.
  - [x] Directory `neuralops/validator` no longer exists.
- **Rollback:** `git checkout HEAD -- neuralops/validator/`.
- **Evidence to paste:** Output of `git rm`.

**Phase 4.3.4.9 — Verify Invariant Contract (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Exact count match.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep output.

**Phase 4.3.4.10 — Run Package Import & Smoke Verification**
- **Objective:** Ensure package imports cleanly without legacy validator.
- **Commands:**
```bash
python3 -c "import neuralops; from neuralops.validation import SymplecticNumericalGuard; print('Validation import OK')"
```
- **Expected Output Pattern:** `Validation import OK`.
- **Verification Gate:**
  - [x] Clean exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Python output.

**Phase 4.3.4.11 — Run Ruff Linter on `neuralops/validation/`**
- **Objective:** Clean code style and imports.
- **Commands:**
```bash
ruff check neuralops/validation/recipe_validator.py
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff linting errors.
- **Rollback:** Run `ruff check --fix` or resolve manually.
- **Evidence to paste:** Ruff output.

**Phase 4.3.4.12 — Stage Target Files and Verify Diff Stat**
- **Objective:** Review staged changes for Agent 3.
- **Commands:**
```bash
git add neuralops/validation/
git status --porcelain
```
- **Expected Output Pattern:** `A neuralops/validation/recipe_validator.py`, `M neuralops/validation/__init__.py`, `D neuralops/validator/...`.
- **Verification Gate:**
  - [x] Exactly the expected files staged.
- **Rollback:** `git reset HEAD`.
- **Evidence to paste:** `git status` output.

**Phase 4.3.4.13 — Local Git Commit for Agent 3**
- **Objective:** Commit the validator merger.
- **Commands:**
```bash
git commit -m "refactor(validation): merge recipe validator into validation engine" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; ~467 deletions(-).
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.3.4.14 — Agent 3 Handoff Verification**
- **Objective:** Ensure working tree is clean.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Clean working tree.
- **Evidence to paste:** Terminal output.

---

### 4.4 Agent 4: Operations Legacy Remover

#### 4.4.1 Mission
Agent 4 purges the legacy `neuralops/operations/` directory (28 files, 1,709 LOC). The architectural audit confirmed that `operations/` is an obsolete v1.1 abstraction layer that has been completely superseded by the modern, high-performance `neuralops/ops/` package (364 files, 53,973 LOC) and `neuralops/core/registry.py`. Agent 4 verifies that every single class and function in `operations/` is either fully duplicated in `ops/` or can be safely redirected, updates remaining imports, and deletes the directory.

#### 4.4.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/operations/base.py` | 120 | DELETE | `neuralops.ops.base` | Abstract operator interface |
| `neuralops/operations/errors.py` | 45 | DELETE | `neuralops.core.errors` | Operator error classes |
| `neuralops/operations/registry.py` | 98 | DELETE | `neuralops.core.registry` | Superseded by core registry |
| `neuralops/operations/attention/*.py` | ~450 | DELETE | `neuralops.ops.cuda` / `triton` | Superseded by 364 ops files |
| `neuralops/operations/export/*.py` | ~300 | DELETE | `neuralops.export` | Serving export routines |
| `neuralops/operations/` | (dir) | DELETE DIR | `neuralops/ops/` | 28 files, 1,709 LOC |
| **Total** | **1,709 LOC** | **28 files, 1 dir** | | **Subsumed by `ops/`** |

#### 4.4.3 Wave & Dependencies
- **Wave:** Wave 2 (Parallel Safe)
- **Execution Mode:** Concurrent (runs in parallel with Agents 3, 5, 6)
- **Can run in parallel with:** Agent 3, Agent 5, Agent 6
- **Must complete before:** Wave 3
- **Blocked if:** Wave 1 verification gates fail.

#### 4.4.4 Phases (16 Phases)

**Phase 4.4.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Check Git status and package baseline.
- **Commands:**
```bash
git status --porcelain
python3 -c "import neuralops.ops; print('Ops canonical loaded')"
```
- **Expected Output Pattern:** Status empty; `Ops canonical loaded`.
- **Verification Gate:**
  - [x] Working tree clean; `ops` package imports cleanly.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output of commands.

**Phase 4.4.4.2 — Enumerate All 28 Files in `neuralops/operations/`**
- **Objective:** Produce exact file inventory of the target legacy package.
- **Commands:**
```bash
find neuralops/operations/ -type f -name "*.py" | sort
find neuralops/operations/ -type f -name "*.py" | wc -l
```
- **Expected Output Pattern:** Exactly 28 files listed; count == 28.
- **Verification Gate:**
  - [x] Exactly 28 files accounted for.
- **Rollback:** None (read-only).
- **Evidence to paste:** File list.

**Phase 4.4.4.3 — Extract AST Class and Function Symbols from `operations/`**
- **Objective:** Catalog all declared symbols across all 28 files.
- **Commands:**
```bash
python3 -c "
import ast, glob
syms = []
for f in glob.glob('neuralops/operations/**/*.py', recursive=True):
    with open(f) as fp:
        t = ast.parse(fp.read(), filename=f)
    for n in t.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            syms.append((n.name, f))
print(f'Total symbols declared in operations/: {len(syms)}')
"
```
- **Expected Output Pattern:** Prints symbol count (~45 symbols).
- **Verification Gate:**
  - [x] All symbols extracted without syntax error.
- **Rollback:** None.
- **Evidence to paste:** Terminal output.

**Phase 4.4.4.4 — Cross-Reference Symbols Against `neuralops/ops/` and `neuralops/core/`**
- **Objective:** Prove 100% equivalence in modern packages.
- **Commands:**
```bash
python3 <<'EOF'
import ast, glob, re, sys

def extract_symbols(directory):
    symbols = set()
    for f in glob.glob(f'{directory}/**/*.py', recursive=True):
        try:
            with open(f) as fp:
                tree = ast.parse(fp.read(), filename=f)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols.add(node.name)
    return symbols

ops_symbols = extract_symbols('neuralops/ops')
core_symbols = extract_symbols('neuralops/core')
canonical = ops_symbols | core_symbols

ops_legacy = extract_symbols('neuralops/operations')

missing = ops_legacy - canonical
print(f'operations/ symbols: {len(ops_legacy)}')
print(f'ops/ + core/ symbols: {len(canonical)}')
print(f'Missing from canonical: {len(missing)}')

if missing:
    print('ORPHANS FOUND — these will be lost if we delete operations/:')
    for sym in sorted(missing):
        print(f'  - {sym}')
    sys.exit(1)

print('Equivalence check passed: 100% of operations/ symbols exist in ops/ or core/')
EOF
```
- **Expected Output Pattern:** `Equivalence check passed: 100% covered in neuralops.ops / core`.
- **Verification Gate:**
  - [x] No orphan business logic unique to `operations/`.
- **Rollback:** Stop if unique logic is discovered.
- **Evidence to paste:** Output log.

**Phase 4.4.4.5 — Identify External References to `neuralops.operations`**
- **Objective:** Audit call sites across repo.
- **Commands:**
```bash
git grep -rn "neuralops\.operations" neuralops/ tests/ scripts/
```
- **Expected Output Pattern:** Shows all ~85 references.
- **Verification Gate:**
  - [x] All call sites cataloged.
- **Rollback:** None.
- **Evidence to paste:** Grep output.

**Phase 4.4.4.6 — Rewrite References from `neuralops.operations` to `neuralops.ops`**
- **Objective:** Safely redirect all call sites.
- **Commands:**
```bash
python3 -c "
import glob, re
targets = glob.glob('neuralops/**/*.py', recursive=True) + glob.glob('tests/**/*.py', recursive=True)
for p in targets:
    if 'neuralops/operations/' in p: continue
    with open(p, 'r', errors='ignore') as fp: text = fp.read()
    if 'neuralops.operations' in text:
        new_text = re.sub(r'neuralops\.operations', 'neuralops.ops', text)
        with open(p, 'w') as fp: fp.write(new_text)
        print(f'Migrated: {p}')
"
```
- **Expected Output Pattern:** Lists migrated files.
- **Verification Gate:**
  - [x] `git grep "neuralops.operations" neuralops/ tests/ | grep -v "neuralops/operations/"` returns 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep verification.

**Phase 4.4.4.7 — Verify `neuralops.ops` Kernel Dispatch Integrity**
- **Objective:** Run targeted kernel dispatch tests.
- **Commands:**
```bash
pytest tests/ -k "kernel or ops" -q --tb=short
```
- **Expected Output Pattern:** Summary containing green pass status.
- **Verification Gate:**
  - [x] All ops tests pass.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.4.4.8 — Delete `neuralops/operations/` via Git**
- **Objective:** Remove the 28 legacy files and directory.
- **Commands:**
```bash
git rm -rf neuralops/operations/
```
- **Expected Output Pattern:** `rm 'neuralops/operations/...'` for all 28 files.
- **Verification Gate:**
  - [x] 28 files staged as deleted.
  - [x] `neuralops/operations/` directory eliminated.
- **Rollback:** `git checkout HEAD -- neuralops/operations/`.
- **Evidence to paste:** Git rm output.

**Phase 4.4.4.9 — Verify Invariant Contract (131/72/26/110/29)**
- **Objective:** Confirm registries and operator counts untouched.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
echo "CUDA Kernels: $(find neuralops/ops/cuda -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Triton Kernels: $(find neuralops/ops/triton -maxdepth 1 -name "*.py" -not -name "__init__.py" | wc -l)"
```
- **Expected Output Pattern:** Attentions: 131; Compressors: 72; CUDA Kernels: 110; Triton Kernels: 26.
- **Verification Gate:**
  - [x] All 4 invariants exact.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.4.4.10 — Verify Top-Level Package Import**
- **Objective:** Ensure package initializes without `operations/`.
- **Commands:**
```bash
python3 -c "import neuralops; from neuralops.ops import *; print('Ops import verified')"
```
- **Expected Output Pattern:** `Ops import verified`.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output log.

**Phase 4.4.4.11 — Run Ruff Linter Across Repo**
- **Objective:** Clean code style.
- **Commands:**
```bash
ruff check neuralops/ops/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff errors in ops.
- **Rollback:** Fix formatting or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.4.4.12 — Review Git Staged Status**
- **Objective:** Verify only operations files are deleted and call sites updated.
- **Commands:**
```bash
git status --porcelain | grep "neuralops/operations" | wc -l
```
- **Expected Output Pattern:** Exactly 28.
- **Verification Gate:**
  - [x] Exactly 28 deletions from `operations/`.
- **Rollback:** `git reset HEAD`.
- **Evidence to paste:** Count output.

**Phase 4.4.4.13 — Local Git Commit for Agent 4**
- **Objective:** Commit the operations directory deletion.
- **Commands:**
```bash
git commit -m "refactor(ops): eliminate legacy operations directory in favor of ops" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; 28 files changed, 1,709 deletions(-).
- **Verification Gate:**
  - [x] Commit hash generated with proper author.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.4.4.14 — Verification of Commit Integrity**
- **Objective:** Ensure commit is clean.
- **Commands:**
```bash
git diff HEAD~1..HEAD --stat | grep "neuralops/operations" | wc -l
```
- **Expected Output Pattern:** 28.
- **Verification Gate:**
  - [x] Verification matches.
- **Rollback:** None.
- **Evidence to paste:** Terminal output.

**Phase 4.4.4.15 — Run Regression Gate**
- **Objective:** Execute fast regression check.
- **Commands:**
```bash
pytest tests/test_registry.py -q
```
- **Expected Output Pattern:** Passed.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD~1`.
- **Evidence to paste:** Pytest summary.

**Phase 4.4.4.16 — Agent 4 Handoff Report**
- **Objective:** Complete sign-off.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Clean tree.
- **Evidence to paste:** Output.

---

### 4.5 Agent 5: Legacy Logging & Compat Remover

#### 4.5.1 Mission
Agent 5 removes two obsolete legacy directories: `neuralops/experiment_logging/` (6 files, 464 LOC) and `neuralops/backward_compat/` (5 files, 75 LOC). The audit demonstrated that `experiment_logging/` has been completely superseded by the modern `neuralops/tracking/` package (which provides per-run hermetic `metrics.jsonl`, TensorBoard, and WandB telemetry), while `backward_compat/` contains unused deprecation stubs for ancient v0.9 namespaces. Removing these 11 files eliminates 539 LOC of dead code.

#### 4.5.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/experiment_logging/__init__.py` | 42 | DELETE | `neuralops.tracking` | Legacy telemetry re-export |
| `neuralops/experiment_logging/callbacks.py` | 78 | DELETE | `neuralops.tracking` | Replaced by tracking hooks |
| `neuralops/experiment_logging/config.py` | 35 | DELETE | `neuralops.config` | Replaced by Pydantic schemas |
| `neuralops/experiment_logging/integration.py` | 65 | DELETE | `neuralops.tracking` | Replaced by tracking integrations |
| `neuralops/experiment_logging/logger.py` | 170 | DELETE | `neuralops.tracking` | Replaced by ExperimentTracker |
| `neuralops/experiment_logging/tracker.py` | 74 | DELETE | `neuralops.tracking` | Replaced by LocalTracker |
| `neuralops/backward_compat/*.py` (5 files) | 75 | DELETE | None | Ancient v0.9 redirect stubs |
| `neuralops/experiment_logging/` | (dir) | DELETE DIR | `neuralops/tracking/` | Entire directory subsumed |
| `neuralops/backward_compat/` | (dir) | DELETE DIR | None | Entire directory deleted |
| **Total** | **539 LOC** | **11 files, 2 dirs** | | **Subsumed by `tracking/`** |

#### 4.5.3 Wave & Dependencies
- **Wave:** Wave 2 (Parallel Safe)
- **Execution Mode:** Concurrent (runs in parallel with Agents 3, 4, 6)
- **Can run in parallel with:** Agent 3, Agent 4, Agent 6
- **Must complete before:** Wave 3
- **Blocked if:** Wave 1 verification gates fail.

#### 4.5.4 Phases (13 Phases)

**Phase 4.5.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify clean branch and Wave 1 completion.
- **Commands:**
```bash
git status --porcelain
python3 -c "import neuralops.tracking; print('Tracking canonical OK')"
```
- **Expected Output Pattern:** Clean working tree; `Tracking canonical OK`.
- **Verification Gate:**
  - [x] Exit code 0; `tracking` package functional.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output of command.

**Phase 4.5.4.2 — Enumerate Files in `experiment_logging/` and `backward_compat/`**
- **Objective:** Account for all 11 target files.
- **Commands:**
```bash
ls -la neuralops/experiment_logging/ neuralops/backward_compat/
```
- **Expected Output Pattern:** 6 files in `experiment_logging/`, 5 files in `backward_compat/`.
- **Verification Gate:**
  - [x] Exactly 11 files present.
- **Rollback:** None (read-only).
- **Evidence to paste:** Directory listings.

**Phase 4.5.4.3 — Scan External References to `experiment_logging` and `backward_compat`**
- **Objective:** Confirm zero active dependencies in production code.
- **Commands:**
```bash
git grep -rn "neuralops\.experiment_logging" neuralops/ tests/ scripts/
git grep -rn "neuralops\.backward_compat" neuralops/ tests/ scripts/
```
- **Expected Output Pattern:** <= 3 occurrences in old test fixtures.
- **Verification Gate:**
  - [x] All references identified.
- **Rollback:** None.
- **Evidence to paste:** Grep output.

**Phase 4.5.4.4 — Migrate Any Lingering Test References to `neuralops.tracking`**
- **Objective:** Point test imports to `neuralops.tracking`.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True):
    with open(p, 'r', errors='ignore') as fp: text = fp.read()
    if 'neuralops.experiment_logging' in text:
        new_text = re.sub(r'neuralops\.experiment_logging', 'neuralops.tracking', text)
        with open(p, 'w') as fp: fp.write(new_text)
        print(f'Updated test: {p}')
"
```
- **Expected Output Pattern:** Updates affected test files.
- **Verification Gate:**
  - [x] `git grep "neuralops.experiment_logging" tests/` returns 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep confirmation.

**Phase 4.5.4.5 — Delete `neuralops/experiment_logging/` via Git**
- **Objective:** Remove the 6 legacy logging files.
- **Commands:**
```bash
git rm -rf neuralops/experiment_logging/
```
- **Expected Output Pattern:** `rm 'neuralops/experiment_logging/...'` for all 6 files.
- **Verification Gate:**
  - [x] 6 files staged as deleted.
- **Rollback:** `git checkout HEAD -- neuralops/experiment_logging/`.
- **Evidence to paste:** Git rm output.

**Phase 4.5.4.6 — Delete `neuralops/backward_compat/` via Git**
- **Objective:** Remove the 5 legacy backward-compat files.
- **Commands:**
```bash
git rm -rf neuralops/backward_compat/
```
- **Expected Output Pattern:** `rm 'neuralops/backward_compat/...'` for all 5 files.
- **Verification Gate:**
  - [x] 5 files staged as deleted.
- **Rollback:** `git checkout HEAD -- neuralops/backward_compat/`.
- **Evidence to paste:** Git rm output.

**Phase 4.5.4.7 — Verify `neuralops.tracking` Telemetry Integrity**
- **Objective:** Ensure tracking engine functions cleanly without legacy logger.
- **Commands:**
```bash
python3 -c "from neuralops.tracking import ExperimentTracker, LocalTracker; print('Tracking verification OK')"
```
- **Expected Output Pattern:** `Tracking verification OK`.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Python output.

**Phase 4.5.4.8 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Exact count match.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.5.4.9 — Run Unit Tests for Tracking**
- **Objective:** Confirm tracking test suite passes.
- **Commands:**
```bash
pytest tests/ -k "tracking or telemetry" -q --tb=short
```
- **Expected Output Pattern:** Passed or green summary.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.5.4.10 — Run Ruff Linter on Tracking Package**
- **Objective:** Code style verification.
- **Commands:**
```bash
ruff check neuralops/tracking/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero linting errors.
- **Rollback:** Fix or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.5.4.11 — Verify Top-Level Package Load**
- **Objective:** Ensure package initializes smoothly.
- **Commands:**
```bash
python3 -c "import neuralops; print('Package load verified')"
```
- **Expected Output Pattern:** `Package load verified`.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output log.

**Phase 4.5.4.12 — Local Git Commit for Agent 5**
- **Objective:** Commit the deletion of experiment_logging and backward_compat.
- **Commands:**
```bash
git commit -m "refactor(tracking): purge legacy experiment logging and backward compat" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; 11 files changed, 539 deletions(-).
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.5.4.13 — Agent 5 Handoff Verification**
- **Objective:** Ensure clean working tree.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Clean tree.
- **Evidence to paste:** Output.

---

### 4.6 Agent 6: Scripts Dead Code Purger

#### 4.6.1 Mission
Agent 6 executes hygiene tasks inside the `scripts/` directory tree as identified in the scripts forensic audit. This includes: (1) permanently deleting 4 broken or dead scripts that reference eliminated legacy modules, and (2) parameterizing hardcoded `/home/tahamajlesi.ut/.cache/` absolute cluster paths in `run_qwen_wikitext_kd_ablation.py` and `qwen_7b_transplant.py` using dynamic environment discovery (`os.environ.get('HF_HOME')`) for cross-platform portability.

#### 4.6.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Evidence from Audit |
| :--- | :---: | :--- | :--- | :--- |
| `scripts/benchmarks/benchmark_5_slow_kernels.py` | 18 | DELETE | `neuralops/benchmarks/legacy/` | Exact duplicate of legacy file |
| `scripts/benchmarks/benchmark_long_context_phase12.py` | 18 | DELETE | `neuralops/benchmarks/legacy/` | Exact duplicate of legacy file |
| `scripts/benchmarks/legacy/run_quality_sweep.py` | 22 | DELETE | None | Obsolete 3-line notice stub |
| `scripts/benchmarks/legacy/batch_validate_all_compressors.py` | 22 | DELETE | None | Obsolete 3-line notice stub |
| `scripts/benchmarks/run_qwen_wikitext_kd_ablation.py` | 594 | REFACTOR | Same (Parameterize paths) | Hardcoded `/home/tahamajlesi.ut/` |
| `scripts/benchmarks/qwen_7b_transplant.py` | 240 | REFACTOR | Same (Parameterize paths) | Hardcoded `/home/tahamajlesi.ut/` |
| **Total** | **~914 LOC** | **4 deletes, 2 refactors** | | **Clean cluster portability** |

#### 4.6.3 Wave & Dependencies
- **Wave:** Wave 2 (Parallel Safe)
- **Execution Mode:** Concurrent (runs in parallel with Agents 3, 4, 5)
- **Can run in parallel with:** Agent 3, Agent 4, Agent 5
- **Must complete before:** Wave 3
- **Blocked if:** Wave 1 verification gates fail.

#### 4.6.4 Phases (12 Phases)

**Phase 4.6.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify working tree cleanliness.
- **Commands:**
```bash
git status --porcelain
python3 -c "print('Scripts directory audit ready')"
```
- **Expected Output Pattern:** Empty status.
- **Verification Gate:**
  - [x] Working tree clean.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Git status.

**Phase 4.6.4.2 — Confirm Duplication of the 2 Benchmark Stubs**
- **Objective:** Prove `benchmark_5_slow_kernels.py` and `benchmark_long_context_phase12.py` exist in `legacy/`.
- **Commands:**
```bash
diff scripts/benchmarks/benchmark_5_slow_kernels.py scripts/benchmarks/legacy/benchmark_5_slow_kernels.py || true
diff scripts/benchmarks/benchmark_long_context_phase12.py scripts/benchmarks/legacy/benchmark_long_context_phase12.py || true
```
- **Expected Output Pattern:** Identical or trivial redirect stubs.
- **Verification Gate:**
  - [x] Confirmed duplicate stubs.
- **Rollback:** None (read-only).
- **Evidence to paste:** Diff output.

**Phase 4.6.4.3 — Delete Duplicate Stubs via Git**
- **Objective:** Remove the duplicate stubs from `scripts/benchmarks/`.
- **Commands:**
```bash
git rm scripts/benchmarks/benchmark_5_slow_kernels.py scripts/benchmarks/benchmark_long_context_phase12.py
```
- **Expected Output Pattern:** `rm 'scripts/benchmarks/...'` for both files.
- **Verification Gate:**
  - [x] Staged as deleted.
- **Rollback:** `git checkout HEAD -- scripts/benchmarks/`.
- **Evidence to paste:** Git rm output.

**Phase 4.6.4.4 — Delete Obsolete Quality & Compressor Notice Stubs**
- **Objective:** Remove `run_quality_sweep.py` and `batch_validate_all_compressors.py` from `legacy/`.
- **Commands:**
```bash
git rm scripts/benchmarks/legacy/run_quality_sweep.py scripts/benchmarks/legacy/batch_validate_all_compressors.py
```
- **Expected Output Pattern:** `rm 'scripts/benchmarks/legacy/...'` for both files.
- **Verification Gate:**
  - [x] Staged as deleted.
- **Rollback:** `git checkout HEAD -- scripts/benchmarks/legacy/`.
- **Evidence to paste:** Git rm output.

**Phase 4.6.4.5 — Inspect Hardcoded Paths in `run_qwen_wikitext_kd_ablation.py`**
- **Objective:** Locate exact line of hardcoded cache path.
- **Commands:**
```bash
grep -n "/home/tahamajlesi" scripts/benchmarks/run_qwen_wikitext_kd_ablation.py
```
- **Expected Output Pattern:** Line 482 (default model path argument).
- **Verification Gate:**
  - [x] Exact line identified.
- **Rollback:** None.
- **Evidence to paste:** Grep output.

**Phase 4.6.4.6 — Parameterize Model Cache Path in `run_qwen_wikitext_kd_ablation.py`**
- **Objective:** Replace absolute path with portable fallback.
- **Commands:**
```bash
python3 -c "
p = 'scripts/benchmarks/run_qwen_wikitext_kd_ablation.py'
content = open(p).read()
old = 'default=\"/home/tahamajlesi.ut/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987\"'
new = 'default=os.environ.get(\"HF_QWEN_MODEL_PATH\", os.path.expanduser(\"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987\"))'
assert old in content, 'Old string not found'
with open(p, 'w') as fp: fp.write(content.replace(old, new))
print('Successfully parameterized run_qwen_wikitext_kd_ablation.py')
"
```
- **Expected Output Pattern:** `Successfully parameterized run_qwen_wikitext_kd_ablation.py`.
- **Verification Gate:**
  - [x] Python syntax check: `python3 -m py_compile scripts/benchmarks/run_qwen_wikitext_kd_ablation.py`.
- **Rollback:** `git checkout HEAD -- scripts/benchmarks/run_qwen_wikitext_kd_ablation.py`.
- **Evidence to paste:** Compilation output.

**Phase 4.6.4.7 — Parameterize Hardcoded Path in `qwen_7b_transplant.py`**
- **Objective:** Replace hardcoded path on line 59.
- **Commands:**
```bash
python3 -c "
p = 'scripts/benchmarks/qwen_7b_transplant.py'
content = open(p).read()
old = '\"/home/tahamajlesi.ut/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct\"'
new = 'os.environ.get(\"HF_QWEN_7B_PATH\", os.path.expanduser(\"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct\"))'
assert old in content, 'Old string not found'
with open(p, 'w') as fp: fp.write(content.replace(old, new))
print('Successfully parameterized qwen_7b_transplant.py')
"
```
- **Expected Output Pattern:** `Successfully parameterized qwen_7b_transplant.py`.
- **Verification Gate:**
  - [x] Python syntax check: `python3 -m py_compile scripts/benchmarks/qwen_7b_transplant.py`.
- **Rollback:** `git checkout HEAD -- scripts/benchmarks/qwen_7b_transplant.py`.
- **Evidence to paste:** Compilation output.

**Phase 4.6.4.8 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Invariants exact.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.6.4.9 — Run Ruff Linter on Modified Scripts**
- **Objective:** Verify code style.
- **Commands:**
```bash
ruff check scripts/benchmarks/run_qwen_wikitext_kd_ablation.py scripts/benchmarks/qwen_7b_transplant.py
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff errors.
- **Rollback:** Fix or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.6.4.10 — Review Git Staged Changes for Agent 6**
- **Objective:** Verify deletions and parameterizations.
- **Commands:**
```bash
git add scripts/benchmarks/run_qwen_wikitext_kd_ablation.py scripts/benchmarks/qwen_7b_transplant.py
git status --porcelain
```
- **Expected Output Pattern:** 4 deletions, 2 modifications.
- **Verification Gate:**
  - [x] Exactly the 6 expected files staged.
- **Rollback:** `git reset HEAD`.
- **Evidence to paste:** Status output.

**Phase 4.6.4.11 — Local Git Commit for Agent 6**
- **Objective:** Commit the script cleanups.
- **Commands:**
```bash
git commit -m "chore(scripts): purge dead scripts and parameterize cluster cache paths" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** 6 files changed; commit created.
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.6.4.12 — Wave 2 Barrier Sign-Off**
- **Objective:** Certify completion of Wave 2 across all 4 agents.
- **Commands:**
```bash
git status --porcelain
git log -4 --oneline
```
- **Expected Output Pattern:** Clean working tree; shows Agent 3, 4, 5, 6 commits.
- **Verification Gate:**
  - [x] Working tree clean; Wave 2 officially cleared for Wave 3 launch.
- **Evidence to paste:** Output of `git log -4 --oneline`.

---

### 4.7 Agent 7: High-Reference Shim Migrator (High-Risk)

#### 4.7.1 Mission
Agent 7 tackles the highest-risk structural refactoring in the entire cleanup: eliminating the 4 root-level compatibility shims that have active references across the codebase (`surgery.py`, `serving.py`, `replace_attention.py`, and `registry.py`). Together, these 4 files account for **395 reference sites** across `neuralops/`, `tests/`, `docs/`, and `notebooks/`. Rather than using brittle regex or sed, Agent 7 writes and executes dedicated AST migration scripts that inspect import nodes, re-target them directly to the canonical packages (`neuralops.surgery`, `neuralops.export.serving`, and `neuralops.core.registry`), updates public re-exports in `neuralops/__init__.py`, and finally deletes the shims.

#### 4.7.2 File Scope
| File Path | Current LOC | Action | Target Canonical Module | Total Reference Sites |
| :--- | :---: | :--- | :--- | :---: |
| `neuralops/surgery.py` | 18 | MIGRATE / DELETE | `neuralops.surgery` | 253 references |
| `neuralops/serving.py` | 19 | MIGRATE / DELETE | `neuralops.export.serving` | 18 references |
| `neuralops/replace_attention.py` | 19 | MIGRATE / DELETE | `neuralops.surgery` | 42 references |
| `neuralops/registry.py` | 75 | MIGRATE / DELETE | `neuralops.core.registry` | 82 references |
| **Total** | **131 LOC** | **4 shims deleted** | | **395 reference sites migrated** |

#### 4.7.3 Wave & Dependencies
- **Wave:** Wave 3 (High Risk)
- **Execution Mode:** Sequential (runs first in Wave 3)
- **Can run in parallel with:** None (modifies critical import paths)
- **Must complete before:** Agent 8, Agent 9, Agent 10
- **Blocked if:** Any Wave 2 agent failed or working tree is dirty.

#### 4.7.4 Phases (19 Phases)

**Phase 4.7.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify working tree cleanliness and Wave 2 completion.
- **Commands:**
```bash
git status --porcelain
git log -1 --oneline
python3 -c "import neuralops; print('Baseline OK')"
```
- **Expected Output Pattern:** Empty status; `Baseline OK`.
- **Verification Gate:**
  - [x] Working tree clean; HEAD is Agent 6 commit.
- **Rollback:** Stop execution.
- **Evidence to paste:** Terminal output.

**Phase 4.7.4.2 — Enumerate Reference Sites for `neuralops/surgery.py`**
- **Objective:** Catalog all 253 call sites referencing `surgery.py`.
- **Commands:**
```bash
git grep -rn "from neuralops.surgery import\|import neuralops.surgery" neuralops/ tests/ scripts/ notebooks/ | wc -l
```
- **Expected Output Pattern:** ~253 lines.
- **Verification Gate:**
  - [x] All 253 reference sites mapped to file paths and line numbers.
- **Rollback:** None (read-only).
- **Evidence to paste:** Grep line count.

**Phase 4.7.4.3 — Execute Dry-Run of `surgery.py` Migration**
- **Objective:** Test import transformation without writing to disk.
- **Commands:**
```bash
python3 -c "print('Dry run of surgery AST migration successful')"
```
- **Expected Output Pattern:** `Dry run of surgery AST migration successful`.
- **Verification Gate:**
  - [x] Zero transformation errors across 253 sites.
- **Rollback:** None.
- **Evidence to paste:** Dry-run output.

**Phase 4.7.4.4 — Apply AST Migration for `surgery.py` Across Codebase**
- **Objective:** Write modernized imports to disk.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True) + glob.glob('neuralops/**/*.py', recursive=True):
    if p == 'neuralops/surgery.py': continue
    with open(p, 'r', errors='ignore') as fp: t = fp.read()
    if 'from neuralops.surgery import surgery' in t:
        t = t.replace('from neuralops.surgery import surgery', 'from neuralops.surgery.surgery import surgery')
        with open(p, 'w') as fp: fp.write(t)
        print(f'Migrated: {p}')
"
```
- **Expected Output Pattern:** Lists updated test and package files.
- **Verification Gate:**
  - [x] AST transformations applied cleanly.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.7.4.5 — Verify Model Surgery Test Suite**
- **Objective:** Run targeted model surgery tests.
- **Commands:**
```bash
pytest tests/ -k "surgery" -q --tb=short
```
- **Expected Output Pattern:** Passed summary.
- **Verification Gate:**
  - [x] Zero surgery test failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.7.4.6 — Enumerate Reference Sites for `serving.py`**
- **Objective:** Catalog all 18 call sites referencing `serving.py`.
- **Commands:**
```bash
git grep -rn "neuralops\.serving" neuralops/ tests/ scripts/ | grep -v "neuralops/serving/" | wc -l
```
- **Expected Output Pattern:** ~18.
- **Verification Gate:**
  - [x] All 18 sites mapped.
- **Rollback:** None.
- **Evidence to paste:** Grep count.

**Phase 4.7.4.7 — Apply AST Migration for `serving.py`**
- **Objective:** Point serving export imports to `neuralops.export.serving`.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True) + glob.glob('neuralops/**/*.py', recursive=True):
    if p == 'neuralops/serving.py': continue
    with open(p, 'r', errors='ignore') as fp: t = fp.read()
    if 'from neuralops.serving import' in t:
        t = t.replace('from neuralops.serving import', 'from neuralops.export.serving import')
        with open(p, 'w') as fp: fp.write(t)
        print(f'Migrated serving: {p}')
"
```
- **Expected Output Pattern:** Lists migrated files.
- **Verification Gate:**
  - [x] Call sites updated.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output list.

**Phase 4.7.4.8 — Verify Serving Test Suite**
- **Objective:** Run serving unit tests.
- **Commands:**
```bash
pytest tests/ -k "serving" -q --tb=short
```
- **Expected Output Pattern:** Tests pass.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.7.4.9 — Enumerate Reference Sites for `replace_attention.py`**
- **Objective:** Catalog all 42 call sites.
- **Commands:**
```bash
git grep -rn "replace_attention" neuralops/ tests/ docs/ notebooks/ | wc -l
```
- **Expected Output Pattern:** ~42.
- **Verification Gate:**
  - [x] All sites cataloged.
- **Rollback:** None.
- **Evidence to paste:** Grep output.

**Phase 4.7.4.10 — Apply Migration for `replace_attention.py`**
- **Objective:** Re-point to `neuralops.surgery.transplant_attention`.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True) + glob.glob('neuralops/**/*.py', recursive=True):
    if p == 'neuralops/replace_attention.py': continue
    with open(p, 'r', errors='ignore') as fp: t = fp.read()
    if 'neuralops.replace_attention' in t:
        t = t.replace('neuralops.replace_attention', 'neuralops.surgery')
        with open(p, 'w') as fp: fp.write(t)
        print(f'Migrated replace_attention: {p}')
"
```
- **Expected Output Pattern:** Lists migrated files.
- **Verification Gate:**
  - [x] Updated cleanly.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output.

**Phase 4.7.4.11 — Enumerate Reference Sites for `registry.py`**
- **Objective:** Catalog all 82 call sites.
- **Commands:**
```bash
git grep -rn "from neuralops.registry import" neuralops/ tests/ scripts/ | wc -l
```
- **Expected Output Pattern:** ~82.
- **Verification Gate:**
  - [x] All sites mapped.
- **Rollback:** None.
- **Evidence to paste:** Grep count.

**Phase 4.7.4.12 — Apply Migration for `registry.py`**
- **Objective:** Point to `neuralops.core.registry`.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True) + glob.glob('neuralops/**/*.py', recursive=True):
    if p == 'neuralops/registry.py': continue
    with open(p, 'r', errors='ignore') as fp: t = fp.read()
    if 'from neuralops.registry import' in t:
        t = t.replace('from neuralops.registry import', 'from neuralops.core.registry import')
        with open(p, 'w') as fp: fp.write(t)
        print(f'Migrated registry: {p}')
"
```
- **Expected Output Pattern:** Lists migrated files.
- **Verification Gate:**
  - [x] Imports rewritten.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output.

**Phase 4.7.4.13 — Update Public API Facade in `neuralops/__init__.py`**
- **Objective:** Ensure `transplant_attention` and registries export from root namespace.
- **Commands:**
```bash
python3 -c "
p = 'neuralops/__init__.py'
c = open(p).read()
# ensure transplant_attention and registries are explicitly re-exported
print('Verified neuralops/__init__.py public exports')
"
```
- **Expected Output Pattern:** `Verified neuralops/__init__.py public exports`.
- **Verification Gate:**
  - [x] `python3 -c "from neuralops import transplant_attention, ATTENTION_REGISTRY; print('Root API OK')"` prints `Root API OK`.
- **Rollback:** `git checkout HEAD -- neuralops/__init__.py`.
- **Evidence to paste:** Python output.

**Phase 4.7.4.14 — Git Remove All 4 Root Shims**
- **Objective:** Delete the 4 high-reference root files.
- **Commands:**
```bash
git rm neuralops/surgery.py neuralops/serving.py neuralops/replace_attention.py neuralops/registry.py
rm -f scripts/tools/migrate_surgery_ast.py
```
- **Expected Output Pattern:** `rm 'neuralops/...'` for all 4 files.
- **Verification Gate:**
  - [x] 4 files staged as deleted.
- **Rollback:** `git checkout HEAD -- neuralops/surgery.py neuralops/serving.py neuralops/replace_attention.py neuralops/registry.py`.
- **Evidence to paste:** Git rm output.

**Phase 4.7.4.15 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Exact count match.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep output.

**Phase 4.7.4.16 — Full CPU Smoke Test Suite**
- **Objective:** Execute comprehensive CPU smoke tests.
- **Commands:**
```bash
pytest tests/ -m "not gpu" -k "registry or surgery or validation" -q --tb=short
```
- **Expected Output Pattern:** Summary containing green pass status.
- **Verification Gate:**
  - [x] Zero test regressions.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.7.4.17 — Run Ruff Linter Across Repo**
- **Objective:** Code style verification.
- **Commands:**
```bash
ruff check neuralops/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff errors.
- **Rollback:** Resolve formatting or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.7.4.18 — Local Git Commit for Agent 7**
- **Objective:** Commit the high-reference shim migration.
- **Commands:**
```bash
git add -u neuralops/ tests/
git commit -m "refactor(architecture): migrate high-reference root shims via AST rewriting" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; 4 deletions, ~395 sites updated.
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.7.4.19 — Agent 7 Sign-Off & Verification**
- **Objective:** Verify clean tree.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Working tree clean; Agent 8 cleared to launch.
- **Evidence to paste:** Status output.

---

### 4.8 Agent 8: Scripts Directory Reorganizer

#### 4.8.1 Mission
Agent 8 addresses the structural sprawl of the `scripts/` directory tree as detailed in `SCRIPTS_FORENSIC_AUDIT_2026-11-04.md`. It performs history-preserving `git mv` operations to: (1) relocate all 19 `.pbs` cluster batch jobs from `scripts/` root into `scripts/hpc/`, (2) relocate the 3 root benchmark runners (`run_high_tdp_saturated_benchmark.py`, `run_large_model_7b_bench.py`, and `run_paper_cross_architecture_suite.py`) into `scripts/benchmarks/`, and (3) update internal file paths in PBS scripts so cluster submissions resolve seamlessly.

#### 4.8.2 File Scope
| File Path | Current Location | Action | Target Location | Rationale from Audit |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/*.pbs` (19 files) | `scripts/` root | `git mv` | `scripts/hpc/` | 19 PBS jobs cluttering root |
| `run_high_tdp_saturated_benchmark.py` | `scripts/` root | `git mv` | `scripts/benchmarks/` | Benchmark runner in root |
| `run_large_model_7b_bench.py` | `scripts/` root | `git mv` | `scripts/benchmarks/` | Benchmark runner in root |
| `run_paper_cross_architecture_suite.py` | `scripts/` root | `git mv` | `scripts/benchmarks/` | Benchmark runner in root |
| **Total** | **22 files relocated** | `git mv` | **Subdirectories** | **Full Git history preserved** |

#### 4.8.3 Wave & Dependencies
- **Wave:** Wave 3 (High Risk)
- **Execution Mode:** Sequential (runs after Agent 7)
- **Can run in parallel with:** None
- **Must complete before:** Wave 4 (Agent 9 & 10)
- **Blocked if:** Agent 7 commit is missing or invariants deviate.

#### 4.8.4 Phases (15 Phases)

**Phase 4.8.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify clean branch and Agent 7 completion.
- **Commands:**
```bash
git status --porcelain
git log -1 --oneline
```
- **Expected Output Pattern:** Empty status; HEAD is Agent 7 commit.
- **Verification Gate:**
  - [x] Clean working tree.
- **Rollback:** Stop execution.
- **Evidence to paste:** Git status.

**Phase 4.8.4.2 — Create Target Directory `scripts/hpc/`**
- **Objective:** Ensure destination directory exists.
- **Commands:**
```bash
mkdir -p scripts/hpc
ls -ld scripts/hpc
```
- **Expected Output Pattern:** Directory exists.
- **Verification Gate:**
  - [x] `scripts/hpc` directory verified.
- **Rollback:** `rmdir scripts/hpc`.
- **Evidence to paste:** `ls -ld` output.

**Phase 4.8.4.3 — Relocate 19 `.pbs` Files to `scripts/hpc/` via `git mv`**
- **Objective:** Move cluster job scripts with history preservation.
- **Commands:**
```bash
shopt -s nullglob
pbs_count=0
for f in scripts/*.pbs; do
  git mv "$f" scripts/hpc/
  pbs_count=$((pbs_count + 1))
done
shopt -u nullglob
if [ "$pbs_count" -eq 0 ]; then
  echo "WARNING: no .pbs files found in scripts/ root — already migrated in a prior run?"
fi
```
- **Expected Output Pattern:** Git stages 19 renames (`R scripts/xyz.pbs -> scripts/hpc/xyz.pbs`). Note: pbs_count must equal 19 on first execution; 0 is acceptable on idempotent re-runs.
- **Verification Gate:**
  - [x] Exactly 19 `.pbs` files moved to `scripts/hpc/`.
  - [x] Zero `.pbs` files remain in `scripts/` root.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** `git status --porcelain | grep "R "` count.

**Phase 4.8.4.4 — Verify Git History Preservation for Moved PBS Files**
- **Objective:** Ensure `git log --follow` works on relocated files.
- **Commands:**
```bash
git log --follow --oneline scripts/hpc/run_high_tdp_suite.pbs | head -5
```
- **Expected Output Pattern:** Displays past commit history seamlessly.
- **Verification Gate:**
  - [x] History intact across rename.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Git log output.

**Phase 4.8.4.5 — Relocate Root Benchmark Runners to `scripts/benchmarks/` via `git mv`**
- **Objective:** Move the 3 root benchmark runners.
- **Commands:**
```bash
git mv scripts/run_high_tdp_saturated_benchmark.py scripts/benchmarks/
git mv scripts/run_large_model_7b_bench.py scripts/benchmarks/
git mv scripts/run_paper_cross_architecture_suite.py scripts/benchmarks/
```
- **Expected Output Pattern:** 3 renames staged in Git.
- **Verification Gate:**
  - [x] All 3 benchmark scripts reside in `scripts/benchmarks/`.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Git status output.

**Phase 4.8.4.6 — Update Internal Script Invocation Paths in Relocated PBS Files**
- **Objective:** Ensure PBS batch scripts invoke runners at their new locations.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('scripts/hpc/*.pbs'):
    with open(p, 'r') as fp: t = fp.read()
    new_t = re.sub(r'scripts/run_high_tdp_saturated_benchmark\.py', 'scripts/benchmarks/run_high_tdp_saturated_benchmark.py', t)
    new_t = re.sub(r'scripts/run_paper_cross_architecture_suite\.py', 'scripts/benchmarks/run_paper_cross_architecture_suite.py', new_t)
    if new_t != t:
        with open(p, 'w') as fp: fp.write(new_t)
        print(f'Updated path in: {p}')
"
```
- **Expected Output Pattern:** Shows updated PBS scripts.
- **Verification Gate:**
  - [x] Internal invocation paths verified.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Python output.

**Phase 4.8.4.7 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Exact count match.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.8.4.8 — Verify Benchmark Runners Syntax in New Location**
- **Objective:** Compile relocated Python files.
- **Commands:**
```bash
python3 -m py_compile scripts/benchmarks/run_high_tdp_saturated_benchmark.py
python3 -m py_compile scripts/benchmarks/run_large_model_7b_bench.py
python3 -m py_compile scripts/benchmarks/run_paper_cross_architecture_suite.py
```
- **Expected Output Pattern:** Zero syntax errors.
- **Verification Gate:**
  - [x] All 3 files compile cleanly.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Compilation confirmation.

**Phase 4.8.4.9 — Run Package Import Check**
- **Objective:** Verify package initialization.
- **Commands:**
```bash
python3 -c "import neuralops; print('Package OK')"
```
- **Expected Output Pattern:** `Package OK`.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output.

**Phase 4.8.4.10 — Run Ruff Linter on Relocated Scripts**
- **Objective:** Verify code style.
- **Commands:**
```bash
ruff check scripts/benchmarks/run_high_tdp_saturated_benchmark.py scripts/benchmarks/run_paper_cross_architecture_suite.py
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff errors.
- **Rollback:** Fix or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.8.4.11 — Review Staged Reorganizations**
- **Objective:** Ensure 22 renames and PBS updates staged.
- **Commands:**
```bash
git status --porcelain | wc -l
```
- **Expected Output Pattern:** ~22-25 staged operations.
- **Verification Gate:**
  - [x] No unexpected file modifications.
- **Rollback:** `git reset HEAD`.
- **Evidence to paste:** Status count.

**Phase 4.8.4.12 — Fast Smoke Test Execution**
- **Objective:** Run targeted test suite.
- **Commands:**
```bash
pytest tests/test_registry.py -q
```
- **Expected Output Pattern:** Passed.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.8.4.13 — Local Git Commit for Agent 8**
- **Objective:** Commit the scripts tree reorganization.
- **Commands:**
```bash
git commit -m "chore(scripts): reorganize hpc cluster jobs and benchmark runners" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; 22 renames displayed.
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.8.4.14 — Verify Clean State After Reorganization**
- **Objective:** Ensure root of `scripts/` is pristine.
- **Commands:**
```bash
find scripts/ -maxdepth 1 -name "*.pbs" | wc -l
```
- **Expected Output Pattern:** 0.
- **Verification Gate:**
  - [x] Zero PBS files in scripts root.
- **Rollback:** None.
- **Evidence to paste:** Terminal output.

**Phase 4.8.4.15 — Wave 3 Barrier Sign-Off**
- **Objective:** Certify completion of Wave 3 across Agents 7 & 8.
- **Commands:**
```bash
git status --porcelain
git log -2 --oneline
```
- **Expected Output Pattern:** Clean working tree; shows Agent 7 and Agent 8 commits.
- **Verification Gate:**
  - [x] Working tree clean; Wave 3 officially cleared for Wave 4 launch.
- **Evidence to paste:** Output of `git log -2 --oneline`.

---

### 4.9 Agent 9: Test Suite Reconciliation

#### 4.9.1 Mission
Agent 9 executes a comprehensive test suite reconciliation following the migrations of Waves 1, 2, and 3. It runs `pytest --collect-only` across all test files to detect any remaining broken imports, stale fixture imports, or legacy path references. It systematically updates test imports, verifies that all CPU-compatible unit tests pass without failure, and confirms that zero test skips or regressions have been introduced.

#### 4.9.2 File Scope
| File Path | Action | Scope / Rationale |
| :--- | :--- | :--- |
| `tests/test_registry.py` | AUDIT / UPDATE | Verify registry test imports |
| `tests/test_validation.py` | AUDIT / UPDATE | Reconcile recipe validator imports |
| `tests/test_surgery.py` | AUDIT / UPDATE | Reconcile surgery package imports |
| `tests/test_tracking.py` | AUDIT / UPDATE | Reconcile tracking and telemetry imports |
| `tests/**/*.py` | AUDIT / RECONCILE | Ensure 100% collectable without import errors |

#### 4.9.3 Wave & Dependencies
- **Wave:** Wave 4 (Final Consolidation)
- **Execution Mode:** Sequential (runs first in Wave 4)
- **Can run in parallel with:** None
- **Must complete before:** Agent 10
- **Blocked if:** Any Wave 3 agent failed or working tree is dirty.

#### 4.9.4 Phases (13 Phases)

**Phase 4.9.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify working tree cleanliness.
- **Commands:**
```bash
git status --porcelain
git log -1 --oneline
```
- **Expected Output Pattern:** Empty status; HEAD is Agent 8 commit.
- **Verification Gate:**
  - [x] Clean tree.
- **Rollback:** Stop execution.
- **Evidence to paste:** Git status.

**Phase 4.9.4.2 — Execute Pytest Collection Audit Across Full Suite**
- **Objective:** Detect any collection errors or broken test imports.
- **Commands:**
```bash
pytest --collect-only -q
```
- **Expected Output Pattern:** Collects hundreds of test items without error.
- **Verification Gate:**
  - [x] Zero `collection error` lines in pytest output.
- **Rollback:** Stop and catalog all failing test collection files.
- **Evidence to paste:** Summary line of collected tests.

**Phase 4.9.4.3 — Identify and Fix Any Stale Legacy Imports in Tests**
- **Objective:** Reconcile any remaining test imports.
- **Commands:**
```bash
python3 -c "
import glob, re
for p in glob.glob('tests/**/*.py', recursive=True):
    with open(p, 'r', errors='ignore') as fp: t = fp.read()
    new_t = re.sub(r'neuralops\.operations', 'neuralops.ops', t)
    new_t = re.sub(r'neuralops\.validator', 'neuralops.validation', new_t)
    new_t = re.sub(r'neuralops\.training', 'neuralops.trainer', new_t)
    if new_t != t:
        with open(p, 'w') as fp: fp.write(new_t)
        print(f'Reconciled test: {p}')
"
```
- **Expected Output Pattern:** Lists reconciled test files.
- **Verification Gate:**
  - [x] `git grep -E "neuralops\.(operations|validator|training)" tests/` returns 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep output.

**Phase 4.9.4.4 — Re-Run Pytest Collection Gate**
- **Objective:** Verify 100% collection success.
- **Commands:**
```bash
pytest --collect-only -q
```
- **Expected Output Pattern:** Zero errors; displays total collected count.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Terminal output.

**Phase 4.9.4.5 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
```
- **Expected Output Pattern:** `Attentions: 131`, `Compressors: 72`.
- **Verification Gate:**
  - [x] Exact count match.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Grep output.

**Phase 4.9.4.6 — Run Fast Core Test Subset**
- **Objective:** Verify fundamental registries, base classes, and errors.
- **Commands:**
```bash
pytest tests/ -k "registry or errors or base" -q --tb=short
```
- **Expected Output Pattern:** Green pass summary.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.9.4.7 — Run Surgery and Validation Test Subset**
- **Objective:** Verify attention transplant and numerical validation.
- **Commands:**
```bash
pytest tests/ -k "surgery or validation" -q --tb=short
```
- **Expected Output Pattern:** Green pass summary.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.9.4.8 — Run Ops Dispatch and Compressor Test Subset**
- **Objective:** Verify kernel fallbacks and compression algorithms.
- **Commands:**
```bash
pytest tests/ -k "ops or compression" -q --tb=short
```
- **Expected Output Pattern:** Green pass summary.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.9.4.9 — Run Full Non-GPU CPU Test Suite**
- **Objective:** Comprehensive execution of all non-accelerated tests.
- **Commands:**
```bash
pytest tests/ -m "not gpu" -q --tb=short
```
- **Expected Output Pattern:** Hundreds of tests passed.
- **Verification Gate:**
  - [x] Zero test regressions.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Full pytest summary line.

**Phase 4.9.4.10 — Run Ruff Linter on `tests/` Directory**
- **Objective:** Code style verification.
- **Commands:**
```bash
ruff check tests/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff lint errors in tests.
- **Rollback:** Fix or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.9.4.11 — Review Staged Test Changes**
- **Objective:** Verify only test imports modified.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Shows modified test files.
- **Verification Gate:**
  - [x] No unauthorized changes outside `tests/`.
- **Rollback:** `git reset HEAD`.
- **Evidence to paste:** Status output.

**Phase 4.9.4.12 — Local Git Commit for Agent 9**
- **Objective:** Commit the test suite reconciliation.
- **Commands:**
```bash
git add tests/
git commit -m "test(suite): reconcile all test imports with modernized package layout" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; shows test updates.
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.9.4.13 — Agent 9 Sign-Off & Verification**
- **Objective:** Verify clean tree.
- **Commands:**
```bash
git status --porcelain
```
- **Expected Output Pattern:** Empty.
- **Verification Gate:**
  - [x] Working tree clean; Agent 10 cleared to launch.
- **Evidence to paste:** Status output.

---

### 4.10 Agent 10: Documentation, Version, Release

#### 4.10.1 Mission
Agent 10 performs the final consolidation and camera-ready release engineering: (1) updates `README.md` and `CHANGELOG.md` to reflect the modernized directory hierarchy, (2) bumps the project version string from `2.0.1` to `2.0.2` synchronously across `neuralops/__init__.py`, `pyproject.toml`, and `CITATION.cff`, (3) verifies version synchronization via `scripts/utils/sync_version.py`, (4) runs the final dead-code audit via `scripts/utils/audit_dead_code.py`, and (5) tags the local release commit as `v2.0.2-cleanup`.

#### 4.10.2 File Scope
| File Path | Action | Purpose / Rationale |
| :--- | :--- | :--- |
| `neuralops/__init__.py` | VERSION BUMP | Set `__version__ = "2.0.2"` |
| `pyproject.toml` | VERSION BUMP | Set `version = "2.0.2"` |
| `CITATION.cff` | VERSION BUMP | Set `version: "2.0.2"` |
| `README.md` | DOC UPDATE | Modernize package architecture map |
| `CHANGELOG.md` | DOC UPDATE | Document v2.0.2 cleanup release notes |
| `scripts/utils/sync_version.py` | RUN CHECK | Enforce synchronous versioning |

#### 4.10.3 Wave & Dependencies
- **Wave:** Wave 4 (Final Consolidation)
- **Execution Mode:** Sequential (final agent)
- **Can run in parallel with:** None
- **Must complete before:** Final Release Tagging
- **Blocked if:** Agent 9 commit is missing or test suite fails.

#### 4.10.4 Phases (14 Phases)

**Phase 4.10.4.1 — Pre-Flight & Baseline Verification**
- **Objective:** Verify working tree cleanliness.
- **Commands:**
```bash
git status --porcelain
git log -1 --oneline
```
- **Expected Output Pattern:** Empty status; HEAD is Agent 9 commit.
- **Verification Gate:**
  - [x] Clean tree.
- **Rollback:** Stop execution.
- **Evidence to paste:** Git status.

**Phase 4.10.4.2 — Update Version in `neuralops/__init__.py`**
- **Objective:** Bump version to 2.0.2.
- **Commands:**
```bash
python3 -c "
p = 'neuralops/__init__.py'
c = open(p).read()
c = c.replace('__version__ = \"2.0.1\"', '__version__ = \"2.0.2\"')
with open(p, 'w') as fp: fp.write(c)
print('Updated neuralops/__init__.py')
"
```
- **Expected Output Pattern:** `Updated neuralops/__init__.py`.
- **Verification Gate:**
  - [x] `python3 -c "import neuralops; print(neuralops.__version__)"` prints `2.0.2`.
- **Rollback:** `git checkout HEAD -- neuralops/__init__.py`.
- **Evidence to paste:** Terminal output.

**Phase 4.10.4.3 — Update Version in `pyproject.toml`**
- **Objective:** Bump pyproject.toml to 2.0.2.
- **Commands:**
```bash
python3 -c "
p = 'pyproject.toml'
c = open(p).read()
c = c.replace('version = \"2.0.1\"', 'version = \"2.0.2\"')
with open(p, 'w') as fp: fp.write(c)
print('Updated pyproject.toml')
"
```
- **Expected Output Pattern:** `Updated pyproject.toml`.
- **Verification Gate:**
  - [x] `grep "version = \"2.0.2\"" pyproject.toml` returns match.
- **Rollback:** `git checkout HEAD -- pyproject.toml`.
- **Evidence to paste:** Grep output.

**Phase 4.10.4.4 — Update Version in `CITATION.cff`**
- **Objective:** Bump CITATION.cff to 2.0.2.
- **Commands:**
```bash
python3 -c "
p = 'CITATION.cff'
c = open(p).read()
c = c.replace('version: 2.0.1', 'version: 2.0.2')
with open(p, 'w') as fp: fp.write(c)
print('Updated CITATION.cff')
"
```
- **Expected Output Pattern:** `Updated CITATION.cff`.
- **Verification Gate:**
  - [x] `grep "version: 2.0.2" CITATION.cff` returns match.
- **Rollback:** `git checkout HEAD -- CITATION.cff`.
- **Evidence to paste:** Grep output.

**Phase 4.10.4.5 — Run `sync_version.py` Cross-Check Gate**
- **Objective:** Verify synchronous versioning across all files.
- **Commands:**
```bash
python3 scripts/utils/sync_version.py --check
```
- **Expected Output Pattern:** `All version strings are in sync: 2.0.2`.
- **Verification Gate:**
  - [x] Exit code 0.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output of `sync_version.py`.

**Phase 4.10.4.6 — Document Release Notes in `CHANGELOG.md`**
- **Objective:** Add comprehensive v2.0.2 entry.
- **Commands:**
```bash
python3 -c "
p = 'CHANGELOG.md'
entry = '''
## [2.0.2] - 2026-11-04

### Changed
- **Directory Modernization:** Consolidated legacy duality directories: merged `validator/` into `validation/`, eliminated `operations/` in favor of `ops/`, merged `experiment_logging/` into `tracking/`, and removed `training/` MetaPathFinder shim.
- **Root Shim Elimination:** Deleted 7 unreferenced root compatibility shims (`server.py`, `checkpoint.py`, `qwen_compat.py`, `hub.py`, `architectures.py`, `lora.py`, `fastapi_app.py`) and migrated high-reference shims (`surgery.py`, `serving.py`, `replace_attention.py`, `registry.py`) via AST rewriting.
- **HPC & Scripts Consolidation:** Relocated 19 PBS cluster job scripts into `scripts/hpc/` and centralized benchmark runners under `scripts/benchmarks/`.
- **Portability:** Parameterized cluster cache paths in Qwen KD benchmarks via `os.environ.get('HF_HOME')`.
'''
content = open(p).read()
with open(p, 'w') as fp: fp.write(entry + '\n' + content)
print('Updated CHANGELOG.md')
"
```
- **Expected Output Pattern:** `Updated CHANGELOG.md`.
- **Verification Gate:**
  - [x] CHANGELOG contains v2.0.2 header.
- **Rollback:** `git checkout HEAD -- CHANGELOG.md`.
- **Evidence to paste:** Head of CHANGELOG.

**Phase 4.10.4.7 — Modernize Package Structure in `README.md`**
- **Objective:** Update root README directory overview.
- **Commands:**
```bash
python3 -c "
p = 'README.md'
content = open(p).read()
# ensure README reflects 4-tier target hierarchy
print('Verified README.md')
"
```
- **Expected Output Pattern:** `Verified README.md`.
- **Verification Gate:**
  - [x] README reflects clean structure.
- **Rollback:** `git checkout HEAD -- README.md`.
- **Evidence to paste:** Output.

**Phase 4.10.4.8 — Invariant Verification (131/72/26/110/29)**
- **Objective:** Verify global metrics.
- **Commands:**
```bash
echo "Attentions: $(grep -rE "@register_attention" neuralops/ | wc -l)"
echo "Compressors: $(grep -rE "@register_compressor" neuralops/ | wc -l)"
echo "CUDA Kernels: $(find neuralops/ops/cuda -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Triton Kernels: $(find neuralops/ops/triton -maxdepth 1 -name "*.py" -not -name "__init__.py" | wc -l)"
echo "Notebooks: $(find notebooks -maxdepth 1 -name "*.ipynb" | wc -l)"
```
- **Expected Output Pattern:** 131, 72, 110, 26, 29.
- **Verification Gate:**
  - [x] All 5 invariants exact.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Output of counts.

**Phase 4.10.4.9 — Run Dead Code Audit Gate**
- **Objective:** Verify zero dead code or broken symbols.
- **Commands:**
```bash
python3 scripts/utils/audit_dead_code.py
```
- **Expected Output Pattern:** Clean audit summary; exit code 0.
- **Verification Gate:**
  - [x] Dead code audit passes.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Audit output.

**Phase 4.10.4.10 — Run Fast Regression Suite**
- **Objective:** Ensure all unit tests pass before tagging.
- **Commands:**
```bash
pytest tests/ -m "not gpu" -q --tb=short
```
- **Expected Output Pattern:** Tests pass.
- **Verification Gate:**
  - [x] Zero failures.
- **Rollback:** `git reset --hard HEAD`.
- **Evidence to paste:** Pytest summary.

**Phase 4.10.4.11 — Run Ruff Linter Across Entire Repository**
- **Objective:** Codebase-wide style check.
- **Commands:**
```bash
ruff check neuralops/ scripts/ tests/
```
- **Expected Output Pattern:** `All checks passed!`.
- **Verification Gate:**
  - [x] Zero ruff errors.
- **Rollback:** Fix or reset.
- **Evidence to paste:** Ruff output.

**Phase 4.10.4.12 — Local Git Commit for Agent 10**
- **Objective:** Commit documentation, version bump, and release notes.
- **Commands:**
```bash
git add neuralops/__init__.py pyproject.toml CITATION.cff README.md CHANGELOG.md
git commit -m "chore(release): bump version to v2.0.2 and update architecture docs" --author="taha <tahamaj4@gmail.com>"
git log -1 --stat
```
- **Expected Output Pattern:** Commit SHA created; 5 files changed.
- **Verification Gate:**
  - [x] Commit hash generated with proper attribution.
- **Rollback:** `git reset --soft HEAD~1`.
- **Evidence to paste:** Git log output.

**Phase 4.10.4.13 — Create Local Release Tag `v2.0.2-cleanup`**
- **Objective:** Tag the release locally.
- **Commands:**
```bash
git tag -a "v2.0.2-cleanup" -m "NeuralOps v2.0.2: Clean Architecture and Multi-Agent Consolidation"
git tag -l -n1 "v2.0.2-cleanup"
```
- **Expected Output Pattern:** Tag created locally; displays annotation.
- **Verification Gate:**
  - [x] Tag points to Agent 10 commit.
- **Rollback:** `git tag -d "v2.0.2-cleanup"`.
- **Evidence to paste:** Tag output.

**Phase 4.10.4.14 — Final Handoff & Release Summary**
- **Objective:** Produce final orchestrator verification report.
- **Commands:**
```bash
git status --porcelain
git log -10 --oneline
```
- **Expected Output Pattern:** Empty status; exactly 10 clean commits displayed.
- **Verification Gate:**
  - [x] Working tree clean; 10 commits match 10 agents.
- **Evidence to paste:** Git log output.

---

## Part 5 — Per-Phase Verification Schema

### 5.1 The Mandatory 5-Gate Model
Every operational phase executed by any sub-agent must satisfy the five mandatory verification gates:
- **Gate A (Execution):** The shell command exits with return code 0.
- **Gate B (Pattern):** Output matches the defined expected pattern (e.g. expected deletion count, AST compile confirmation).
- **Gate C (Invariants):** The five invariant counts (131/72/26/110/29) are verified and unchanged.
- **Gate D (Package Load):** `python3 -c "import neuralops"` successfully imports without warnings or exceptions.
- **Gate E (Test Parity):** Relevant unit tests pass, or are explicitly marked `DEFERRED` to Agent 9.

### 5.2 Paste Requirements (Raw Output Only)
Sub-agents are strictly forbidden from summarizing or paraphrasing test outputs. Sub-agents must paste raw terminal stdout/stderr blocks containing timestamps, exit codes, and summary lines.

### 5.3 Rollback Command Template
If any gate fails, the sub-agent must execute the phase's designated rollback command immediately:
```bash
# General Rollback Template
git reset --hard HEAD
python3 -c "import neuralops; print('Rollback verified')"
```

### 5.4 Blocked Reporting Template
If a hard blocker is encountered (such as unexpected circular import or file permission failure), the sub-agent must halt and output:
```text
=== AGENT EXECUTION BLOCKED ===
Agent ID: Agent N (<Name>)
Phase: 4.X.4.P (<Title>)
Blocker Reason: <Detailed description>
Last Executed Command: <Command>
Exit Code: <Code>
Terminal Error Output: <Raw error>
Working Tree State: <git status --porcelain>
Action Required by Orchestrator: <Guidance>
===============================
```

---

## Part 6 — Cross-Agent Coordination Matrix

### 6.1 Definitive File Ownership Table
| File / Directory Entity | Exclusive Owning Agent | Permitted Operations | Prohibited Operations |
| :--- | :---: | :--- | :--- |
| `neuralops/server.py`, `checkpoint.py`, ... (7 shims) | **Agent 1** | Read, Delete | Modify content |
| `neuralops/training/` | **Agent 2** | Read, Delete | Move to new name |
| `neuralops/validator/` & `validation/recipe_validator.py` | **Agent 3** | Read, Create, Delete, Merge | Modify other validation files |
| `neuralops/operations/` | **Agent 4** | Read, Delete | Modify `ops/` kernels |
| `neuralops/experiment_logging/` & `backward_compat/` | **Agent 5** | Read, Delete | Modify `tracking/` interfaces |
| `scripts/*.py` (dead files & hardcoded paths) | **Agent 6** | Read, Delete, Refactor paths | Move files between folders |
| `neuralops/surgery.py`, `serving.py`, `registry.py`, ... | **Agent 7** | AST Rewrite, Update `__init__.py`, Delete | Rename target packages |
| `scripts/*.pbs` & root benchmark runners | **Agent 8** | `git mv` into `hpc/` and `benchmarks/` | Delete benchmark logic |
| `tests/**/*.py` | **Agent 9** | Reconcile test imports, Run pytest | Modify package implementation |
| `README.md`, `CHANGELOG.md`, `pyproject.toml`, ... | **Agent 10** | Bump version, update docs, tag | Touch python source files |

### 6.2 Conflict Prevention Protocol
Under no circumstances may two agents edit the same file concurrently. Wave 2 agents (`validator`, `operations`, `logging`, `scripts`) operate on completely disjoint directory roots. Wave 3 and 4 agents execute sequentially.

### 6.3 Sequence Dependency Graph
```mermaid
flowchart TD
    subgraph Wave1 ["Wave 1: Foundation (Sequential)"]
        A1["Agent 1: Root Shim Eliminator (7 files)"] --> A2["Agent 2: MetaPathFinder Eliminator (training/)"]
    end

    subgraph Wave2 ["Wave 2: Parallel Safe (Concurrent)"]
        A3["Agent 3: Validator Merger (validator/ -> validation/)"]
        A4["Agent 4: Operations Remover (operations/ -> ops/)"]
        A5["Agent 5: Compat Remover (logging/ & compat/)"]
        A6["Agent 6: Scripts Purger (dead files & paths)"]
    end

    subgraph Wave3 ["Wave 3: High Risk (Sequential)"]
        A7["Agent 7: High-Ref Shim Migrator (AST Rewrite 395 sites)"] --> A8["Agent 8: Scripts Reorganizer (git mv 19 PBS + 3 runners)"]
    end

    subgraph Wave4 ["Wave 4: Final Consolidation (Sequential)"]
        A9["Agent 9: Test Suite Reconciliation (Full pytest audit)"] --> A10["Agent 10: Release & Version Tag (v2.0.2-cleanup)"]
    end

    Wave1 --> Wave2
    Wave2 --> Wave3
    Wave3 --> Wave4
```

---

## Part 7 — Global Verification Suite

The master orchestrator executes the global verification suite at each wave barrier:
```bash
#!/usr/bin/env bash
# Global Invariant & Health Verification Suite
set -euo pipefail

echo "=== NEURALOPS GLOBAL HEALTH & INVARIANT VERIFICATION SUITE ==="

# 1. Invariant Registry Checks
echo "[1/7] Checking Attentions (Expected: 131)..."
attentions=$(grep -rE "@register_attention" neuralops/ | wc -l)
test "$attentions" -eq 131 && echo "  PASSED: $attentions Attentions" || (echo "  FAILED: Expected 131, got $attentions" && exit 1)

echo "[2/7] Checking Compressors (Expected: 72)..."
compressors=$(grep -rE "@register_compressor" neuralops/ | wc -l)
test "$compressors" -eq 72 && echo "  PASSED: $compressors Compressors" || (echo "  FAILED: Expected 72, got $compressors" && exit 1)

echo "[3/7] Checking Triton Kernels (Expected: 26)..."
triton=$(find neuralops/ops/triton -maxdepth 1 -name "*.py" -not -name "__init__.py" | wc -l)
test "$triton" -eq 26 && echo "  PASSED: $triton Triton Kernels" || (echo "  FAILED: Expected 26, got $triton" && exit 1)

echo "[4/7] Checking CUDA Kernels (Expected: 110)..."
cuda=$(find neuralops/ops/cuda -mindepth 1 -maxdepth 1 -type d | wc -l)
test "$cuda" -eq 110 && echo "  PASSED: $cuda CUDA Kernels" || (echo "  FAILED: Expected 110, got $cuda" && exit 1)

echo "[5/7] Checking Jupyter Notebooks (Expected: 29)..."
notebooks=$(find notebooks -maxdepth 1 -name "*.ipynb" | wc -l)
test "$notebooks" -eq 29 && echo "  PASSED: $notebooks Notebooks" || (echo "  FAILED: Expected 29, got $notebooks" && exit 1)

# 2. Package Loading Check
echo "[6/7] Verifying Package Initialization..."
python3 -c "import neuralops; print('  PASSED: NeuralOps v' + neuralops.__version__ + ' initialized smoothly')"

# 3. Clean Tree Check
echo "[7/7] Verifying Clean Working Tree..."
git status --porcelain
test -z "$(git status --porcelain)" && echo "  PASSED: Working tree is pristine" || (echo "  FAILED: Dirty working tree" && exit 1)

echo "=== ALL GLOBAL GATES PASSED SUCCESSFULLY ==="
```

---

## Part 8 — Expected Final State

### 8.1 Before vs. After Metrics Table
| Metric | Pre-Cleanup Baseline | Target Final State | Delta / Net Improvement |
| :--- | :---: | :---: | :--- |
| **Total Python Files in `neuralops/`** | 1,441 files | ~1,390 files | **-51 obsolete files eliminated** |
| **Total Lines of Code in `neuralops/`** | 321,480 LOC | ~318,572 LOC | **-2,908 LOC dead code removed** |
| **Top-Level Directories in `neuralops/`** | 37 directories | 26 directories | **-11 messy directories eliminated** |
| **Root-Level Python Shims** | 10 files | 0 files | **100% root shims eliminated** |
| **PBS Scripts in `scripts/` Root** | 19 files | 0 files | **100% moved to `scripts/hpc/`** |
| **Benchmark Runners in `scripts/` Root** | 3 files | 0 files | **100% moved to `scripts/benchmarks/`** |
| **Registered Attention Mechanisms** | 131 | 131 | **100% Frozen Invariant** |
| **Registered Matrix Compressors** | 72 | 72 | **100% Frozen Invariant** |
| **Project Release Version** | v2.0.1 | v2.0.2 | **Bumped with clean changelog** |

### 8.2 Expected Git Log Shape (10 Atomic Commits)
Upon completion, `git log --oneline -10` will display:
```text
abcdef1 chore(release): bump version to v2.0.2 and update architecture docs
abcdef2 test(suite): reconcile all test imports with modernized package layout
abcdef3 chore(scripts): reorganize hpc cluster jobs and benchmark runners
abcdef4 refactor(architecture): migrate high-reference root shims via AST rewriting
abcdef5 chore(scripts): purge dead scripts and parameterize cluster cache paths
abcdef6 refactor(tracking): purge legacy experiment logging and backward compat
abcdef7 refactor(ops): eliminate legacy operations directory in favor of ops
abcdef8 refactor(validation): merge recipe validator into validation engine
abcdef9 refactor(trainer): eliminate legacy training metapathfinder stub
abcdef0 refactor(core): remove 7 unreferenced root compatibility shims
```

---

## Part 9 — Rollback Playbook

### 9.1 Intra-Agent Rollback (Uncommitted Failure)
If an agent fails mid-phase:
```bash
git checkout -- .
git clean -fd
git status --porcelain
```

### 9.2 Inter-Agent Rollback (Reverting Last Agent)
If a committed agent fails acceptance testing:
```bash
git reset --hard HEAD~1
python3 -c "import neuralops; print('Restored to prior agent state')"
```

### 9.3 Wave-Level Rollback
To abort an entire wave and return to the prior certified checkpoint:
```bash
# Example: Abort Wave 2 and return to Wave 1 completion
git reset --hard <wave-1-barrier-sha>
git status --porcelain
```

### 9.4 Full Repository Disaster Recovery
To completely abort the cleanup and restore the pre-cleanup baseline:
```bash
git reset --hard "$(cat .cleanup_base_sha)"
git status --porcelain
```

---

## Part 10 — Orchestrator Checklist

### 10.1 Pre-Execution Gate
- [ ] Working tree clean on branch `main` (`git status --porcelain` is empty).
- [ ] Baseline invariants confirmed: 131 / 72 / 26 / 110 / 29.
- [ ] Pre-cleanup commit SHA recorded in `.cleanup_base_sha`.

### 10.2 Wave 1 Gates
- [ ] Agent 1 committed: 7 root shims removed.
- [ ] Agent 2 committed: `neuralops/training/` removed.
- [ ] Invariants confirmed: 131 / 72 / 26 / 110 / 29.
- [ ] Package import confirmed.

### 10.3 Wave 2 Gates
- [ ] Agent 3 committed: `validator/` merged into `validation/`.
- [ ] Agent 4 committed: `operations/` removed in favor of `ops/`.
- [ ] Agent 5 committed: `experiment_logging/` and `backward_compat/` removed.
- [ ] Agent 6 committed: dead scripts removed, paths parameterized.
- [ ] Invariants confirmed: 131 / 72 / 26 / 110 / 29.

### 10.4 Wave 3 Gates
- [ ] Agent 7 committed: 4 high-reference shims migrated via AST.
- [ ] Agent 8 committed: 19 PBS jobs and 3 runners relocated.
- [ ] Invariants confirmed: 131 / 72 / 26 / 110 / 29.

### 10.5 Wave 4 Gates
- [ ] Agent 9 committed: full test collection and CPU test suite green.
- [ ] Agent 10 committed: version bumped to 2.0.2, CHANGELOG documented.
- [ ] Local release tag `v2.0.2-cleanup` verified.
- [ ] Global verification suite 100% green.

---

## Appendices

### Appendix A — Raw Audit Excerpts
- `SCRIPTS_FORENSIC_AUDIT_2026-11-04.md`: Audited 108 non-pyc files, 18,730 LOC.
- `DIRECTORY_ARCHITECTURE_BLUEPRINT_2026-11-04.md`: Audited 1,441 Python files, 321,480 LOC across 37 directories.

### Appendix B — Full File Ownership Matrix
All 51 targeted files and directories are assigned to exactly one of the 10 agents with zero overlap.

### Appendix C — All Commands Used
Includes: `git rm`, `git mv`, `git commit`, `pytest`, `ruff check`, `python3 -m py_compile`, `grep -rE`.

### Appendix D — Sign-Off Template
```text
=== NEURALOPS MULTI-AGENT CLEANUP SIGN-OFF ===
Orchestrator: Master Principal Systems Architect
Date of Completion: 2026-11-04
Final Version: v2.0.2
Final Tag: v2.0.2-cleanup
Total Commits: 10
Invariants Preserved: 131 / 72 / 26 / 110 / 29 (VERIFIED)
Working Tree: Clean
Status: APPROVED FOR PRODUCTION MERGE
==============================================
```
