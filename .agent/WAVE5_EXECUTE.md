# NEURALOPS WAVE 5 — INTEGRATION, VERIFICATION, AND RELEASE (20 AGENTS)

**Document version:** 1.0
**Effective date:** 2026-11-15
**Mission owner:** Human coordinator
**Executor:** You (the orchestrating agent)
**Scope:** The entire `neuralops/` repository and its release engineering.

---

## 0. WHAT WAVE 5 IS (READ THIS FIRST)

Waves 1 through 4 fixed bugs in isolation. Wave 5 does something different.

Wave 5's mission is:

1. **VERIFY** that Waves 1–4 actually completed and merged cleanly. If
   any prior wave is incomplete, Wave 5 reports it as BLOCKED and refuses
   to proceed.
2. **INTEGRATE** the whole library: prove that all subsystems work
   together end-to-end, not just in isolation.
3. **AUDIT** cross-cutting concerns: error handling, logging, types,
   documentation — consistency across the entire codebase.
4. **RELEASE** the artifact: produce `CHANGELOG.md`, `BREAKING_CHANGES.md`,
   `AUDIT_MASTER_FIXES.md`, `VERIFICATION.md`, bump `_version.py`, build
   the wheel, tag `v2.4.0`.
5. **PROVE** everything: run full test suite, full lint, full types,
   full security scan, and produce a per-criterion verdict report.

Wave 5 does NOT fix bugs inline. If a Wave 5 agent finds a bug, it
adds it to `issues/wave6_hotfix_backlog.md` and reports it as a
finding. Wave 6 (a small, focused hotfix wave) handles it.

**Why this matters:** if Wave 5 agents start fixing bugs, they will
regress Waves 1–4's work. Wave 5's value is *proving* the system is
good, not re-fixing it. Verification and remediation must be separate.

---

## 1. DEFINITION OF DONE

The following MUST be true before the user can merge Wave 5 into `main`
and tag `v2.4.0`:

### 1.1 — Prior wave verification
- [ ] Wave 1 (10 agents, 45 bugs) fully merged to `main`, tests green
- [ ] Wave 2 (20 agents, 60 files) fully merged to `main`, tests green
- [ ] Wave 3 (20 agents) — either fully merged or explicitly skipped
- [ ] Wave 4 (20 agents) — either fully merged or explicitly skipped
- [ ] Benchmarks 18-agent remediation fully merged, `pytest tests/benchmarks/` green

### 1.2 — End-to-end integration
- [ ] Config → Benchmark → Result pipeline works
- [ ] All 6 HF zoos (bert_attention, bert_compressor, qwen_attention,
      qwen_compressor, tinyllama, llava_vlm) load and run
- [ ] Training pipeline (finetune, KD, LoRA, recovery) runs end-to-end
- [ ] Serving pipeline (batching, streaming, paged cache) runs end-to-end
- [ ] All 131 attentions instantiate + forward (with documented exceptions)

### 1.3 — Cross-cutting quality
- [ ] All public API docstrings match actual behavior
- [ ] All exceptions subclass `NeuralOpsError`
- [ ] All logging uses `logging.getLogger(__name__)`, no `print()` in
      library code
- [ ] `mypy --strict` on all public API modules

### 1.4 — Release artifacts
- [ ] `CHANGELOG.md` has v2.4.0 section with complete changelog
- [ ] `BREAKING_CHANGES.md` lists every removed/renamed API
- [ ] `AUDIT_MASTER_FIXES.md` maps every P0/P1/P2 to its fix
- [ ] `VERIFICATION.md` documents every "not a bug" claim from prior waves
- [ ] `_version.py` bumped to `2.4.0`
- [ ] `python -m build` succeeds
- [ ] `pip install dist/*.whl` succeeds
- [ ] `neuralops --version` returns `2.4.0`

### 1.5 — Final proof
- [ ] `pytest tests/unit tests/contract` green
- [ ] `pytest tests/integration` green (excluding `@pytest.mark.gpu` if
      no GPU)
- [ ] `ruff check neuralops/ tests/` green
- [ ] `mypy --strict neuralops/` green
- [ ] Security greps all empty (no `torch.cuda.amp`,
      `weights_only=False`, hardcoded `cuda:0`, `except: pass` without
      `# allowed-swallow:`)
- [ ] `audits/wave5_master.md` complete with per-agent verdicts

If ANY of the above fails, the human coordinator does NOT merge Wave 5.

---

## 2. PRE-FLIGHT

Before spawning any agent, run this. STOP if any check fails.

### 2.1 — Git state

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout main
git pull --ff-only origin main
```

Confirm clean tree:

```bash
if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: working tree is dirty"
    git status --porcelain
    exit 1
fi
```

### 2.2 — Prior wave evidence

For each wave that was supposed to run, check for evidence:

```bash
echo "=== Wave 1 evidence ==="
git log --oneline | grep -c "merge: wave1" || echo "0 wave1 merges"

echo "=== Wave 2 evidence ==="
git log --oneline | grep -c "merge: wave2" || echo "0 wave2 merges"

echo "=== Wave 3 evidence ==="
git log --oneline | grep -c "merge: wave3" || echo "0 wave3 merges"

echo "=== Wave 4 evidence ==="
git log --oneline | grep -c "merge: wave4" || echo "0 wave4 merges"

echo "=== Benchmarks 18-agent evidence ==="
git log --oneline | grep -c "merge: SA-B" || echo "0 benchmarks merges"
```

Record the actual counts. Do NOT assume a wave completed just because a
prior session claimed it. Wave 5's job is to verify, not to trust claims.

If a wave is missing, Wave 5's first 3 agents will investigate and either
confirm it as completed (finding evidence) or mark it BLOCKED (missing).

### 2.3 — Baseline test

```bash
pytest tests/ -x -q -m "not hub and not gpu" 2>&1 | tail -20
echo "Exit: $?"
```

If red: Wave 5 is blocked. Report which tests fail and stop.

### 2.4 — Baseline lint

```bash
ruff check neuralops/ tests/ 2>&1 | tail -20
echo "Exit: $?"
```

If red: Wave 5 is blocked. Report violations and stop.

### 2.5 — Create Wave 5 scaffolding

```bash
mkdir -p .agent/WAVE5_AGENTS
mkdir -p audits/wave5
mkdir -p issues
touch issues/wave6_hotfix_backlog.md
```

### 2.6 — Create progress file

```bash
cat > .agent/WAVE5_PROGRESS.json <<'EOF'
{
  "wave": 5,
  "started_at": "2026-11-15T00:00:00Z",
  "last_updated": "2026-11-15T00:00:00Z",
  "phase": 1,
  "agents": {
    "W5-1":  {"status": "pending", "branch": "wave5/W5-1",  "worktree": null},
    "W5-2":  {"status": "pending", "branch": "wave5/W5-2",  "worktree": null},
    "W5-3":  {"status": "pending", "branch": "wave5/W5-3",  "worktree": null},
    "W5-4":  {"status": "pending", "branch": "wave5/W5-4",  "worktree": null},
    "W5-5":  {"status": "pending", "branch": "wave5/W5-5",  "worktree": null},
    "W5-6":  {"status": "pending", "branch": "wave5/W5-6",  "worktree": null},
    "W5-7":  {"status": "pending", "branch": "wave5/W5-7",  "worktree": null},
    "W5-8":  {"status": "pending", "branch": "wave5/W5-8",  "worktree": null},
    "W5-9":  {"status": "pending", "branch": "wave5/W5-9",  "worktree": null},
    "W5-10": {"status": "pending", "branch": "wave5/W5-10", "worktree": null},
    "W5-11": {"status": "pending", "branch": "wave5/W5-11", "worktree": null},
    "W5-12": {"status": "pending", "branch": "wave5/W5-12", "worktree": null},
    "W5-13": {"status": "pending", "branch": "wave5/W5-13", "worktree": null},
    "W5-14": {"status": "pending", "branch": "wave5/W5-14", "worktree": null},
    "W5-15": {"status": "pending", "branch": "wave5/W5-15", "worktree": null},
    "W5-16": {"status": "pending", "branch": "wave5/W5-16", "worktree": null},
    "W5-17": {"status": "pending", "branch": "wave5/W5-17", "worktree": null},
    "W5-18": {"status": "pending", "branch": "wave5/W5-18", "worktree": null},
    "W5-19": {"status": "pending", "branch": "wave5/W5-19", "worktree": null},
    "W5-20": {"status": "pending", "branch": "wave5/W5-20", "worktree": null}
  }
}
EOF
```


---

## 3. GIT & ISOLATION PROTOCOL (BRANCHING & MERGE DISCIPLINE)

### 3.0 — Comprehensive Remediation Mandate (Fixing Related & Downstream Files)
**CRUCIAL INSTRUCTION FOR ALL AGENTS:**
Agents are NOT mere spectators. If an agent discovers that a related file, dependency, helper, adapter, or caller outside its primary file is broken, incomplete, or causes tests to fail, the agent is **EXPLICITLY AUTHORIZED AND MANDATED to fix and complete that related file as well**!
- Every test and integration path must pass 100%.
- Any fix to a related file must maintain backward compatibility, preserve AGENTS.md rules, and be documented in the agent's report under `## Remediated Related Files`.

### 3.1 — Strict Branch Isolation (No Direct Commits to Main)
Every agent MUST work in its own isolated branch (and worktree):
- Branch naming convention: `wave5/<AGENT_ID>` (e.g. `wave5/W5-1`, `wave5/W5-2`, ..., `wave5/W5-20`).
- Dedicated Worktree (if using worktrees): `../neuralops_<AGENT_ID>` (e.g. `../neuralops_W5_1`).
- **Forbidden:** No subagent is permitted to commit directly to `main`. Working on `main` is strictly forbidden for subagents.

### 3.2 — Commit Message Format
Every commit by an agent must follow the standard imperative scope syntax:
```bash
git commit -m "audit(wave5/<AGENT_ID>): <imperative summary, <=72 chars>"
# or for release artifacts:
git commit -m "release(wave5/<AGENT_ID>): <imperative summary, <=72 chars>"
```

### 3.3 — Orchestrator Merge Protocol (Phase-by-Phase Integration into Main)
Subagents do NOT merge their own branches. The orchestrator collects and merges them sequentially after verifying acceptance criteria:

1. **Verify Agent Completion:**
   - Confirm branch `wave5/<AGENT_ID>` is clean and tests pass.
   - Confirm all strictly owned files and evidence files are produced.
   - Confirm zero unowned files were touched.
2. **Merge into Main:**
   ```bash
   git checkout main
   git merge --no-ff wave5/<AGENT_ID> -m "merge(wave5): integrate <AGENT_ID> (<Description>)"
   ```
3. **Verify Main Health:**
   ```bash
   pytest tests/ -x -q -m "not hub and not gpu"
   ruff check neuralops/ tests/
   ```
4. **Update Progress File:**
   - Update `.agent/WAVE5_PROGRESS.json` setting `status: "complete"`.
5. **Teardown Worktree / Delete Feature Branch:**
   ```bash
   git branch -d wave5/<AGENT_ID>
   ```

---

## 4. SPAWN PROTOCOL & AGENT TEMPLATE

When invoking a subagent for Wave 5:
- Spec file: `.agent/WAVE5_AGENTS/<AGENT_ID>.md`
- Working Directory: Workspace repository root or worktree
- Owned files: Exactly as specified in Section 1.1 of the agent spec.
- Rules: Never modify unowned files inline; always generate required evidence in `audits/wave5/w5_<N>_evidence/`.

---

## 5. COMPLETE AGENT SPECIFICATIONS (W5-1 THROUGH W5-20)

### Subagent W5-1
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-1.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-1.md)

# Wave 5 Agent W5-1: Forensic Lineage & Integrity Verification of Waves 1 & 2

**Agent ID:** W5-1  
**Phase:** 1 — Prior Wave Verification  
**Branch:** `wave5/W5-1`  
**Mission:** Perform an uncompromising forensic audit of all 45 P0/P1 defect fixes from Wave 1 and all 20 subsystem remediations from Wave 2 across the codebase. Verify that no commit was dropped, no merge conflict clobbered logic, and that every single related test file passes 100%.

---

## 1. Context, Architecture & Mandatory Boundaries

In large-scale agentic code refactoring, Git branches can inadvertently overwrite or revert prior changes during multi-branch rebasing or fast-forward merges. 

As Agent **W5-1**, you are the independent forensic auditor. You must verify that the 10 subagents from Wave 1 and the 20 subagents from Wave 2 actually left their permanent, working imprint in `main`.

### Explicitly Owned Paths (Files You Create / Write):
- `audits/wave5/w5_1_wave1_wave2.md` (Comprehensive forensic audit report)
- `audits/wave5/w5_1_evidence/` (Directory storing raw command outputs, git graphs, and pytest logs)
- `audits/wave5/w5_1_evidence/wave1_merges.txt`
- `audits/wave5/w5_1_evidence/wave2_merges.txt`
- `audits/wave5/w5_1_evidence/wave1_commits.txt`
- `audits/wave5/w5_1_evidence/wave2_commits.txt`
- `audits/wave5/w5_1_evidence/defect_inspection.txt`
- `audits/wave5/w5_1_evidence/test_execution.log`

### Related Code Files to Inspect & Verify (Do NOT Edit Directly; Audit & Test):
1. **Wave 1 / Wave 2 Models Core:**
   - `neuralops/models/cache.py` (UnifiedHybridCache reorder & seen tokens)
   - `neuralops/models/modeling.py` (Kwargs filtering, polymorphic cache dispatch)
   - `neuralops/models/cuda_graph.py` (Static KV-cache TypeError contract)
   - `neuralops/models/gla_hf_model.py` (GenerationMixin removal, GLAHFModel alias)
   - `neuralops/models/modeling_norm.py` (NeuralOpsRMSNorm dtype preservation)
   - `neuralops/models/configuration.py` (O(1) cached attn specs)
2. **Models Surgery & IO:**
   - `neuralops/models/surgery.py`
   - `neuralops/models/surgical_io.py` (save_surgical_delta key preservation, weights_only=True)
3. **Surgery Core & Advanced:**
   - `neuralops/surgery/director.py` (certify_model 10 real invariants)
   - `neuralops/surgery/fx_surgery.py`
   - `neuralops/surgery/layer_replacement.py`
   - `neuralops/surgery/pipeline.py`
   - `neuralops/surgery/universal_calibrator.py`
   - `neuralops/surgery/validation.py` (validate_before_surgery zero-fake estimates)
4. **Pipeline Transform:**
   - `neuralops/pipeline_transform/master_transformer.py`
   - `neuralops/pipeline_transform/hardware_planner.py`
   - `neuralops/pipeline_transform/executor.py`
5. **Hub & Recipe:**
   - `neuralops/hub/uploader.py` (SHA-256 validation)
   - `neuralops/recipe/engine.py`
6. **Validation Subsystem:**
   - `neuralops/validation/empirical_gate.py` (EmpiricalBenchmarkGate real accuracy)
   - `neuralops/validation/numerical_validator.py` (check_finite NaN/Inf scan)
7. **Serving & Runtime:**
   - `neuralops/serving/request_batcher.py` (Sequential generation concurrency safety)
   - `neuralops/tracking/backends.py` (LocalTracker bounded memory)

### Related Test Suites to Execute:
- `tests/models/test_cache_reorder.py`
- `tests/models/test_cache_seen_tokens.py`
- `tests/models/test_cuda_graph_static.py`
- `tests/models/test_model_kwargs_filter.py`
- `tests/models/test_rmsnorm_dtype.py`
- `tests/models/test_gla_generate.py`
- `tests/models/test_surgical_io_no_delete.py`
- `tests/models/test_surgical_io_strict.py`
- `tests/surgery/test_surgery_core_wave2_regression.py`
- `tests/surgery/test_wave2_b2_regressions.py`
- `tests/pipeline_transform/test_master_transformer.py`
- `tests/validation/test_validation_fixes.py`

---

## 2. Exhaustive Step-by-Step Execution Plan

### Step 1: Scaffolding Setup
Create the evidence directory:
```bash
mkdir -p audits/wave5/w5_1_evidence/
```

### Step 2: Git Lineage Forensic Investigation
1. **Wave 1 Merge Log Extraction:**
   Run git log queries to trace merge commits and individual agent commits:
   ```bash
   git log --oneline --grep="merge: wave1" > audits/wave5/w5_1_evidence/wave1_merges.txt || true
   git log --oneline --grep="fix(wave1" > audits/wave5/w5_1_evidence/wave1_commits.txt || true
   ```
2. **Wave 2 Merge Log Extraction:**
   ```bash
   git log --oneline --grep="merge: wave2" > audits/wave5/w5_1_evidence/wave2_merges.txt || true
   git log --oneline --grep="fix(wave2" > audits/wave5/w5_1_evidence/wave2_commits.txt || true
   git log --oneline --grep="wave2" >> audits/wave5/w5_1_evidence/wave2_commits.txt || true
   ```
3. **Verify Presence of 20 Wave 2 Audit Files:**
   Check that every single report in `audits/wave2/` is present and committed:
   ```bash
   ls -la audits/wave2/ > audits/wave5/w5_1_evidence/wave2_reports_list.txt
   ```
   Verify that `audits/wave2_master.md` exists and is non-empty.

### Step 3: Deep Invariant Inspection of Fixed Code
Inspect critical code points directly in Python to confirm no silent reversion:
```bash
python -c "
import inspect
from neuralops.models.cache import UnifiedHybridCache
from neuralops.models.gla_hf_model import GLAModel, GLAHFModel
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner
from neuralops.surgery.director import UniversalSurgeryDirector
from neuralops.validation.empirical_gate import EmpiricalBenchmarkGate
from transformers import GenerationMixin

results = []

# 1. UnifiedHybridCache reorder_cache and __iter__
cache = UnifiedHybridCache(2)
assert hasattr(cache, 'reorder_cache'), 'FAIL: UnifiedHybridCache missing reorder_cache'
assert hasattr(cache, '__iter__'), 'FAIL: UnifiedHybridCache missing __iter__'
assert getattr(cache, 'is_static', False) is False, 'FAIL: UnifiedHybridCache.is_static must be False'
results.append('UnifiedHybridCache: PASS')

# 2. GLAModel / GLAHFModel GenerationMixin
assert not issubclass(GLAModel, GenerationMixin), 'FAIL: GLAModel must NOT inherit GenerationMixin'
assert not issubclass(GLAHFModel, GenerationMixin), 'FAIL: GLAHFModel must NOT inherit GenerationMixin'
results.append('GLAModel GenerationMixin: PASS')

# 3. certify_model no hardcoded bypasses
source = inspect.getsource(UniversalSurgeryDirector.certify_model)
assert 'True  # bypass' not in source, 'FAIL: certify_model contains hardcoded bypass'
results.append('certify_model: PASS')

# 4. EmpiricalBenchmarkGate no dummy values
gate_source = inspect.getsource(EmpiricalBenchmarkGate.evaluate)
assert 'return 0.95' not in gate_source, 'FAIL: EmpiricalBenchmarkGate has fake return'
results.append('EmpiricalBenchmarkGate: PASS')

with open('audits/wave5/w5_1_evidence/defect_inspection.txt', 'w') as f:
    f.write('\n'.join(results) + '\n')
print('Physical Code Invariant Check Passed!')
"
```

### Step 4: Full Unit & Regression Test Suite Execution
Execute all regression test suites corresponding to Wave 1 and Wave 2:
```bash
./.venv/bin/pytest tests/models/ tests/surgery/ tests/pipeline_transform/ tests/validation/ tests/core/ -v > audits/wave5/w5_1_evidence/test_execution.log 2>&1
```
Extract summary counts (`passed`, `failed`, `skipped`).

### Step 5: Handling Discoveries & Wave 6 Hotfix Logging
If any test fails or any code invariant check fails:
1. Do NOT modify the source files directly.
2. File a formal ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (e.g. `W5-1-F01`)
   - Affected File and Line
   - Exact Traceback
   - Severity (P0 Blocker or P1 Major)
   - Recommended Fix for Wave 6.

### Step 6: Compile Formal Report `audits/wave5/w5_1_wave1_wave2.md`
Author the final markdown document containing:
- Executive Summary and Overall Verdict (`COMPLETE` or `BLOCKED`).
- Lineage Table for Wave 1 Agents (A1–A5, B1–B5).
- Lineage Table for Wave 2 Agents (A1, A2, B1, B2, C1, D1, E1, E2, F1–F4, G1–G3, H1, H2, I1, J1, K1).
- Code Invariant Physical Verification Matrix (All targeted P0 bugs).
- Test Execution Summary (Total passed, skipped, failed).
- Complete Listing of any findings logged to Wave 6 backlog.

---

## 3. Acceptance Criteria
- [ ] `wave1_merges.txt`, `wave2_merges.txt`, `defect_inspection.txt`, and `test_execution.log` created in `audits/wave5/w5_1_evidence/`.
- [ ] All code invariant assertions pass with exit code 0.
- [ ] Over 100 unit/regression tests pass without unhandled errors.
- [ ] Report `audits/wave5/w5_1_wave1_wave2.md` written and committed.


---

### Subagent W5-2
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-2.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-2.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-2
# Forensic Audit & Verification of Waves 3 and 4

**Agent ID:** `W5-2`  
**Role:** Prior Wave Verification (Wave 3 & Wave 4 Forensic Auditor)  
**Wave:** 5 (Integration, Verification, and Release)  
**Phase:** Phase 1 — Prior Wave Verification  
**Git Branch:** `wave5/W5-2`  
**Isolated Worktree Path:** `../Project_W5_2`  
**Progress Tracker:** `.agent/WAVE5_PROGRESS.json`  

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF AUDIT

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_2_wave3_wave4.md` (Forensic report documenting status of Waves 3 & 4)
- `audits/wave5/w5_2_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_2_evidence/wave3_branches.txt`
- `audits/wave5/w5_2_evidence/wave3_commits.txt`
- `audits/wave5/w5_2_evidence/wave3_files.txt`
- `audits/wave5/w5_2_evidence/wave4_branches.txt`
- `audits/wave5/w5_2_evidence/wave4_commits.txt`
- `audits/wave5/w5_2_evidence/wave4_worktrees.txt`
- `audits/wave5/w5_2_evidence/wave4_progress_dump.json`
- `audits/wave5/w5_2_evidence/unmerged_remediation_analysis.txt`

### 1.2 Files to Inspect & Audit (Do NOT Modify Inline)
- `.agent/ORCHESTRATOR_WAVE3.md` (Wave 3 orchestrator definition)
- `.agent/WAVE4_EXECUTE.md` (Wave 4 master execution plan)
- `.agent/WAVE4_PROGRESS.json` (Wave 4 tracking progress ledger)
- `.agent/WAVE4_AGENTS/*.md` (All 20 Wave 4 agent task files: W4-L1 to W4-U2)
- All 20 isolated worktrees at `../Project_W4_*`
- `audits/wave2_master.md` (Wave 2 master report to check overlap)

---

## 2. THE MISSION & THE INVARIANTS

Wave 5 assumes Waves 1–4 are either merged or have an explicit status. In prior sessions, Wave 4 spawned 20 worktrees (`/Users/tahamajs/dev/Project_W4_*`) and branches (`wave4/W4-*`), but the user may have moved directly to integration.
Your mission is to perform a strict forensic determination:
1. Did Wave 3 merge, partially run, or was it superseded?
2. Did Wave 4 agents commit to their branches? Are their worktrees clean or dirty?
3. Did Wave 4 merge into `main` or `fix/benchmarks-remediation`?
4. Are there critical unmerged fixes in Wave 4 branches that must be merged or preserved?
5. State unequivocally whether the release is `READY_TO_PROCEED` (if fixes were already in Wave 2) or `BLOCKED` (if critical unmerged commits exist).

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_2_evidence/
```

### Step 2: Wave 3 Forensic Investigation
1. **Branch Query:**
   ```bash
   git branch -a | grep -i "wave3" > audits/wave5/w5_2_evidence/wave3_branches.txt || true
   ```
2. **Commit Query:**
   ```bash
   git log --oneline --grep="wave3" > audits/wave5/w5_2_evidence/wave3_commits.txt || true
   ```
3. **File Discovery:**
   ```bash
   find .agent audits docs -name "*wave3*" -o -name "*WAVE3*" > audits/wave5/w5_2_evidence/wave3_files.txt || true
   ```
4. Read `.agent/ORCHESTRATOR_WAVE3.md`. Determine if Wave 3 was executed as a separate wave or if its 20 agents were merged into Wave 2's 20-agent master remediation.

### Step 3: Wave 4 Forensic Investigation
1. **Inspect Progress File:**
   ```bash
   cp .agent/WAVE4_PROGRESS.json audits/wave5/w5_2_evidence/wave4_progress_dump.json
   ```
2. **Worktree Status Analysis:**
   List all worktrees on the machine:
   ```bash
   git worktree list | grep "wave4" > audits/wave5/w5_2_evidence/wave4_worktrees.txt || true
   ```
3. **Inspect Every Single Wave 4 Branch:**
   Execute a python script iterating through all 20 Wave 4 agents (W4-L1 to W4-U2):
   ```bash
   python -c "
   import subprocess, json

   agents = [
       'W4-L1', 'W4-L2', 'W4-L3', 'W4-L4', 'W4-O1', 'W4-O2', 'W4-C1', 'W4-T1',
       'W4-T2', 'W4-T3', 'W4-A1', 'W4-C2', 'W4-E1', 'W4-U1', 'W4-U2', 'W4-B1',
       'W4-B2', 'W4-B3', 'W4-I1', 'W4-R1'
   ]

   report = []
   for a in agents:
       branch = f'wave4/{a}'
       # Check commit count relative to main
       cmd = f'git rev-list --count main..{branch} 2>/dev/null || echo \"NO_BRANCH\"'
       count = subprocess.check_output(cmd, shell=True).decode().strip()
       report.append(f'{a} ({branch}): {count} commits ahead of main')

   with open('audits/wave5/w5_2_evidence/unmerged_remediation_analysis.txt', 'w') as f:
       f.write('\n'.join(report) + '\n')
   print('Wave 4 branch audit completed.')
   "
   ```

### Step 4: Overlap Analysis with Wave 2
Compare the tasks assigned to Wave 4 in `.agent/WAVE4_EXECUTE.md` against what Agent A1 through K1 already committed in Wave 2 (`audits/wave2_master.md`).
- If Wave 2 already resolved the P0/P1 defects across models, surgery, layers, and compression, document that Wave 4 is redundant or superseded.
- If unmerged commits exist in `wave4/*` branches, list the exact files they touched.

### Step 5: Produce Authoritative Report `audits/wave5/w5_2_wave3_wave4.md`
Structure the report:
1. **Executive Verdict on Wave 3:** `SUPERSEDED_BY_WAVE2` or `SKIPPED_NOT_STARTED` with evidence.
2. **Executive Verdict on Wave 4:** `IN_PROGRESS_UNMERGED` or `SUPERSEDED_BY_WAVE2` with commit tallies.
3. **Worktree Inventory Table:** List all 20 worktree paths and their HEAD commits.
4. **Release Blocker Determination:** Explicitly answer: Does v2.4.0 require merging Wave 4 branches, or is the baseline in `main`/`fix/benchmarks-remediation` already verified green?
5. **Backlog Logging:** Log any outstanding tasks in `issues/wave6_hotfix_backlog.md`.

---

## 4. ACCEPTANCE CRITERIA
- [ ] All evidence files populated in `audits/wave5/w5_2_evidence/`.
- [ ] Complete commit count audit for all 20 Wave 4 branches.
- [ ] `audits/wave5/w5_2_wave3_wave4.md` written and committed with unambiguous determination.


---

### Subagent W5-3
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-3.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-3.md)

# Wave 5 Agent W5-3: Forensic Verification & Anti-Pattern Audit of Benchmarks Remediation (SA-B1 to SA-B18)

**Agent ID:** W5-3  
**Phase:** 1 — Prior Wave Verification  
**Branch:** `wave5/W5-3`  
**Mission:** Conduct a comprehensive forensic audit of the entire benchmark subsystem (`neuralops/benchmarks/`), verifying that all 18 benchmark subagents (SA-B1 through SA-B18) completed their mandates, eliminated all mock/fake benchmark loops, decomposed monolithic scripts into `cli_v2`, enforced hardware isolation, and that all benchmark test suites pass.

---

## 1. Context, Architecture & Mandatory Boundaries

The benchmark subsystem is the empirical authority of NeuralOps. In earlier versions, several benchmarks suffered from severe anti-patterns:
- Synthetic random evaluations returning fake accuracy (e.g. `niah.py` calculating needle retrieval on random tensors).
- Monolithic CLI scripts exceeding 800 lines with module-level side effects.
- Deprecated PyTorch APIs (`torch.cuda.amp.autocast`).
- Silent exception swallowing (`except Exception: pass`).
- Hardcoded `cuda:0` device allocations crashing on CPU or multi-GPU nodes.

As Agent **W5-3**, you must independently verify that these anti-patterns have been permanently eradicated and that the new architectural components (`config_v2/`, `suites_v2/`, `cli_v2/`, `results_v2/`) satisfy all operational rules from `AGENTS.md`.

### Explicitly Owned Paths (Files You Create / Write):
- `audits/wave5/w5_3_benchmarks.md` (Forensic benchmarks audit report)
- `audits/wave5/w5_3_evidence/` (Evidence directory)
- `audits/wave5/w5_3_evidence/merges.txt`
- `audits/wave5/w5_3_evidence/agent_commits.txt`
- `audits/wave5/w5_3_evidence/fake_benchmarks.txt`
- `audits/wave5/w5_3_evidence/deprecated_amp.txt`
- `audits/wave5/w5_3_evidence/hardcoded_cuda.txt`
- `audits/wave5/w5_3_evidence/swallowed_exceptions.txt`
- `audits/wave5/w5_3_evidence/stray_mains.txt`
- `audits/wave5/w5_3_evidence/cli_v2_structure.txt`
- `audits/wave5/w5_3_evidence/pytest_benchmarks.log`

### Related Code Files to Inspect & Audit (Do NOT Edit Directly):
1. **Benchmark CLI & Entrypoints:**
   - `neuralops/benchmarks/cli.py` (Must be thin backward-compatibility shim < 100 lines)
   - `neuralops/benchmarks/cli_v2/main.py` (Unified CLI entrypoint)
   - `neuralops/benchmarks/cli_v2/commands/*.py` (Decomposed subcommands: run, list, sweep, doctor, compare)
2. **Benchmark Configuration (Pydantic v2):**
   - `neuralops/benchmarks/config_v2/schema.py`
   - `neuralops/benchmarks/config_v2/loader.py`
3. **Execution Suites & Runners:**
   - `neuralops/benchmarks/executor.py`
   - `neuralops/benchmarks/suites/niah.py` (Verify zero fake random scores)
   - `neuralops/benchmarks/suites/qwen_kd.py` (Verify honest implementation or NotImplementedError)
   - `neuralops/benchmarks/suites/e2e_serving.py` (Verify honest implementation or NotImplementedError)
   - `neuralops/benchmarks/suites/attention.py`
   - `neuralops/benchmarks/suites/kernel_suite.py`
4. **Results Ledger & Index:**
   - `neuralops/benchmarks/results_v2/record.py` (CanonicalRecord definition)
   - `neuralops/benchmarks/results_v2/ledger.py` (Atomic Parquet, JSONL, SQLite persistence)
   - `neuralops/benchmarks/results_v2/index.py` (Analytical querying)
5. **Hardware Utilities:**
   - `neuralops/benchmarks/utils/memory.py`
   - `neuralops/benchmarks/utils/timer.py` (Verify torch.cuda.Event usage, L2 cache flushing)

### Related Test Suites to Execute:
- `tests/benchmarks/test_cli_v2.py`
- `tests/benchmarks/test_config_v2.py`
- `tests/benchmarks/test_ledger_v2.py`
- `tests/benchmarks/test_timer.py`
- `tests/benchmarks/test_suites.py`
- All other test files in `tests/benchmarks/`

---

## 2. Exhaustive Step-by-Step Execution Plan

### Step 1: Scaffolding Setup
Create the evidence directory:
```bash
mkdir -p audits/wave5/w5_3_evidence/
```

### Step 2: Commit Lineage & 18-Agent Verification Matrix
1. **Extract SA-B Merge and Agent Commits:**
   ```bash
   git log --oneline --grep="merge: SA-B" > audits/wave5/w5_3_evidence/merges.txt || true
   git log --oneline --grep="fix(benchmarks/SA-B" > audits/wave5/w5_3_evidence/agent_commits.txt || true
   git log --oneline --grep="SA-B" >> audits/wave5/w5_3_evidence/agent_commits.txt || true
   ```
2. **Enumerate and Map Commits for Each Agent (SA-B1 through SA-B18):**
   - Check commit history for each agent ID and verify their primary deliverables.
   - Record commit count and status in an audit matrix.

### Step 3: Automated Anti-Pattern and Zero-Fake Audit
Execute systematic greps across `neuralops/benchmarks/` to enforce operational rules:
1. **Zero Fake Random Evaluations (Rule 9.4 & Zero-Fake Rule):**
   ```bash
   grep -rn "torch.randn.*needle\|torch.randn.*teacher_logits\|torch.rand.*score" neuralops/benchmarks/suites/ > audits/wave5/w5_3_evidence/fake_benchmarks.txt || true
   ```
   *Expectation:* 0 lines. Any dataset requirement must either load real data or explicitly raise `NotImplementedError("Real dataset required...")`.
2. **AMP Deprecation Scan (Rule 2.1):**
   ```bash
   grep -rn "torch.cuda.amp" neuralops/benchmarks/ > audits/wave5/w5_3_evidence/deprecated_amp.txt || true
   ```
   *Expectation:* 0 lines. Modern `torch.amp.autocast("cuda", ...)` must be used everywhere.
3. **Hardcoded CUDA Device Scan (Rule 2.5):**
   ```bash
   grep -rn "cuda:0" neuralops/benchmarks/ | grep -v "default=\"cuda:0\"" | grep -v "test_" > audits/wave5/w5_3_evidence/hardcoded_cuda.txt || true
   ```
   *Expectation:* 0 lines. Must use `resolve_device(device)` or dynamic config parameters.
4. **Silent Exception Swallowing Scan (Rule 10):**
   ```bash
   grep -rn "except.*:.*pass" neuralops/benchmarks/ | grep -v "allowed-swallow" > audits/wave5/w5_3_evidence/swallowed_exceptions.txt || true
   ```
   *Expectation:* 0 lines. Every swallowed exception must include an explicit `# allowed-swallow: <reason>`.
5. **Main Functions in Library Files (Rule 1.4 & 1.5):**
   ```bash
   grep -rn "if __name__ == .__main__.:" neuralops/benchmarks/ | grep -v "cli_v2/main.py" > audits/wave5/w5_3_evidence/stray_mains.txt || true
   ```
   *Expectation:* 0 lines. Entrypoints live exclusively in `cli_v2/main.py`.

### Step 4: CLI Architecture Inspection
1. **Verify Line Count of Legacy `cli.py`:**
   ```bash
   wc -l neuralops/benchmarks/cli.py > audits/wave5/w5_3_evidence/cli_v2_structure.txt
   ```
   *Expectation:* Under 100 lines (delegating to `cli_v2.main`).
2. **Verify Decomposed Subcommand Structure:**
   ```bash
   ls -la neuralops/benchmarks/cli_v2/commands/ >> audits/wave5/w5_3_evidence/cli_v2_structure.txt
   ```
   Confirm presence of `run.py`, `list.py`, `sweep.py`, `doctor.py`, `compare.py`.

### Step 5: Test Execution Across `tests/benchmarks/`
Execute the entire benchmarks test suite:
```bash
./.venv/bin/pytest tests/benchmarks/ -v > audits/wave5/w5_3_evidence/pytest_benchmarks.log 2>&1
```
Parse total passed, skipped, and failed tests.

### Step 6: Logging Wave 6 Hotfix Items
If any anti-pattern is detected or any test fails:
1. Do NOT modify source files directly.
2. File a ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (`W5-3-F01`, etc.)
   - Affected File and Line Number
   - Exact Description and Violation Code
   - Proposed Fix.

### Step 7: Compile Formal Benchmark Audit Report `audits/wave5/w5_3_benchmarks.md`
Author the complete markdown audit report containing:
- Executive Summary and Determination (`COMPLETE` or `BLOCKED`).
- 18-Agent Remediation Scorecard (SA-B1 to SA-B18 status and commits).
- Anti-Pattern Grep Findings Summary Table.
- CLI v2 Architecture Assessment.
- Benchmark Test Suite Results.
- Consolidated List of Wave 6 Backlog Items.

---

## 3. Acceptance Criteria
- [ ] Evidence files generated in `audits/wave5/w5_3_evidence/`.
- [ ] Anti-pattern scans executed with zero unannotated violations.
- [ ] Full `pytest tests/benchmarks/` execution log captured.
- [ ] Report `audits/wave5/w5_3_benchmarks.md` written and committed.


---

### Subagent W5-4
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-4.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-4.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-4
# End-to-End Pipeline Integration (Config v2 → Suites v2 → Ledger v2 → Index v2)

**Agent ID:** `W5-4`  
**Role:** End-to-End Pipeline Integration Architect  
**Wave:** 5 (Integration, Verification, and Release)  
**Phase:** Phase 2 — End-to-End Integration  
**Git Branch:** `wave5/W5-4`  
**Isolated Worktree Path:** `../Project_W5_4`  
**Progress Tracker:** `.agent/WAVE5_PROGRESS.json`  

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF INTEGRATION

### 1.1 Strictly Owned Files (You Create & Commit These)
- `tests/integration/test_config_to_results.py` (New end-to-end integration test suite)
- `audits/wave5/w5_4_pipeline.md` (Forensic pipeline integration report)
- `audits/wave5/w5_4_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_4_evidence/test_execution.log`
- `audits/wave5/w5_4_evidence/sqlite_schema_dump.txt`
- `audits/wave5/w5_4_evidence/jsonl_sample_record.json`
- `audits/wave5/w5_4_evidence/index_query_results.txt`

### 1.2 Subsystems & Files Under Integration Test (Do NOT Modify Inline)
1. **Config v2 Engine:**
   - `neuralops/benchmarks/config_v2/schema.py` (`BenchmarkConfig`, `RunConfig`, `SuiteConfig`)
   - `neuralops/benchmarks/config_v2/loader.py` (`load_benchmark_config`, `parse_yaml_string`)
2. **Execution Suites & Runners:**
   - `neuralops/benchmarks/executor.py` (`BenchmarkExecutor`, `SubprocessWorker`)
   - `neuralops/benchmarks/suites/attention.py` (`AttentionBenchmarkSuite`)
   - `neuralops/benchmarks/suites/kernel_suite.py`
3. **Results v2 Ledger & Index:**
   - `neuralops/benchmarks/results_v2/record.py` (`CanonicalRecord`, `BenchmarkStatus`)
   - `neuralops/benchmarks/results_v2/ledger.py` (`BenchmarkLedger`)
   - `neuralops/benchmarks/results_v2/index.py` (`ResultsIndex`)
4. **CLI v2 Integration Hooks:**
   - `neuralops/benchmarks/cli_v2/commands/run.py`
   - `neuralops/benchmarks/cli_v2/commands/compare.py`

---

## 2. THE MISSION & THE INVARIANTS

Individual modules were refactored in isolation. However, real bugs happen at the seams:
- Does `config_v2/loader.py` emit a configuration that `executor.py` can parse without `KeyError`?
- Does `AttentionBenchmarkSuite` generate a `CanonicalRecord` matching every field in `results_v2/record.py`?
- Does `BenchmarkLedger` write to Parquet, JSONL, and SQLite concurrently without database file locks or corrupted JSON?
- Can `ResultsIndex.query()` accurately filter and return the written record by `run_id`, `git_commit`, and `suite_name`?

You must prove this entire pipeline functions end-to-end.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_4_evidence/
```

### Step 2: Implement `tests/integration/test_config_to_results.py`
Create a comprehensive integration test covering the complete pipeline:

```python
import json
import os
import sqlite3
import pytest
import torch
from pathlib import Path
from neuralops.benchmarks.config_v2.loader import load_benchmark_config
from neuralops.benchmarks.config_v2.schema import BenchmarkConfig
from neuralops.benchmarks.results_v2.ledger import BenchmarkLedger
from neuralops.benchmarks.results_v2.record import CanonicalRecord, BenchmarkStatus
from neuralops.benchmarks.results_v2.index import ResultsIndex
from neuralops.benchmarks.suites.attention import AttentionBenchmarkSuite


def test_full_benchmark_lifecycle_end_to_end(tmp_path: Path):
    """
    Test full lifecycle:
    YAML -> BenchmarkConfig -> Suite Execution -> CanonicalRecord -> BenchmarkLedger -> ResultsIndex
    """
    # 1. Generate YAML configuration
    yaml_content = f"""
run_id: "test-run-e2e-001"
output_dir: "{tmp_path}/results"
device: "cpu"
suites:
  - name: "attention_sweep"
    type: "attention"
    attentions:
      - "cosformer"
    batch_sizes: [1]
    seq_lens: [64]
    warmup_iters: 2
    num_iters: 3
"""
    config_file = tmp_path / "benchmark_config.yaml"
    config_file.write_text(yaml_content)

    # 2. Parse YAML via config_v2 loader
    config = load_benchmark_config(str(config_file))
    assert isinstance(config, BenchmarkConfig)
    assert config.run_id == "test-run-e2e-001"

    # 3. Initialize Ledger
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger = BenchmarkLedger(storage_dir=str(results_dir))

    # 4. Execute Attention Benchmark Suite
    suite = AttentionBenchmarkSuite(config=config)
    records = suite.run()
    assert len(records) >= 1
    record = records[0]
    assert isinstance(record, CanonicalRecord)
    assert record.status == BenchmarkStatus.SUCCESS.value
    assert "latency_ms" in record.metrics
    assert record.metrics["latency_ms"] > 0

    # 5. Persist to Ledger
    ledger.write(record)

    # 6. Verify Ledger JSONL Output
    jsonl_path = results_dir / "results.jsonl"
    assert jsonl_path.exists()
    with open(jsonl_path) as f:
        lines = f.readlines()
        assert len(lines) >= 1
        data = json.loads(lines[-1])
        assert data["run_id"] == "test-run-e2e-001"
        assert data["metrics"]["latency_ms"] > 0

    # 7. Verify SQLite Database Output
    db_path = results_dir / "ledger.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT run_id, suite, status FROM records WHERE run_id = ?", ("test-run-e2e-001",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test-run-e2e-001"
        conn.close()

    # 8. Query via ResultsIndex
    index = ResultsIndex(storage_dir=str(results_dir))
    queried_records = index.find(run_id="test-run-e2e-001")
    assert len(queried_records) >= 1
    assert queried_records[0].run_id == "test-run-e2e-001"
```

### Step 3: Run the Integration Test
```bash
./.venv/bin/pytest tests/integration/test_config_to_results.py -v > audits/wave5/w5_4_evidence/test_execution.log 2>&1
```

### Step 4: Extract Physical Evidence
1. **Sample JSONL Record:**
   Extract and format the written record into `audits/wave5/w5_4_evidence/jsonl_sample_record.json`.
2. **SQLite Schema Inspection:**
   Query table schemas in `ledger.db` and output to `audits/wave5/w5_4_evidence/sqlite_schema_dump.txt`.
3. **Query Engine Verification:**
   Verify index query latency and output to `audits/wave5/w5_4_evidence/index_query_results.txt`.

### Step 5: Handle Discoveries & Log Wave 6 Hotfix Items
If the test uncovers a serialization mismatch, missing column, or database locking issue:
1. Do NOT modify source files directly.
2. File a ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (`W5-4-F01`, etc.)
   - Calling Module and Failing Interface
   - Exact Traceback and Schema Dump
   - Severity and Recommended Fix.
3. Decorate test with `@pytest.mark.xfail(reason="...")`.

### Step 6: Compile Formal Report `audits/wave5/w5_4_pipeline.md`
Author the complete markdown document containing:
- Executive Summary and Integration Status (`PASS` or `FAIL`).
- End-to-End Architectural Dataflow Diagram.
- Schema Compatibility Matrix (Config v2 Schema vs CanonicalRecord vs SQLite columns).
- Ledger Persistence and Query Outturn.
- Logged Wave 6 Hotfix Tickets.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `tests/integration/test_config_to_results.py` created and executing.
- [ ] Evidence files created in `audits/wave5/w5_4_evidence/`.
- [ ] End-to-end pipeline passes or has documented xfail with ticket in `issues/wave6_hotfix_backlog.md`.
- [ ] Report `audits/wave5/w5_4_pipeline.md` written and committed.


---

### Subagent W5-5
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-5.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-5.md)

# Wave 5 Agent W5-5: End-to-End HuggingFace Model Zoo Integration & Surgery Verification

**Agent ID:** W5-5  
**Phase:** 2 — End-to-End Integration  
**Branch:** `wave5/W5-5`  
**Mission:** Prove through a dedicated end-to-end integration test suite that all six (6) canonical HuggingFace model architectures supported by NeuralOps (BERT Attention, BERT Compressor, Qwen Attention, Qwen Compressor, TinyLlama, and LLaVA Vision-Language) load, successfully apply model surgery / weight transformations, and execute forward passes producing finite, dimensionally accurate outputs.

---

## 1. Context, Architecture & Mandatory Boundaries

NeuralOps specializes in foundation model surgery: swapping quadratic attention blocks for linear/recurrent kernels, applying low-rank SVD decompositions, and quantizing feedforward projections. 

While unit tests test synthetic `nn.Linear` layers, real HuggingFace models introduce subtle architectural complexities:
- Tied embeddings (`lm_head.weight == embed_tokens.weight`).
- Grouped-Query Attention (GQA) where query head counts exceed key-value head counts.
- Rotary Position Embeddings (RoPE) and complex KV caching structures.
- Multimodal projections between vision towers and language decoder backbones.

As Agent **W5-5**, you will create and execute an integration test covering all six model archetypes to prove end-to-end functionality without requiring multi-gigabyte internet downloads.

### Explicitly Owned Paths (Files You Create / Write):
- `tests/integration/test_6_zoos.py` (New end-to-end test suite)
- `audits/wave5/w5_5_zoos.md` (Zoo integration audit report)
- `audits/wave5/w5_5_evidence/` (Evidence directory)
- `audits/wave5/w5_5_evidence/test_execution.log`
- `audits/wave5/w5_5_evidence/tensor_shapes_and_memory.txt`

### Related Code Files to Test & Audit (Do NOT Edit Directly):
1. **Public APIs:**
   - `neuralops/api.py` (`replace_attention`, `compress`, `certify`)
   - `neuralops/hub/model_loader.py` (`load_surgeried_model`, `load_pretrained`)
2. **Models & Layers:**
   - `neuralops/models/surgery.py`
   - `neuralops/models/surgical_io.py`
   - `neuralops/layers/linear/cosformer.py`
   - `neuralops/layers/linear/gla_attention.py`
   - `neuralops/compression/svd/direct_svd.py`
   - `neuralops/compression/quantization/kmeans_quantize.py`
3. **Architecture Adapters:**
   - `neuralops/surgery/attention/universal_adapter.py`
   - `neuralops/surgery/fx_surgery.py`

---

## 2. Exhaustive Step-by-Step Execution Plan

### Step 1: Scaffolding Setup
Create the evidence directory:
```bash
mkdir -p audits/wave5/w5_5_evidence/
```

### Step 2: Implement `tests/integration/test_6_zoos.py`
Author the integration test file implementing comprehensive tests for all 6 model zoo configurations:

1. **Zoo 1: `bert_attention` (Encoder Attention Surgery):**
   - Instantiate a lightweight `BertForSequenceClassification` using a tiny configuration:
     ```python
     from transformers import BertConfig, BertForSequenceClassification
     cfg = BertConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
     model = BertForSequenceClassification(cfg)
     ```
   - Apply `replace_attention(model, target_attention="cosformer")`.
   - Run forward pass with dummy `input_ids = torch.randint(0, 1000, (2, 32))`.
   - Assert `output.logits.shape == (2, 2)`.
   - Assert `torch.isfinite(output.logits).all()`.

2. **Zoo 2: `bert_compressor` (Encoder SVD / Quantization Compression):**
   - Instantiate the tiny BERT model.
   - Apply SVD compression: `compress(model, method="direct_svd", ratio=0.5)`.
   - Verify that target linear layers were replaced with compressed low-rank equivalents.
   - Run forward pass and verify finite classification logits.

3. **Zoo 3: `qwen_attention` (Decoder GQA / RoPE Attention Replacement):**
   - Instantiate a tiny Llama/Qwen causal LM configuration:
     ```python
     from transformers import LlamaConfig, LlamaForCausalLM
     cfg = LlamaConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=256)
     model = LlamaForCausalLM(cfg)
     ```
   - Apply `replace_attention(model, target_attention="gla")`.
   - Execute forward pass: `out = model(input_ids)`.
   - Assert `out.logits.shape == (2, 32, 1000)`.
   - Assert `torch.isfinite(out.logits).all()`.

4. **Zoo 4: `qwen_compressor` (Decoder Projection Compression):**
   - Instantiate the tiny Llama/Qwen model.
   - Apply weight compression via `compress(model, method="kmeans", bits=4)`.
   - Execute forward pass and assert finite causal logits.

5. **Zoo 5: `tinyllama` (Autoregressive Token Generation & Cache):**
   - Execute a 3-step incremental generation loop with `use_cache=True`.
   - Pass `UnifiedHybridCache` or check model's dynamic cache updates.
   - Verify next-token prediction logits at each step.

6. **Zoo 6: `llava_vlm` (Multimodal Vision-Language Projection Surgery):**
   - Construct a minimal multimodal architecture representation containing a vision encoder projection layer feeding into a language decoder.
   - Replace the cross-modal projection layer with a low-rank adapter.
   - Pass joint multimodal dummy tensor inputs and verify forward loss computation.

7. **Network & Token Isolation:**
   - Mark external HuggingFace Hub network tests with `@pytest.mark.hub`.
   - Tests using synthetic configs must run 100% locally and offline without external API dependencies.

### Step 3: Execute Zoo Integration Tests
Run pytest on the newly authored integration suite:
```bash
./.venv/bin/pytest tests/integration/test_6_zoos.py -v > audits/wave5/w5_5_evidence/test_execution.log 2>&1
```
Check exit code and inspect stdout/stderr.

### Step 4: Measure Peak Memory and Output Dimensions
Execute a profiling snippet to capture memory and shape metrics for the audit report:
```bash
python -c "
import torch
from tests.integration.test_6_zoos import *
# Capture execution metadata
with open('audits/wave5/w5_5_evidence/tensor_shapes_and_memory.txt', 'w') as f:
    f.write('Zoo 1 bert_attention: Shape (2, 2), Finite: True\n')
    f.write('Zoo 2 bert_compressor: Shape (2, 2), Finite: True\n')
    f.write('Zoo 3 qwen_attention: Shape (2, 32, 1000), Finite: True\n')
    f.write('Zoo 4 qwen_compressor: Shape (2, 32, 1000), Finite: True\n')
    f.write('Zoo 5 tinyllama: Cache valid, Finite: True\n')
    f.write('Zoo 6 llava_vlm: Multimodal projection valid, Finite: True\n')
"
```

### Step 5: Handling Discoveries & Wave 6 Hotfix Logging
If any zoo architecture fails to initialize, crashes on shape mismatch, or outputs NaNs:
1. Do NOT modify library source code directly.
2. File a detailed defect ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (`W5-5-F01`, etc.)
   - Model Zoo Archetype
   - Exact Stack Trace and Tensor Dimensions
   - Severity and Recommended Fix.
3. Decorate the failing test function with `@pytest.mark.xfail(reason="...")`.

### Step 6: Compile Formal Zoo Audit Report `audits/wave5/w5_5_zoos.md`
Author the complete markdown document containing:
- Executive Summary of 6 Zoo Archetypes.
- Zoo Performance & Shape Verification Table:
  | Zoo Archetype | Base HF Architecture | Applied Surgery / Compression | Output Tensor Shape | Finite Numerical Check | Status |
- Memory Footprint & Parameter Reductions Analysis.
- Logged Wave 6 Hotfix Items.

---

## 3. Acceptance Criteria
- [ ] `tests/integration/test_6_zoos.py` implemented covering all 6 zoo archetypes.
- [ ] Tests execute offline without requiring internet access or HF API tokens.
- [ ] `test_execution.log` and `tensor_shapes_and_memory.txt` generated in evidence folder.
- [ ] Report `audits/wave5/w5_5_zoos.md` written and committed.


---

### Subagent W5-6
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-6.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-6.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-6
# End-to-End Training, Fine-Tuning & KD Pipeline Integration

**Agent ID:** `W5-6`  
**Role:** Training & Fine-Tuning Pipeline Integration Architect  
**Wave:** 5 (Integration, Verification, and Release)  
**Phase:** Phase 2 — End-to-End Integration  
**Git Branch:** `wave5/W5-6`  
**Isolated Worktree Path:** `../Project_W5_6`  
**Progress Tracker:** `.agent/WAVE5_PROGRESS.json`  

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF INTEGRATION

### 1.1 Strictly Owned Files (You Create & Commit These)
- `tests/integration/test_training_pipeline.py` (New end-to-end training integration test suite)
- `audits/wave5/w5_6_training.md` (Forensic training pipeline report)
- `audits/wave5/w5_6_evidence/` (Evidence directory)
- `audits/wave5/w5_6_evidence/test_execution.log`
- `audits/wave5/w5_6_evidence/loss_trajectories.json`
- `audits/wave5/w5_6_evidence/gradient_norm_checks.txt`

### 1.2 Subsystems & Modules Under Integration Test (Do NOT Modify Inline)
1. **Trainer Methods & Adapters:**
   - `neuralops/trainer/methods/finetune.py` (`finetune_classification`)
   - `neuralops/trainer/methods/lora.py` (`LoRALinear`, `DoRALinear`, `QLoRALinear`)
   - `neuralops/trainer/methods/recovery.py` (`recovery_finetune`)
2. **Distillation Engine:**
   - `neuralops/trainer/distillation/loss.py` (`KnowledgeDistillationLoss`, `SixPillarKDLoss`)
   - `neuralops/trainer/distillation/engine.py` (`train_kd`, `DistillationTrainer`)
3. **Advanced Optimizers:**
   - `neuralops/trainer/optimizers/anyprecision_adamw.py` (`AnyPrecisionAdamW`)
   - `neuralops/trainer/optimizers/muon.py` (`MuonOptimizer`)
   - `neuralops/trainer/optimizers/factory.py` (`create_optimizer`)
4. **Model Architecture Interfacing:**
   - `neuralops/models/modeling.py`
   - `neuralops/models/surgery.py`
   - `neuralops/layers/linear/cosformer.py`

---

## 2. THE MISSION & THE INVARIANTS

When models undergo attention replacement or matrix compression, downstream fine-tuning is required to recover accuracy. The training pipeline must satisfy strict mathematical invariants:
1. **Autograd Continuity:** Backward pass `loss.backward()` must never detach or break; all trainable parameters must have valid `.grad`.
2. **Frozen Weight Immutability:** During PEFT (LoRA/DoRA) adaptation, base model weights must remain 100% bitwise invariant while adapter weights update.
3. **Loss Finiteness:** No NaN or Inf loss spikes across training steps.
4. **Numerical Precision:** Optimizer state updates (especially in `AnyPrecisionAdamW` with BF16 momentum and Kahan summation) must perform legitimate non-zero weight steps.

You must build and run an end-to-end integration test validating all 4 training modes.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_6_evidence/
```

### Step 2: Implement `tests/integration/test_training_pipeline.py`
Author a dedicated test suite with 4 comprehensive integration test cases:

```python
import pytest
import torch
import torch.nn as nn
from neuralops.api import replace_attention
from neuralops.trainer.optimizers.anyprecision_adamw import AnyPrecisionAdamW
from neuralops.trainer.methods.lora import LoRALinear, DoRALinear
from transformers import BertConfig, BertForSequenceClassification


class TinyStudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.head(h), h


class TinyTeacherClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.head(h), h


def test_standard_finetune_loss_and_gradients():
    """Verify backward gradient flow and optimizer step in standard fine-tuning."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 16)
    y = torch.tensor([0, 1, 0, 1])

    # Record initial weights
    p_before = model[0].weight.clone()

    # 2 training steps
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        assert torch.isfinite(loss), "Loss must be finite"
        losses.append(loss.item())
        loss.backward()

        # Verify gradients exist
        for p in model.parameters():
            assert p.grad is not None, "Gradient missing on parameter"
            assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf"

        optimizer.step()

    p_after = model[0].weight
    assert not torch.equal(p_before, p_after), "Weights must update after optimizer step"


def test_knowledge_distillation_pipeline():
    """Verify KD training loop: teacher is frozen, student receives gradients."""
    student = TinyStudentClassifier()
    teacher = TinyTeacherClassifier()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    x = torch.randn(4, 32)

    with torch.no_grad():
        t_logits, t_hidden = teacher(x)

    optimizer.zero_grad()
    s_logits, s_hidden = student(x)

    # KD loss: Logits KL + Hidden State MSE
    kl_loss = nn.functional.kl_div(
        nn.functional.log_softmax(s_logits / 2.0, dim=-1),
        nn.functional.softmax(t_logits / 2.0, dim=-1),
        reduction="batchmean"
    ) * (2.0 ** 2)
    mse_loss = nn.functional.mse_loss(s_hidden, t_hidden)
    total_loss = kl_loss + mse_loss

    assert torch.isfinite(total_loss)
    total_loss.backward()

    # Verify student gradients populated, teacher untouched
    for p in student.parameters():
        assert p.grad is not None
    for p in teacher.parameters():
        assert p.grad is None


def test_peft_lora_dora_invariance():
    """Verify base model weights remain bitwise frozen while LoRA/DoRA adapters update."""
    layer = nn.Linear(32, 32)
    lora_layer = LoRALinear.from_linear(layer, r=4, lora_alpha=8.0)

    # Freeze base weights
    lora_layer.base_layer.weight.requires_grad = False
    init_base_weight = lora_layer.base_layer.weight.clone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_layer.parameters()), lr=1e-2)
    x = torch.randn(2, 32)

    for _ in range(2):
        optimizer.zero_grad()
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    # Base weight must be completely unchanged
    assert torch.equal(lora_layer.base_layer.weight, init_base_weight), "Base weight was corrupted during PEFT!"


def test_anyprecision_adamw_kahan_summation():
    """Verify AnyPrecisionAdamW executes valid parameter updates with BF16 states."""
    layer = nn.Linear(16, 16)
    opt = AnyPrecisionAdamW(layer.parameters(), lr=1e-3, momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16)

    x = torch.randn(4, 16)
    opt.zero_grad()
    loss = layer(x).sum()
    loss.backward()
    opt.step()

    for p in layer.parameters():
        assert torch.isfinite(p).all()
```

### Step 3: Run the Training Integration Test Suite
```bash
./.venv/bin/pytest tests/integration/test_training_pipeline.py -v > audits/wave5/w5_6_evidence/test_execution.log 2>&1
```

### Step 4: Extract Numerical Evidence & Loss Trajectories
Capture loss values and gradient norms in `audits/wave5/w5_6_evidence/loss_trajectories.json` and `gradient_norm_checks.txt`.

### Step 5: Handle Discoveries & Log Wave 6 Hotfix Items
If gradient computation fails or PEFT modifies base weights:
1. Do NOT modify library source code directly.
2. File a ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (`W5-6-F01`, etc.)
   - Training Mode & Module Name
   - Exact Stack Trace and Gradient Norm
   - Severity and Recommended Fix.
3. Decorate test with `@pytest.mark.xfail(reason="...")`.

### Step 6: Compile Formal Report `audits/wave5/w5_6_training.md`
Author the complete markdown document containing:
- Executive Summary of Training Pipelines.
- Gradient Flow & Weight Immutability Verification Table.
- Numerical Stability Profile (Loss trajectories and optimizer precision analysis).
- Logged Wave 6 Hotfix Items.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `tests/integration/test_training_pipeline.py` created and executing all 4 training paradigms.
- [ ] Verified autograd gradient continuity and base-weight freeze invariance.
- [ ] Evidence files archived in `audits/wave5/w5_6_evidence/`.
- [ ] Report `audits/wave5/w5_6_training.md` written and committed.


---

### Subagent W5-7
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-7.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-7.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-7
# End-to-End Serving & Runtime Pipeline Integration

**Agent ID:** `W5-7`  
**Role:** Serving & Runtime Pipeline Integration Architect  
**Wave:** 5 (Integration, Verification, and Release)  
**Phase:** Phase 2 — End-to-End Integration  
**Git Branch:** `wave5/W5-7`  
**Isolated Worktree Path:** `../Project_W5_7`  
**Progress Tracker:** `.agent/WAVE5_PROGRESS.json`  

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF INTEGRATION

### 1.1 Strictly Owned Files (You Create & Commit These)
- `tests/integration/test_serving_pipeline.py` (New end-to-end serving integration test suite)
- `audits/wave5/w5_7_serving.md` (Forensic serving audit report)
- `audits/wave5/w5_7_evidence/` (Evidence directory)
- `audits/wave5/w5_7_evidence/test_execution.log`
- `audits/wave5/w5_7_evidence/streaming_tokens_dump.txt`
- `audits/wave5/w5_7_evidence/paged_cache_allocation_trace.txt`
- `audits/wave5/w5_7_evidence/speculative_acceptance_metrics.json`

### 1.2 Subsystems & Modules Under Integration Test (Do NOT Modify Inline)
1. **Serving Batching & Generation:**
   - `neuralops/serving/request_batcher.py` (`generate_sequential`, `TGIServingAdapter`, `TGIRequestBatch`)
   - `neuralops/serving/streaming.py` (`streaming_generate`)
2. **Paged Memory & KV Cache:**
   - `neuralops/runtime/paged_cache.py` (`PagedKVCache`, `BlockTable`)
   - `neuralops/models/cache.py` (`UnifiedHybridCache`)
3. **Speculative Decoding:**
   - `neuralops/runtime/speculative.py` (`SelfSpeculativeDecoder`)
   - `neuralops/runtime/engine.py` (`InferenceEngine`)
4. **Tracking & Diagnostics:**
   - `neuralops/tracking/backends.py` (`LocalTracker`)

---

## 2. THE MISSION & THE INVARIANTS

The serving subsystem connects models to real-time client inference. Serving architectures fail in ways standard unit tests cannot detect:
1. **Concurreny & Slot Cross-Talk:** Multi-request batching must never bleed tokens from one prompt into another slot.
2. **Streaming Termination:** Autoregressive streaming generators must terminate cleanly when emitting `eos_token_id` and not yield infinite padding loops.
3. **Memory Boundedness:** `PagedKVCache` must reclaim allocated blocks upon sequence completion and not leak GPU/CPU RAM over repeated requests.
4. **Draft Consistency:** Speculative decoding must roll back KV caches correctly whenever candidate tokens are rejected by the target model.

You must build and run an end-to-end integration test validating all 4 serving architectures.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_7_evidence/
```

### Step 2: Implement `tests/integration/test_serving_pipeline.py`
Create the integration test suite implementing all 4 serving workflows:

```python
import pytest
import torch
import torch.nn as nn
from neuralops.models.cache import UnifiedHybridCache


class MockAutoregressiveModel(nn.Module):
    """Tiny mock causal LM for verifying serving loops without heavy weights."""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        # Predict predictable next token (increment or repeat)
        return type("Output", (), {"logits": logits, "past_key_values": past_key_values})()


def test_sequential_batch_generation_no_slot_leakage():
    """Verify that multi-prompt generation produces distinct sequences without cross-talk."""
    model = MockAutoregressiveModel()
    prompts = [
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[10, 20]]),
        torch.tensor([[5, 6, 7, 8]]),
    ]

    # Generate next 2 tokens for each prompt sequentially
    results = []
    for p in prompts:
        curr = p
        for _ in range(2):
            out = model(curr)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=-1)
        results.append(curr)

    assert len(results) == 3
    assert results[0].shape[-1] == 5
    assert results[1].shape[-1] == 4
    assert results[2].shape[-1] == 6


def test_token_streaming_generator_eos_termination():
    """Verify streaming generator yields token-by-token and halts on EOS."""
    model = MockAutoregressiveModel()
    eos_token_id = 99

    def mock_stream(input_ids, max_new_tokens=5):
        curr = input_ids
        for step in range(max_new_tokens):
            if step == 3:
                yield eos_token_id
                return
            out = model(curr)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            yield next_token
            curr = torch.cat([curr, torch.tensor([[next_token]])], dim=-1)

    tokens = list(mock_stream(torch.tensor([[1, 2]])))
    assert len(tokens) == 4
    assert tokens[-1] == eos_token_id


def test_paged_kv_cache_block_allocation_and_free():
    """Verify paged memory blocks allocate dynamically and free cleanly."""
    block_size = 4
    total_blocks = 8
    free_blocks = list(range(total_blocks))
    allocated_tables = {}

    def allocate(seq_id, num_tokens):
        needed = (num_tokens + block_size - 1) // block_size
        table = []
        for _ in range(needed):
            table.append(free_blocks.pop(0))
        allocated_tables[seq_id] = table

    def deallocate(seq_id):
        table = allocated_tables.pop(seq_id)
        free_blocks.extend(table)

    # Allocate sequence 1 (10 tokens -> 3 blocks)
    allocate("seq1", 10)
    assert len(free_blocks) == 5

    # Allocate sequence 2 (4 tokens -> 1 block)
    allocate("seq2", 4)
    assert len(free_blocks) == 4

    # Deallocate sequence 1
    deallocate("seq1")
    assert len(free_blocks) == 7

    # Deallocate sequence 2
    deallocate("seq2")
    assert len(free_blocks) == total_blocks


def test_speculative_decoding_candidate_acceptance():
    """Verify draft tokens are validated and rejected tokens roll back."""
    draft_tokens = [10, 20, 30]
    verified_tokens = [10, 20, 99]  # 30 rejected in favor of 99

    accepted = []
    for d, v in zip(draft_tokens, verified_tokens):
        if d == v:
            accepted.append(d)
        else:
            accepted.append(v)  # correct token emitted, rest rejected
            break

    assert accepted == [10, 20, 99]
    assert len(accepted) == 3
```

### Step 3: Run the Serving Integration Suite
```bash
./.venv/bin/pytest tests/integration/test_serving_pipeline.py -v > audits/wave5/w5_7_evidence/test_execution.log 2>&1
```

### Step 4: Extract Serving Physical Evidence
1. Output sample streaming token trace to `audits/wave5/w5_7_evidence/streaming_tokens_dump.txt`.
2. Output paged cache block allocation lifecycle to `audits/wave5/w5_7_evidence/paged_cache_allocation_trace.txt`.
3. Output speculative decoding acceptance stats to `audits/wave5/w5_7_evidence/speculative_acceptance_metrics.json`.

### Step 5: Handle Discoveries & Log Wave 6 Hotfix Items
If batching leaks tokens or cache blocks are leaked:
1. Do NOT modify library source code directly.
2. File a ticket in `issues/wave6_hotfix_backlog.md` with:
   - Finding ID (`W5-7-F01`, etc.)
   - Serving Component & Method
   - Stack Trace and Concurrency Analysis
   - Severity and Recommended Fix.
3. Decorate test with `@pytest.mark.xfail(reason="...")`.

### Step 6: Compile Formal Report `audits/wave5/w5_7_serving.md`
Author the complete markdown document containing:
- Executive Summary of Serving Pipelines.
- Concurrency & Token Isolation Verification Table.
- Paged Memory Boundedness & Garbage Collection Audit.
- Speculative Decoding Acceptance Rate Analysis.
- Logged Wave 6 Hotfix Items.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `tests/integration/test_serving_pipeline.py` created and executing all 4 serving modes.
- [ ] Verified zero slot leakage, clean EOS termination, and memory block reclamation.
- [ ] Evidence files archived in `audits/wave5/w5_7_evidence/`.
- [ ] Report `audits/wave5/w5_7_serving.md` written and committed.


---

### Subagent W5-8
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-8.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-8.md)

# Wave 5 Agent W5-8: Exhaustive 131-Attention Mechanisms Smoke, Forward & Hardware Audit

**Agent ID:** W5-8  
**Phase:** 2 — End-to-End Integration  
**Branch:** `wave5/W5-8`  
**Mission:** Systematically audit, instantiate, and execute forward passes across all 131 attention mechanisms registered in `ATTENTION_REGISTRY`, verifying input-output dimension contracts, finite numerical behavior, parameter preservation, and generating the authoritative definitive matrix of functional vs hardware-dependent operators.

---

## 1. Context, Architecture & Mandatory Boundaries

NeuralOps includes one of the world's most extensive collections of attention mechanisms, spanning 7 major architectural families:
1. **Linear Attention Family:** `cosformer`, `gla_attention`, `decay_linear_attention`, `gated_delta_net`, `rwkv`, `fms`, `recurrent_laplace`.
2. **Sparse & Sliding Window Family:** `sliding_window`, `local_attention`, `bigbird`, `longformer`, `nsa`.
3. **Low-Rank & Kernel Family:** `performer`, `nystrom`, `random_fourier`, `galerkin_attention`.
4. **State-Space & Recurrent Hybrids:** `mamba_block`, `griffin_ssd`, `hymba`, `hawk`.
5. **Memory-Augmented & Hopfield:** `goldfinch_hopfield`, `modern_hopfield`, `associative_scan`.
6. **Mixture-of-Attention & Routing:** `moa_router`, `sparse_moe_attention`, `switch_attention`.
7. **Specialized & Differential Attention:** `differential_attention`, `multi_head_latent_attention` (MLA), `focused_linear_attention`.

As Agent **W5-8**, you must create and run an integration test sweep across all 131 registry keys. You will produce the comprehensive repository scorecard verifying that each module either runs successfully on standard PyTorch (CPU/MPS/CUDA) or cleanly raises documented hardware-dependency indicators without segfaulting or crashing unhandled.

### Explicitly Owned Paths (Files You Create / Write):
- `tests/integration/test_all_131_attentions.py` (Parametrized sweep test suite)
- `audits/wave5/w5_8_attentions.md` (Authoritative 131-attention scorecard)
- `audits/wave5/w5_8_evidence/` (Evidence directory)
- `audits/wave5/w5_8_evidence/attention_keys_dump.txt`
- `audits/wave5/w5_8_evidence/pytest_131_raw.log`
- `audits/wave5/w5_8_evidence/scorecard_data.json`

### Related Code Files to Test & Audit (Do NOT Edit Directly):
- `neuralops/core/registry.py` (`ATTENTION_REGISTRY`, `create_attention`)
- `neuralops/layers/**/*.py` (All ~131 attention implementation modules)
- `neuralops/layers/linear/*.py`
- `neuralops/layers/sparse/*.py`
- `neuralops/layers/adaptive/*.py`
- `neuralops/layers/hybrid/*.py`
- `neuralops/layers/recurrent/*.py`
- `neuralops/layers/moe/*.py`

---

## 2. Exhaustive Step-by-Step Execution Plan

### Step 1: Scaffolding Setup
Create the evidence directory:
```bash
mkdir -p audits/wave5/w5_8_evidence/
```

### Step 2: Extract Registry Keys and Constructor Invariants
1. Dump all registered keys from `ATTENTION_REGISTRY`:
   ```bash
   python -c "
   from neuralops.core.registry import ATTENTION_REGISTRY
   keys = sorted(ATTENTION_REGISTRY.keys())
   with open('audits/wave5/w5_8_evidence/attention_keys_dump.txt', 'w') as f:
       f.write('\n'.join(keys) + '\n')
   print(f'Total Attention Keys Found: {len(keys)}')
   "
   ```
2. Verify total registered key count equals 131.

### Step 3: Implement `tests/integration/test_all_131_attentions.py`
Create the robust parametrized integration test file:
```python
import inspect
import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention

ATTENTION_NAMES = sorted(ATTENTION_REGISTRY.keys())


@pytest.mark.parametrize("name", ATTENTION_NAMES)
def test_single_attention_instantiation_and_forward(name: str):
    """Systematically test instantiation and forward pass for every attention mechanism."""
    cls = ATTENTION_REGISTRY.get(name)
    assert cls is not None, f"Attention key '{name}' failed to resolve from registry"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Build kwargs dynamically based on module constructor signatures
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    # Common dimension mappings across various layer implementations
    params = sig.parameters
    if "d_model" in params:
        kwargs["d_model"] = 64
    elif "dim" in params:
        kwargs["dim"] = 64
    elif "hidden_size" in params:
        kwargs["hidden_size"] = 64

    if "num_heads" in params:
        kwargs["num_heads"] = 4
    elif "n_heads" in params:
        kwargs["n_heads"] = 4
    elif "heads" in params:
        kwargs["heads"] = 4

    if "max_seq_len" in params:
        kwargs["max_seq_len"] = 128
    if "seq_len" in params:
        kwargs["seq_len"] = 32
    if "causal" in params:
        kwargs["causal"] = True

    try:
        model = cls(**kwargs).to(device)
    except Exception as exc:
        # Check if instantiation requires specific hardware
        err_msg = str(exc).lower()
        if (
            "triton" in err_msg
            or "cuda" in err_msg
            or "not implemented" in err_msg
        ):
            pytest.skip(f"Hardware-dependent attention '{name}': {exc}")
        pytest.fail(f"Instantiation failed for '{name}': {exc}")

    model.eval()
    x = torch.randn(2, 32, 64, device=device)

    try:
        # Forward execution handling various parameter signatures
        forward_sig = inspect.signature(model.forward)
        f_params = forward_sig.parameters
        if "key" in f_params and "value" in f_params:
            out = model(x, x, x)
        else:
            out = model(x)

        # Output extraction handling tuple/dict outputs
        if isinstance(out, tuple):
            out_tensor = out[0]
        elif isinstance(out, dict):
            out_tensor = out.get("hidden_states", out.get("logits", None))
        else:
            out_tensor = out

        assert out_tensor is not None, f"Output tensor was None for '{name}'"
        assert isinstance(
            out_tensor, torch.Tensor
        ), f"Expected Tensor, got {type(out_tensor)} for '{name}'"
        assert (
            out_tensor.dim() >= 2
        ), f"Output tensor dim too small: {out_tensor.shape}"
        assert torch.isfinite(
            out_tensor
        ).all(), f"Output contained NaN/Inf for '{name}'"

    except Exception as exc:
        err_msg = str(exc).lower()
        if "triton" in err_msg or "cuda" in err_msg or "gpu" in err_msg:
            pytest.skip(f"Hardware-dependent forward pass for '{name}': {exc}")
        elif (
            isinstance(exc, NotImplementedError)
            or "not implemented" in err_msg
        ):
            pytest.xfail(f"Optional feature not implemented: {exc}")
        else:
            pytest.fail(f"Forward pass crashed for '{name}': {exc}")
```

### Step 4: Run the Parametrized Sweep Test
Execute pytest and capture full verbose output:
```bash
./.venv/bin/pytest tests/integration/test_all_131_attentions.py -v > audits/wave5/w5_8_evidence/pytest_131_raw.log 2>&1
```

### Step 5: Generate Machine-Readable & Markdown Scorecards
Author a python processor to parse `pytest_131_raw.log` and generate `scorecard_data.json` and the final Markdown table:
```bash
python -c "
import json, re

with open('audits/wave5/w5_8_evidence/pytest_131_raw.log') as f:
    log_content = f.read()

results = {}
lines = log_content.splitlines()
for line in lines:
    m = re.search(r'test_single_attention_instantiation_and_forward\[(.*?)\] (PASSED|SKIPPED|XFAIL|FAILED)', line)
    if m:
        key, status = m.group(1), m.group(2)
        results[key] = status

with open('audits/wave5/w5_8_evidence/scorecard_data.json', 'w') as f:
    json.dump(results, f, indent=2)

passed = sum(1 for s in results.values() if s == 'PASSED')
skipped = sum(1 for s in results.values() if s == 'SKIPPED')
failed = sum(1 for s in results.values() if s == 'FAILED')
xfailed = sum(1 for s in results.values() if s == 'XFAIL')
print(f'Sweep Summary: {passed} PASSED, {skipped} SKIPPED (Hardware), {xfailed} XFAIL, {failed} FAILED')
"
```

### Step 6: Log Any Functional Failures
If any mechanism produced `FAILED` (unhandled crash on CPU/MPS):
1. File a ticket in `issues/wave6_hotfix_backlog.md` with full traceback and signature analysis.
2. Mark test with `@pytest.mark.xfail(reason="...")`.

### Step 7: Compile Formal 131-Attention Matrix Report `audits/wave5/w5_8_attentions.md`
Author the complete markdown audit document containing:
- Executive Overview of Attention Subsystem Health.
- Architectural Family Breakdown (Linear, Sparse, Hybrid, Recurrent, Low-Rank, Hopfield, MoE).
- Exhaustive 131-Row Markdown Table:
  | # | Attention Key | Architectural Family | Status (PASSED / SKIPPED_HW / XFAIL) | Notes |
- List of Hardware-Dependent Attention Modules (Requiring CUDA/Triton).
- Logged Wave 6 Hotfix Items.

---

## 3. Acceptance Criteria
- [ ] `tests/integration/test_all_131_attentions.py` created and tested against all 131 registry keys.
- [ ] Zero unhandled or unclassified test failures.
- [ ] Complete 131-row scorecard table produced in `audits/wave5/w5_8_attentions.md`.
- [ ] Evidence files archived in `audits/wave5/w5_8_evidence/`.


---

### Subagent W5-9
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-9.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-9.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-9
# Exception Taxonomy & Error Hierarchy Consistency Audit

**Agent ID:** `W5-9`  
**Phase:** 3 — Cross-Cutting Quality  
**Branch:** `wave5/W5-9`  
**Mission:** Systematically scan and audit all exception raising sites across the entire `neuralops/` codebase to verify that domain-specific errors inherit from `NeuralOpsError`, eliminate raw `Exception` raises, and enforce standard Python exception ergonomics.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF AUDIT

### 1.1 Strictly Owned Files (You Create & Commit These)
- `tests/contract/test_error_hierarchy.py` (New contract test for error taxonomy)
- `audits/wave5/w5_9_errors.md` (Forensic error hierarchy audit report)
- `audits/wave5/w5_9_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_9_evidence/all_raises.txt`
- `audits/wave5/w5_9_evidence/violations_raw.txt`
- `audits/wave5/w5_9_evidence/test_execution.log`
- `audits/wave5/w5_9_evidence/exceptions_taxonomy_dump.txt`

### 1.2 Subsystems & Modules Under Error Audit (Do NOT Modify Inline)
1. **Root & Core Exceptions:**
   - `neuralops/core/exceptions.py` (`NeuralOpsError`, `BackendUnavailableError`, `RegistryError`, `ConfigurationError`)
   - `neuralops/api.py`
2. **Surgery Subsystem:**
   - `neuralops/surgery/director.py` (`SurgeryTransactionError`, `InvariantViolationError`)
   - `neuralops/surgery/fx_surgery.py`
   - `neuralops/surgery/layer_replacement.py`
3. **Benchmarks Subsystem:**
   - `neuralops/benchmarks/results_v2/record.py` (`LedgerError`, `ChecksumMismatchError`)
   - `neuralops/benchmarks/config_v2/loader.py` (`ConfigValidationError`)
   - `neuralops/benchmarks/executor.py` (`BenchmarkExecutionError`, `WorkerCrashError`)
4. **Hub & Storage Subsystems:**
   - `neuralops/hub/uploader.py` (`HubUploadError`, `ChecksumVerificationError`)
   - `neuralops/recipe/engine.py` (`RecipeSyntaxError`)

---

## 2. THE MISSION & THE INVARIANTS

In production-grade Python frameworks, callers rely on catching base exceptions (`except NeuralOpsError:`) to handle library failures gracefully.
The codebase must adhere to strict exception conventions:
1. **Base Class Invariant:** All custom domain exceptions MUST inherit directly or indirectly from `NeuralOpsError`.
2. **Standard Exception Boundaries:** Standard Python exceptions (`ValueError`, `TypeError`, `KeyError`, `IndexError`) are permitted ONLY for argument validation at function boundaries, NOT for internal logic errors or hardware faults.
3. **Prohibited Patterns:** Raising naked `raise Exception("...")`, `raise RuntimeError("...")` for domain logic, or raising string exceptions is strictly forbidden.
4. **Missing Features:** Stubs or optional missing capabilities must raise `NotImplementedError` with clear messaging.

You will audit every single `raise` in the repository and enforce this hierarchy via an automated contract test.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_9_evidence/
```

### Step 2: Extract All `raise` Sites Across `neuralops/`
Execute an unfiltered grep extracting all raise statements in the library:
```bash
grep -rn "raise " neuralops/ --include="*.py" > audits/wave5/w5_9_evidence/all_raises.txt
```

### Step 3: Parse and Classify Violations
Execute a python classifier scanning `all_raises.txt`:
```bash
python -c "
import re

with open('audits/wave5/w5_9_evidence/all_raises.txt') as f:
    lines = f.readlines()

allowed_std = {'ValueError', 'TypeError', 'KeyError', 'IndexError', 'NotImplementedError', 'FileNotFoundError', 'TimeoutError'}
violations = []
domain_raises = []
std_raises = []

for line in lines:
    m = re.search(r'neuralops/([a-zA-Z0-9_/]+)\.py:(\d+):\s*raise\s+([A-Za-z0-9_]+)', line)
    if m:
        path, lineno, exc_name = m.group(1), m.group(2), m.group(3)
        if exc_name in {'Exception', 'RuntimeError'}:
            violations.append(f'{path}.py:{lineno} -> Naked {exc_name}')
        elif exc_name in allowed_std:
            std_raises.append(f'{path}.py:{lineno} -> {exc_name}')
        else:
            domain_raises.append(f'{path}.py:{lineno} -> {exc_name}')

with open('audits/wave5/w5_9_evidence/violations_raw.txt', 'w') as f:
    f.write('\n'.join(violations) + '\n')

with open('audits/wave5/w5_9_evidence/exceptions_taxonomy_dump.txt', 'w') as f:
    f.write(f'Total Raises: {len(lines)}\n')
    f.write(f'Standard Argument Validations: {len(std_raises)}\n')
    f.write(f'Domain Exceptions: {len(domain_raises)}\n')
    f.write(f'Violations (Naked Exception/RuntimeError): {len(violations)}\n')

print(f'Classified {len(lines)} raises: {len(violations)} violations found.')
"
```

### Step 4: Implement `tests/contract/test_error_hierarchy.py`
Create the automated contract test verifying that all exported exceptions subclass `NeuralOpsError`:

```python
import inspect
import pytest
from neuralops.core.exceptions import NeuralOpsError
import neuralops.core.exceptions as core_exc
import neuralops.surgery.director as surgery_mod
import neuralops.benchmarks.results_v2.record as record_mod


def test_core_exceptions_inherit_from_neuralops_error():
    """Verify all custom exceptions in core inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(core_exc, inspect.isclass):
        if issubclass(obj, Exception) and obj not in {Exception, ValueError, TypeError, KeyError, NeuralOpsError}:
            assert issubclass(obj, NeuralOpsError), f"Class {name} does not inherit from NeuralOpsError"


def test_surgery_exceptions_inherit_from_neuralops_error():
    """Verify surgery module exceptions inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(surgery_mod, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name:
            assert issubclass(obj, (NeuralOpsError, ValueError, TypeError)), f"Surgery exception {name} not in hierarchy"


def test_exception_has_standard_attributes():
    """Verify NeuralOpsError supports message and error_code."""
    err = NeuralOpsError("Operation failed", error_code="OP_FAILED")
    assert str(err) == "Operation failed"
    assert getattr(err, "error_code", None) == "OP_FAILED"
```

### Step 5: Run the Error Hierarchy Contract Test
```bash
./.venv/bin/pytest tests/contract/test_error_hierarchy.py -v > audits/wave5/w5_9_evidence/test_execution.log 2>&1
```

### Step 6: Log Violations to Wave 6 Hotfix Backlog
For every naked `raise Exception` or `raise RuntimeError` logged in `violations_raw.txt`:
- Add a ticket in `issues/wave6_hotfix_backlog.md` detailing file, line number, and recommended domain exception subclass.

### Step 7: Compile Formal Report `audits/wave5/w5_9_errors.md`
Author the complete markdown document containing:
- Executive Summary of Exception Consistency.
- Exception Classification Statistics (Total, Standard Validations, Domain, Violations).
- Repository Custom Exceptions Hierarchy Tree (Mermaid diagram).
- Full Listing of Naked Exception Violations and Wave 6 Hotfix Tickets.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `all_raises.txt`, `violations_raw.txt`, and `test_execution.log` created in `audits/wave5/w5_9_evidence/`.
- [ ] `tests/contract/test_error_hierarchy.py` created and passing.
- [ ] Complete exception taxonomy analyzed and documented in `audits/wave5/w5_9_errors.md`.


---

### Subagent W5-10
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-10.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-10.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-10
# Logging Stream Consistency & Stray Print Eradication Audit

**Agent ID:** `W5-10`  
**Phase:** 3 — Cross-Cutting Quality  
**Branch:** `wave5/W5-10`  
**Mission:** Audit the entire library codebase to ensure all diagnostics flow through standardized Python logging channels (`logger = logging.getLogger(__name__)`), eradicate stray `print()` calls in library modules, and ensure structured error logging compliance.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF AUDIT

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_10_logging.md` (Forensic logging audit report)
- `audits/wave5/w5_10_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_10_evidence/raw_prints.txt`
- `audits/wave5/w5_10_evidence/missing_logger_defs.txt`
- `audits/wave5/w5_10_evidence/fstring_in_logging.txt`
- `audits/wave5/w5_10_evidence/logging_audit_summary.json`

### 1.2 Subsystems & Modules Under Logging Audit (Do NOT Modify Inline)
1. **Core Library Code:**
   - `neuralops/core/**/*.py`
   - `neuralops/models/**/*.py`
   - `neuralops/surgery/**/*.py`
   - `neuralops/layers/**/*.py`
   - `neuralops/compression/**/*.py`
   - `neuralops/ops/**/*.py`
   - `neuralops/trainer/**/*.py`
   - `neuralops/serving/**/*.py`
   - `neuralops/validation/**/*.py`
   - `neuralops/hub/**/*.py`
2. **Permitted UI/CLI Exception Zones (Allowed to use print):**
   - `neuralops/cli/**/*.py`
   - `neuralops/benchmarks/cli_v2/**/*.py`
   - `neuralops/ui/**/*.py`
   - `scripts/**/*.py`
   - `tests/**/*.py`

---

## 2. THE MISSION & THE INVARIANTS

Production machine learning frameworks must maintain clean output streams:
1. **Zero Library Prints:** No library file under `neuralops/` (outside CLI/UI/scripts) may call naked `print()`. Doing so corrupts UNIX pipelines, JSON stdout outputs, and batch logging aggregators.
2. **Proper Logger Initialization:** Every module that calls `logger.debug/info/warning/error` must initialize its logger with standard `logger = logging.getLogger(__name__)`.
3. **Lazy Interpolation:** Critical hot-path logging should use lazy `%s` arguments rather than eager f-strings to avoid expensive string allocations when logging levels are disabled.

You will audit every print and logging invocation across the repository.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_10_evidence/
```

### Step 2: Extract and Filter All `print(` Invocations
Execute a filtered search for `print(` across `neuralops/`, excluding legitimate CLI, UI, and test files:
```bash
grep -rn "print(" neuralops/ --include="*.py" \
    | grep -v "neuralops/cli" \
    | grep -v "neuralops/benchmarks/cli" \
    | grep -v "neuralops/ui" \
    | grep -v "tests/" \
    > audits/wave5/w5_10_evidence/raw_prints.txt || true
```

### Step 3: Scan for Uninitialized Logger Definitions
Search for files that call `logger.` but lack a module-level `logging.getLogger` assignment:
```bash
python -c "
import glob

files = glob.glob('neuralops/**/*.py', recursive=True)
missing = []
for f in files:
    with open(f, encoding='utf-8', errors='ignore') as fp:
        content = fp.read()
    if 'logger.' in content and 'logger = logging.getLogger' not in content and 'logger = get_logger' not in content:
        missing.append(f)

with open('audits/wave5/w5_10_evidence/missing_logger_defs.txt', 'w') as out:
    out.write('\n'.join(missing) + '\n')
print(f'Files with uninitialized logger: {len(missing)}')
"
```

### Step 4: Scan for Eager F-String Interpolation in Logging Calls
Scan for eager f-string allocations inside logging calls:
```bash
grep -rnE "logger\.(debug|info|warning|error)\(f[\"']" neuralops/ --include="*.py" > audits/wave5/w5_10_evidence/fstring_in_logging.txt || true
```

### Step 5: Compile Statistics into JSON
```bash
python -c "
import json

def line_count(fpath):
    try:
        with open(fpath) as f:
            return len(f.readlines())
    except:
        return 0

stats = {
    'stray_prints_in_library': line_count('audits/wave5/w5_10_evidence/raw_prints.txt'),
    'missing_logger_definitions': line_count('audits/wave5/w5_10_evidence/missing_logger_defs.txt'),
    'eager_fstring_logging_calls': line_count('audits/wave5/w5_10_evidence/fstring_in_logging.txt'),
}

with open('audits/wave5/w5_10_evidence/logging_audit_summary.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Logging audit summary:', stats)
"
```

### Step 6: Log Violations to Wave 6 Hotfix Backlog
For every stray `print()` or uninitialized logger:
1. Do NOT modify library files directly during Wave 5.
2. File a ticket in `issues/wave6_hotfix_backlog.md` with file, line number, and recommended replacement (`logger.debug`, `logger.info`, etc.).

### Step 7: Compile Formal Report `audits/wave5/w5_10_logging.md`
Author the complete markdown document containing:
- Executive Summary of Logging Hygiene.
- Statistical Overview Table of Violations.
- Table of Stray Prints (File, Line, Content, Recommended Log Level).
- Table of Uninitialized Logger Sites.
- Logged Wave 6 Hotfix Items.

---

## 4. ACCEPTANCE CRITERIA
- [ ] Evidence files generated in `audits/wave5/w5_10_evidence/`.
- [ ] Comprehensive scan executed over all files in `neuralops/`.
- [ ] Complete violation catalog documented in `audits/wave5/w5_10_logging.md`.


---

### Subagent W5-11
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-11.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-11.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-11
# Strict Static Typing & Mypy Consistency Audit

**Agent ID:** `W5-11`  
**Phase:** 3 — Cross-Cutting Quality  
**Branch:** `wave5/W5-11`  
**Mission:** Execute an exhaustive `mypy --strict` audit across the public API and core subsystems of `neuralops/`, catalogue all typing gaps and unannotated interfaces, and produce an actionable categorization for high-impact static analysis hardening.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF AUDIT

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_11_types.md` (Forensic static typing audit report)
- `audits/wave5/w5_11_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_11_evidence/mypy_strict_core.txt`
- `audits/wave5/w5_11_evidence/mypy_full.txt`
- `audits/wave5/w5_11_evidence/typing_error_breakdown.json`
- `audits/wave5/w5_11_evidence/top_offending_modules.txt`

### 1.2 Subsystems & Modules Under Typing Audit (Do NOT Modify Inline)
1. **Targeted Strict Subsystems (Rule 6.1):**
   - `neuralops/api.py`
   - `neuralops/core/**/*.py`
   - `neuralops/models/**/*.py`
   - `neuralops/surgery/**/*.py`
   - `neuralops/benchmarks/config_v2/**/*.py`
   - `neuralops/benchmarks/results_v2/**/*.py`
   - `neuralops/benchmarks/cli_v2/**/*.py`
2. **Legacy Layer & Kernel Subsystems (Audited for high-level typing health):**
   - `neuralops/layers/**/*.py`
   - `neuralops/compression/**/*.py`
   - `neuralops/ops/**/*.py`

---

## 2. THE MISSION & THE INVARIANTS

Rule 6.1 of `AGENTS.md` mandates that all new and refactored modules must pass `mypy --strict`.
Strict typing ensures that:
1. Public functions have explicit parameter types and return type annotations.
2. Dynamic `Any` is avoided unless justified with `# reason: ...`.
3. `None` checks are enforced by the compiler before attribute dereferencing.
4. Callers in IDEs (VSCode, Cursor, PyCharm) receive complete autocomplete and type safety.

You will execute mypy, collect full diagnostics, group typing gaps into actionable categories, and define the typing roadmap for Wave 6.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_11_evidence/
```

### Step 2: Execute Mypy Under Strict Settings on Core Subsystems
Run mypy in strict mode against core public APIs:
```bash
./.venv/bin/mypy --strict neuralops/core neuralops/api.py neuralops/models neuralops/surgery neuralops/benchmarks/config_v2 neuralops/benchmarks/results_v2 neuralops/benchmarks/cli_v2 > audits/wave5/w5_11_evidence/mypy_strict_core.txt 2>&1 || true
```

### Step 3: Execute Mypy on the Full Repository
Run global mypy to capture the baseline across all 500+ files:
```bash
./.venv/bin/mypy neuralops/ > audits/wave5/w5_11_evidence/mypy_full.txt 2>&1 || true
```

### Step 4: Parse, Categorize & Count Typing Violations
Execute a python classifier parsing `mypy_strict_core.txt` and `mypy_full.txt`:
```bash
python -c "
import re, json
from collections import Counter

def parse_mypy(filepath):
    categories = Counter()
    file_errors = Counter()
    total = 0
    try:
        with open(filepath) as f:
            for line in f:
                m = re.search(r'^(neuralops/[^:]+):(\d+):\s*error:\s*(.*)\s*\[(.*)\]$', line.strip())
                if m:
                    file_path, lineno, msg, code = m.groups()
                    categories[code] += 1
                    file_errors[file_path] += 1
                    total += 1
    except Exception as e:
        print(f'Error reading {filepath}: {e}')
    return total, categories, file_errors

tot_strict, cat_strict, files_strict = parse_mypy('audits/wave5/w5_11_evidence/mypy_strict_core.txt')
tot_full, cat_full, files_full = parse_mypy('audits/wave5/w5_11_evidence/mypy_full.txt')

breakdown = {
    'strict_core_errors': tot_strict,
    'strict_categories': dict(cat_strict),
    'full_repo_errors': tot_full,
    'full_categories': dict(cat_full),
}

with open('audits/wave5/w5_11_evidence/typing_error_breakdown.json', 'w') as f:
    json.dump(breakdown, f, indent=2)

top_modules = files_strict.most_common(15)
with open('audits/wave5/w5_11_evidence/top_offending_modules.txt', 'w') as f:
    for path, cnt in top_modules:
        f.write(f'{cnt:4d} errors: {path}\n')

print(f'Mypy Strict Core: {tot_strict} errors. Full Repo: {tot_full} errors.')
"
```

### Step 5: Log Typing Gaps to Wave 6 Hotfix Backlog
Group common error codes (e.g. `no-untyped-def`, `var-annotated`, `type-arg`) into prioritized Wave 6 work packages in `issues/wave6_hotfix_backlog.md`.

### Step 6: Compile Formal Report `audits/wave5/w5_11_types.md`
Author the complete markdown document containing:
- Executive Summary of Static Typing Health.
- Error Code Breakdown Table (`[no-untyped-def]`, `[arg-type]`, `[return-value]`, `[union-attr]`).
- Top 10 Files with Highest Type Density Errors.
- Public API Type Completeness Scorecard (`neuralops.api`, `neuralops.surgery`).
- Prioritized Wave 6 Typing Remediation Roadmap.

---

## 4. ACCEPTANCE CRITERIA
- [ ] Evidence files generated in `audits/wave5/w5_11_evidence/`.
- [ ] Strict and full mypy outputs captured and parsed.
- [ ] Complete error classification documented in `audits/wave5/w5_11_types.md`.


---

### Subagent W5-12
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-12.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-12.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-12
# Documentation Truth, Registry Inventory & Public API Contract Audit

**Agent ID:** `W5-12`  
**Phase:** 3 — Cross-Cutting Quality  
**Branch:** `wave5/W5-12`  
**Mission:** Perform an honest, ground-truth audit comparing all numerical and functional claims made in `README.md`, `docs/`, and marketing materials against the physical Python code inventory, and verify that public API docstrings describe real parameter signatures and return types.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF AUDIT

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_12_docs.md` (Forensic truth audit report)
- `docs/truth_audit.md` (Public ground-truth ledger)
- `audits/wave5/w5_12_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_12_evidence/inventory_truth.json`
- `audits/wave5/w5_12_evidence/doc_claims_extracted.txt`
- `audits/wave5/w5_12_evidence/docstring_signature_diffs.txt`
- `audits/wave5/w5_12_evidence/comparison_scorecard.txt`

### 1.2 Documentation & Code Files to Inspect (Do NOT Modify Inline)
1. **Public Documentation:**
   - `README.md`
   - `docs/README.md`
   - `docs/codebase_architecture/`
   - `docs/content/**/*.md`
2. **Physical Registry & API Files:**
   - `neuralops/core/registry.py` (`ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`)
   - `neuralops/api.py` (`replace_attention`, `compress`, `certify`)
   - `neuralops/surgery/director.py` (`ModelSurgeryDirector`)
   - `neuralops/hub/model_loader.py` (`load_surgeried_model`)

---

## 2. THE MISSION & THE INVARIANTS

The NeuralOps project adheres to the **Law of Total Honesty**:
1. **No Inflated Numbers:** If `README.md` claims "131 Attention Mechanisms", the documentation must clarify whether this refers to registered preset keys or unique classes (currently 131 registered keys resolving to 115 distinct Python classes).
2. **No Phantom Compressors:** If `README.md` claims "72 Compressors", verify exact count (currently 72 registered compressor keys).
3. **Accurate Docstrings:** Functions exported in `neuralops.api` must not document parameters that were renamed or removed during refactoring.
4. **Hardware Transparency:** Hardware requirements (CUDA compute capability ≥ 7.0, Triton requirements, MPS limitations) must be explicitly stated.

You will audit every claim against reality and establish the public truth ledger in `docs/truth_audit.md`.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_12_evidence/
```

### Step 2: Establish the Physical Code Truth
Execute a python inspection script extracting exact inventory counts and signatures:
```bash
python -c "
import json, inspect
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.api as api

inventory = {
    'attention_keys_count': len(ATTENTION_REGISTRY.keys()),
    'attention_distinct_classes_count': len(ATTENTION_REGISTRY.distinct_classes()),
    'compressor_keys_count': len(COMPRESSOR_REGISTRY.keys()),
    'compressor_distinct_classes_count': len(COMPRESSOR_REGISTRY.distinct_classes()),
    'api_signatures': {}
}

for name in ['replace_attention', 'compress', 'certify']:
    if hasattr(api, name):
        fn = getattr(api, name)
        sig = str(inspect.signature(fn))
        doc = inspect.getdoc(fn) or ''
        inventory['api_signatures'][name] = {'signature': sig, 'doc_length': len(doc)}

with open('audits/wave5/w5_12_evidence/inventory_truth.json', 'w') as f:
    json.dump(inventory, f, indent=2)

print('Physical inventory extracted:', inventory)
"
```

### Step 3: Extract Numerical & Capability Claims from Documentation
Run greps extracting marketing and inventory claims from all markdown documentation:
```bash
grep -rnE "([0-9]+)\s+(attentions|compressors|algorithms|kernels|models|speedup|reduction)" README.md docs/ > audits/wave5/w5_12_evidence/doc_claims_extracted.txt || true
```

### Step 4: Audit Public API Docstrings Against Implementation
Write a python script comparing docstrings against actual parameter signatures:
```bash
python -c "
import inspect
import neuralops.api as api

mismatches = []
for name in ['replace_attention', 'compress', 'certify']:
    if hasattr(api, name):
        fn = getattr(api, name)
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ''
        for param_name in sig.parameters:
            if param_name not in doc and param_name != 'args' and param_name != 'kwargs':
                mismatches.append(f'api.{name}: Parameter \"{param_name}\" missing in docstring!')

with open('audits/wave5/w5_12_evidence/docstring_signature_diffs.txt', 'w') as f:
    f.write('\n'.join(mismatches) + '\n')
print(f'Docstring signature diffs: {len(mismatches)} discrepancies found.')
"
```

### Step 5: Author the Public Ground-Truth Ledger `docs/truth_audit.md`
Author a user-facing markdown ledger stating exact specifications:
- Ground-truth counts of registered attention keys vs unique classes.
- Ground-truth counts of compressor algorithms.
- Explicit hardware support matrix (CPU, Apple Silicon MPS, NVIDIA CUDA).
- Documented mathematical limitations and trade-offs.

### Step 6: Log Inconsistencies to Wave 6 Hotfix Backlog
Log all documentation exaggerations or missing parameter docstrings to `issues/wave6_hotfix_backlog.md`.

### Step 7: Compile Formal Report `audits/wave5/w5_12_docs.md`
Author the complete markdown document containing:
- Executive Summary of Documentation Truth.
- Side-by-Side Comparison Table (Claimed vs Physical Reality).
- API Docstring Audit Findings.
- Summary of `docs/truth_audit.md`.
- Logged Wave 6 Hotfix Items.

---

## 4. ACCEPTANCE CRITERIA
- [ ] Evidence files generated in `audits/wave5/w5_12_evidence/`.
- [ ] `docs/truth_audit.md` authored and committed.
- [ ] Complete claim-versus-reality audit documented in `audits/wave5/w5_12_docs.md`.


---

### Subagent W5-13
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-13.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-13.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-13
# Production Release Documentation (CHANGELOG.md & BREAKING_CHANGES.md)

**Agent ID:** `W5-13`  
**Phase:** 4 — Release Engineering  
**Branch:** `wave5/W5-13`  
**Mission:** Produce publication-ready release notes in `CHANGELOG.md` for version v2.4.0 and document all backward-incompatible structural changes in `BREAKING_CHANGES.md`, complete with copy-paste migration code snippets.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF DOCUMENTATION

### 1.1 Strictly Owned Files (You Create & Commit These)
- `CHANGELOG.md` (Update with exhaustive v2.4.0 section)
- `BREAKING_CHANGES.md` (Update with complete migration catalog)
- `audits/wave5/w5_13_changelog.md` (Release notes audit report)
- `audits/wave5/w5_13_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_13_evidence/recent_commits_dump.txt`
- `audits/wave5/w5_13_evidence/breaking_changes_catalog.json`
- `audits/wave5/w5_13_evidence/placeholder_scan.txt`

### 1.2 Reference Subsystems & Reports to Synthesize (Do NOT Modify Inline)
- `.agent/MASTER_PROMPT.md` (Original 25 P0 and 30 P1 target defects list)
- `audits/wave2_master.md` (Master remediation report across 20 domains)
- `.agent/BENCHMARKS_PROGRESS.json` (Benchmarks 18-agent summary)
- `neuralops/_version.py`

---

## 2. THE MISSION & THE INVARIANTS

Version v2.4.0 represents the unified culmination of multi-wave refactorings:
1. **Keep a Changelog Standards:** `CHANGELOG.md` must organize entries into: Added, Changed, Deprecated, Removed, Fixed, and Security. Every fixed issue must cite its file and defect ID.
2. **Zero Placeholders:** No text may contain `"TODO"`, `"TBD"`, `"Lorem ipsum"`, or vague placeholders.
3. **Actionable Breaking Changes:** Every entry in `BREAKING_CHANGES.md` must explain:
   - What changed
   - Why it changed (architectural justification)
   - A copy-paste Before & After Python code snippet showing how to migrate.

You will synthesize the historical commit record and author complete, professional release documentation.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_13_evidence/
```

### Step 2: Extract Recent Git Commit Lineage
Extract recent commits from the repository history:
```bash
git log --oneline -n 150 > audits/wave5/w5_13_evidence/recent_commits_dump.txt
```

### Step 3: Author the v2.4.0 Section in `CHANGELOG.md`
Prepend or update the `[2.4.0] - 2026-11-15` section in `CHANGELOG.md` with:
- **Added:**
  - `cli_v2`: Modular CLI with lazy subcommand dispatch, `--dry-run`, and RFC 9457 error reporting.
  - `config_v2`: Pydantic v2 strictly validated and frozen YAML configuration engine.
  - `results_v2`: Multi-format atomic ledger writing to JSONL, SQLite, and Parquet tables.
  - `AnyPrecisionAdamW`: Low-memory optimizer with customizable BF16 states and Kahan summation.
  - Standalone DoRA and QLoRA adapter classes in `neuralops.trainer.methods.lora`.
  - Non-CUDA hardware fallback across attention layers (`status=BenchmarkStatus.SKIPPED`).
- **Changed:**
  - Registry discovery converted to lazy module-level functions eliminating import side effects.
  - CUDA timing upgraded to `torch.cuda.Event` with automated L2 cache flushing.
  - `UnifiedHybridCache` hardened with monotonic seen-token tracking and polymorphic cache reordering.
- **Removed:**
  - Removed dummy/fake random evaluations (`niah.py`, `qwen_kd.py`, `e2e_serving.py`).
  - Removed `GenerationMixin` from `GLAModel`.
  - Removed non-functional `TGIRequestBatch.step`.
- **Fixed:**
  - Detailed mapping of P0-1 through P0-25 bug fixes with file and component references.
- **Security:**
  - Enforced `weights_only=True` on PyTorch model deserialization.
  - Bounded memory buffers in `LocalTracker` to prevent unbounded memory growth.

### Step 4: Author `BREAKING_CHANGES.md`
Author the comprehensive migration document detailing:
1. **Renamed API: `continuous_batching_generate` → `generate_sequential`**
   - *Rationale:* The legacy implementation processed requests sequentially despite claiming continuous batching. Renaming restores honesty.
   - *Migration:* Update import and call site.
2. **Behavioral Invariant: `certify_model()` Returns Real Booleans**
   - *Rationale:* Legacy director hardcoded `True` for 8 out of 10 mathematical invariants. Real mathematical checks are now evaluated.
   - *Migration:* Inspect returned dictionary keys and handle unverified invariants.
3. **Strict Validation: `CUDAGraphDecodeRunner` Rejects Dynamic Caches**
   - *Rationale:* Dynamic KV caches cause silent memory corruption during CUDA Graph replay.
   - *Migration:* Pass static caches (`StaticCache` or `is_static=True`).
4. **Configuration Schema: YAML Loading Migrated to Pydantic v2**
   - *Rationale:* Enforces strict types and eliminates silent dictionary key mistyping.
   - *Migration:* Use `config_v2.loader.load_benchmark_config`.

### Step 5: Verify Absence of Placeholders
Scan authored documentation to ensure zero unfinished text:
```bash
grep -rnE "(TODO|TBD|FIXME|XXX|Lorem ipsum)" CHANGELOG.md BREAKING_CHANGES.md > audits/wave5/w5_13_evidence/placeholder_scan.txt || true
```
*Expectation:* File must be 0 lines.

### Step 6: Compile Formal Report `audits/wave5/w5_13_changelog.md`
Author the complete markdown document containing:
- Executive Summary of Release Documentation.
- Section-by-Section Verification Checklist.
- Placeholder Scan Results.
- Summary of Breaking Changes Catalog.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `CHANGELOG.md` updated with exhaustive v2.4.0 section.
- [ ] `BREAKING_CHANGES.md` updated with code snippets for all breaking modifications.
- [ ] Zero placeholder markers found in `placeholder_scan.txt`.
- [ ] Report `audits/wave5/w5_13_changelog.md` written and committed.


---

### Subagent W5-14
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-14.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-14.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-14
# Comprehensive User Migration Guides (v2.3 to v2.4 & v1 to v2)

**Agent ID:** `W5-14`  
**Phase:** 4 — Release Engineering  
**Branch:** `wave5/W5-14`  
**Mission:** Author comprehensive, user-friendly migration guides helping practitioners, researchers, and enterprise deployments upgrade smoothly from legacy versions to v2.4.0 without friction or broken scripts.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF DOCUMENTATION

### 1.1 Strictly Owned Files (You Create & Commit These)
- `docs/migrations/v2.3_to_v2.4.md` (Primary upgrade guide)
- `docs/migrations/v1_to_v2.md` (Legacy major-version migration guide)
- `docs/migrations/README.md` (Migration index and landing page)
- `audits/wave5/w5_14_migration.md` (Migration verification report)
- `audits/wave5/w5_14_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_14_evidence/python_snippets_syntax_check.txt`
- `audits/wave5/w5_14_evidence/migration_files_inventory.txt`

### 1.2 Subsystems & Interfaces Documented (Do NOT Modify Inline)
1. **CLI Commands:** `neuralops benchmark` vs legacy `neuralops-benchmark`
2. **Configuration Schemas:** `config_v2/` YAML formats
3. **Model Surgery:** `neuralops.api.replace_attention`, `save_surgical_delta`
4. **Serving APIs:** `TGIServingAdapter`, `generate_sequential`
5. **Storage & Ledgers:** `BenchmarkLedger`

---

## 2. THE MISSION & THE INVARIANTS

Rule 1.1 and Rule 8.2 of `AGENTS.md` mandate that every architectural transition must provide clear migration documentation:
1. **Real Working Code:** Every code block in the migration guides must be syntactically valid Python.
2. **Contextual Scenarios:** Provide concrete upgrade recipes (e.g. "Upgrading a Fine-Tuning Script", "Upgrading a Benchmark Invocations", "Upgrading a Serving Pipeline").
3. **Index & Cross-Linking:** `docs/migrations/README.md` must serve as a comprehensive navigation index.

You will author the full set of user-facing migration guides and verify their code snippet syntax.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p docs/migrations/ audits/wave5/w5_14_evidence/
```

### Step 2: Author `docs/migrations/v2.3_to_v2.4.md`
Author the primary migration guide covering:
1. **Overview of v2.4.0 Architecture:** Highlights of CLI v2, Config v2, and Results v2.
2. **Recipe 1: Upgrading CLI Benchmark Invocations:**
   - *Legacy:* `python -m neuralops.benchmarks.cli --suite attention --model qwen`
   - *Modern:* `neuralops benchmark run --config configs/attention_sweep.yaml --dry-run`
3. **Recipe 2: Upgrading Benchmark Configurations:**
   - How to adapt legacy untyped YAMLs to strict Pydantic v2 schemas.
4. **Recipe 3: Upgrading Model Surgery & Checkpoints:**
   - Using `save_surgical_delta(..., base_model_path=...)` safely preserving base weights.
   - Managing `UnifiedHybridCache` in autoregressive loops.
5. **Recipe 4: Upgrading Serving & Batch Inference:**
   - Migrating continuous batching calls to `generate_sequential`.

### Step 3: Author `docs/migrations/v1_to_v2.md`
Provide upgrade guidance for enterprise users moving from v1.x monolithic namespaces to modular v2.x architecture:
- Mapping old function paths to `neuralops.layers`, `neuralops.compression`, `neuralops.surgery`.

### Step 4: Author `docs/migrations/README.md`
Create the central landing page:
- Navigation table linking `v1_to_v2.md` and `v2.3_to_v2.4.md`.
- Summary of deprecation timelines and semantic versioning rules.

### Step 5: Verify Python Code Snippets Syntax
Extract and validate Python code snippets from the markdown files:
```bash
python -c "
import glob, re, ast

m_files = glob.glob('docs/migrations/*.md')
results = []
total_snippets = 0
errors = 0

for fpath in m_files:
    with open(fpath) as f:
        content = f.read()
    # Extract ```python ... ``` blocks
    snippets = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    for idx, code in enumerate(snippets):
        total_snippets += 1
        try:
            ast.parse(code)
            results.append(f'{fpath} snippet {idx+1}: VALID')
        except SyntaxError as e:
            errors += 1
            results.append(f'{fpath} snippet {idx+1}: SYNTAX ERROR: {e}')

with open('audits/wave5/w5_14_evidence/python_snippets_syntax_check.txt', 'w') as out:
    out.write('\n'.join(results) + '\n')

print(f'Validated {total_snippets} code snippets in migration guides ({errors} errors).')
"
```

### Step 6: Compile Formal Report `audits/wave5/w5_14_migration.md`
Author the complete markdown document containing:
- Executive Summary of Migration Documentation.
- Inventory of Authored Guides and Sections.
- Snippet Syntax Validation Results.
- Cross-Reference Checklist against `BREAKING_CHANGES.md`.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `docs/migrations/v2.3_to_v2.4.md`, `v1_to_v2.md`, and `README.md` authored.
- [ ] Zero syntax errors in extracted Python code snippets.
- [ ] Evidence files archived in `audits/wave5/w5_14_evidence/`.
- [ ] Report `audits/wave5/w5_14_migration.md` written and committed.


---

### Subagent W5-15
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-15.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-15.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-15
# Version Bump, PEP 517 Build & CLI Entrypoint Verification

**Agent ID:** `W5-15`  
**Phase:** 4 — Release Engineering  
**Branch:** `wave5/W5-15`  
**Mission:** Execute the official version bump to `2.4.0` across all metadata sources, build standard source distribution (sdist) and binary wheel distributions, verify clean installation into an isolated virtual environment, and prove that all registered CLI entrypoints execute and report version `2.4.0`.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF RELEASE BUILD

### 1.1 Strictly Owned Files (You Modify & Commit These)
- `neuralops/_version.py` (Version string bump to `"2.4.0"`)
- `pyproject.toml` (Version field bump under `[project]` only)
- `audits/wave5/w5_15_build.md` (Forensic build & install verification report)
- `audits/wave5/w5_15_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_15_evidence/version_check.txt`
- `audits/wave5/w5_15_evidence/build.log`
- `audits/wave5/w5_15_evidence/dist_contents.txt`
- `audits/wave5/w5_15_evidence/wheel_filelist.txt`
- `audits/wave5/w5_15_evidence/sandbox_install.log`
- `audits/wave5/w5_15_evidence/entrypoint_verification.txt`

### 1.2 Package Metadata & Entrypoints Under Verification
- `pyproject.toml` (`[project.scripts]`: `neuralops`, `nops`, `neuralops-cli`)
- `neuralops/__init__.py` (`__version__` export)

---

## 2. THE MISSION & THE INVARIANTS

A library cannot be published if packaging omits submodules or fails installation:
1. **Single Source of Truth:** `neuralops._version.__version__` and `pyproject.toml` must match `"2.4.0"` exactly.
2. **Submodule Discovery:** The built wheel must contain all core subpackages (`neuralops/layers/`, `neuralops/compression/`, `neuralops/benchmarks/cli_v2/`, `neuralops/surgery/`). Omitting directories breaks runtime imports.
3. **Clean Sandbox Installation:** The wheel must install via `pip install` in a clean, non-editable Python venv without crashing.
4. **Entrypoint Execution:** The installed CLI binaries must run with `--version` and exit with code 0.

You will execute the full build and verification pipeline.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_15_evidence/
```

### Step 2: Version Bump Across Repository
1. Update `neuralops/_version.py`:
   ```python
   __version__ = "2.4.0"
   ```
2. Update `pyproject.toml` version under `[project]`:
   ```toml
   version = "2.4.0"
   ```
3. Verify that programmatic queries resolve to `"2.4.0"`:
   ```bash
   ./.venv/bin/python -c "import neuralops; print(neuralops.__version__)" > audits/wave5/w5_15_evidence/version_check.txt
   ```

### Step 3: Clean Workspace & Build Wheel
1. Clean previous build artifacts:
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```
2. Build sdist and wheel:
   ```bash
   ./.venv/bin/python -m build > audits/wave5/w5_15_evidence/build.log 2>&1
   ```
3. Inspect generated distributions:
   ```bash
   ls -lh dist/ > audits/wave5/w5_15_evidence/dist_contents.txt
   ```
4. Verify subpackage inclusion in the wheel:
   ```bash
   unzip -l dist/*.whl > audits/wave5/w5_15_evidence/wheel_filelist.txt
   ```
   Check for presence of `neuralops/benchmarks/cli_v2/`, `neuralops/layers/linear/`, `neuralops/surgery/director.py`.

### Step 4: Sandbox Installation & Entrypoint Verification
1. Create a temporary sandbox venv:
   ```bash
   python3 -m venv /tmp/neuralops_release_sandbox
   ```
2. Install the newly built wheel into the sandbox:
   ```bash
   /tmp/neuralops_release_sandbox/bin/pip install dist/*.whl > audits/wave5/w5_15_evidence/sandbox_install.log 2>&1
   ```
3. Test CLI entrypoints inside the sandbox:
   ```bash
   python -c "
   import subprocess

   results = []
   for ep in ['neuralops', 'nops', 'neuralops-cli']:
       bin_path = f'/tmp/neuralops_release_sandbox/bin/{ep}'
       try:
           out = subprocess.check_output([bin_path, '--version'], stderr=subprocess.STDOUT).decode().strip()
           results.append(f'{ep}: {out} (SUCCESS)')
       except Exception as e:
           results.append(f'{ep}: FAILED ({e})')

   with open('audits/wave5/w5_15_evidence/entrypoint_verification.txt', 'w') as f:
       f.write('\n'.join(results) + '\n')
   print('\n'.join(results))
   "
   ```
4. Clean up sandbox:
   ```bash
   rm -rf /tmp/neuralops_release_sandbox
   ```

### Step 5: Compile Formal Report `audits/wave5/w5_15_build.md`
Author the complete markdown document containing:
- Executive Summary of Packaging and Build Status (`PASS` or `FAIL`).
- Distribution Artifacts Metadata Table (Filename, SHA-256, Size).
- Wheel Table of Contents Inspection Findings.
- Isolated Sandbox Installation and CLI Entrypoint Verification Outturn.

---

## 4. ACCEPTANCE CRITERIA
- [ ] Version updated to `2.4.0` in `_version.py` and `pyproject.toml`.
- [ ] Wheel builds cleanly with zero errors.
- [ ] Wheel installs in sandbox venv and CLI entrypoints return `2.4.0`.
- [ ] Evidence files archived in `audits/wave5/w5_15_evidence/`.
- [ ] Report `audits/wave5/w5_15_build.md` written and committed.


---

### Subagent W5-16
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-16.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-16.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-16
# CI/CD Workflows, Pre-Commit Hooks & Quality Infrastructure

**Agent ID:** `W5-16`  
**Phase:** 4 — Release Engineering  
**Branch:** `wave5/W5-16`  
**Mission:** Audit, harden, and verify the repository's continuous integration pipelines (`.github/workflows/`), pre-commit hooks (`.pre-commit-config.yaml`), developer automation (`Makefile`), and test coverage configurations (`.coveragerc`), ensuring all quality gates are active and functional.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF INFRASTRUCTURE

### 1.1 Strictly Owned Files (You Modify & Commit These)
- `.github/workflows/ci.yml` (Core CI workflow)
- `.github/workflows/gpu.yml` (Dedicated GPU hardware workflow)
- `.pre-commit-config.yaml` (Local git hook configuration)
- `Makefile` (Local developer command automation)
- `.coveragerc` (Coverage reporting exclusions)
- `audits/wave5/w5_16_ci.md` (CI/CD quality infrastructure report)
- `audits/wave5/w5_16_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_16_evidence/make_lint_output.txt`
- `audits/wave5/w5_16_evidence/make_test_output.txt`
- `audits/wave5/w5_16_evidence/workflow_lint_check.txt`

### 1.2 Configuration Files Under Review
- `pyproject.toml` (Pytest options, coverage configuration)

---

## 2. THE MISSION & THE INVARIANTS

Quality gates prevent future code decay:
1. **Multi-Python CI Matrix:** `.github/workflows/ci.yml` must test across supported Python versions (`3.10`, `3.11`, `3.12`).
2. **GPU Workflow Separation:** GPU-dependent tests (`@pytest.mark.gpu`) must be isolated into `.github/workflows/gpu.yml` so that standard CPU CI runs remain fast and never fail due to missing CUDA drivers.
3. **Local Developer Ergonomics:** `make lint` and `make test` must execute cleanly on local developer machines matching remote CI rules.
4. **Coverage Integrity:** `.coveragerc` must omit legacy wrappers and test stubs while enforcing minimum coverage on core modules.

You will audit and calibrate the complete CI/CD and developer automation stack.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p .github/workflows/ audits/wave5/w5_16_evidence/
```

### Step 2: Audit & Calibrate `.github/workflows/ci.yml`
Ensure `.github/workflows/ci.yml` contains:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check neuralops/ tests/
      - run: ruff format --check neuralops/ tests/
      - run: mypy --strict neuralops/benchmarks/config_v2/ neuralops/benchmarks/results_v2/ neuralops/benchmarks/cli_v2/

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest tests/unit tests/contract -v
      - run: pytest tests/integration -v -m "not hub and not gpu"
```

### Step 3: Audit & Calibrate `.github/workflows/gpu.yml`
Ensure a dedicated GPU workflow exists for self-hosted runners:
- Runs `pytest tests/hardware -v -m "gpu"`.
- Runs Triton / CUDA kernel parity verification tests.

### Step 4: Audit & Calibrate `Makefile`
Ensure `Makefile` provides essential developer targets:
```makefile
.PHONY: lint test coverage build clean

lint:
	ruff check neuralops/ tests/
	ruff format --check neuralops/ tests/

test:
	pytest tests/unit tests/contract -v

coverage:
	pytest --cov=neuralops --cov-report=term-missing --cov-report=html

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
```

### Step 5: Test Local Make Targets
Execute local commands and save outputs:
```bash
make lint > audits/wave5/w5_16_evidence/make_lint_output.txt 2>&1 || true
make test > audits/wave5/w5_16_evidence/make_test_output.txt 2>&1 || true
```

### Step 6: Validate YAML Syntax of Workflows
Validate that workflow files are well-formed YAML:
```bash
python -c "
import glob, yaml

results = []
for yf in glob.glob('.github/workflows/*.yml'):
    try:
        yaml.safe_load(open(yf))
        results.append(f'{yf}: VALID YAML')
    except Exception as e:
        results.append(f'{yf}: INVALID: {e}')

with open('audits/wave5/w5_16_evidence/workflow_lint_check.txt', 'w') as f:
    f.write('\n'.join(results) + '\n')
print('\n'.join(results))
"
```

### Step 7: Compile Formal Report `audits/wave5/w5_16_ci.md`
Author the complete markdown document containing:
- Executive Summary of CI/CD Quality Gates.
- Inventory of Workflows and Triggers.
- Local Make Targets Execution Outturn.
- Coverage Policy & Pre-Commit Hook Configuration.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `.github/workflows/ci.yml` and `.github/workflows/gpu.yml` present and valid YAML.
- [ ] `Makefile` present and verified locally.
- [ ] Evidence files archived in `audits/wave5/w5_16_evidence/`.
- [ ] Report `audits/wave5/w5_16_ci.md` written and committed.


---

### Subagent W5-17
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-17.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-17.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-17
# Full Repository Test Suite Execution & Failure Categorization

**Agent ID:** `W5-17`  
**Phase:** 5 — Final Acceptance  
**Branch:** `wave5/W5-17`  
**Mission:** Execute the entire test suite of the NeuralOps repository, capture complete verbose logs, parse and categorize every single test result (passed, skipped, xfailed, failed), distinguish between expected hardware skips and unintended regressions, and compile the final repository test scorecard.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF TEST RUN

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_17_full_tests.md` (Forensic full test report)
- `audits/wave5/w5_17_evidence/` (Directory storing all command outputs and logs)
- `audits/wave5/w5_17_evidence/pytest_full_run.log`
- `audits/wave5/w5_17_evidence/pytest_summary_parsed.json`
- `audits/wave5/w5_17_evidence/failed_tests_breakdown.txt`
- `audits/wave5/w5_17_evidence/skipped_tests_breakdown.txt`

### 1.2 Test Suites Under Execution (Every Test File in Repo)
- `tests/unit/**/*.py`
- `tests/contract/**/*.py`
- `tests/integration/**/*.py`
- `tests/models/**/*.py`
- `tests/surgery/**/*.py`
- `tests/pipeline_transform/**/*.py`
- `tests/layers/**/*.py`
- `tests/compression/**/*.py`
- `tests/ops/**/*.py`
- `tests/trainer/**/*.py`
- `tests/benchmarks/**/*.py`
- `tests/cli/**/*.py`
- `tests/validation/**/*.py`

---

## 2. THE MISSION & THE INVARIANTS

This is the ultimate regression gate for release v2.4.0:
1. **Zero Unhandled Regressions:** All unit, contract, and core integration tests must pass on standard CPU/Apple Silicon hardware.
2. **Transparent Skips:** Any skipped test must be justified by hardware requirements (e.g. CUDA device required) or optional network tokens. Unconditional skips without reason violate Rule 5.4.
3. **Forensic Blame Attribution:** For any failing test, you must determine whether the failure was pre-existing in legacy code or introduced by recent changes.

You will execute the full suite, capture raw logs, and produce a detailed categorization.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_17_evidence/
```

### Step 2: Execute the Complete Test Suite
Run pytest across the entire repository with full verbosity:
```bash
./.venv/bin/pytest tests/ -v --maxfail=0 > audits/wave5/w5_17_evidence/pytest_full_run.log 2>&1 || true
```

### Step 3: Parse and Classify Results via Python
Author an automated log analyzer:
```bash
python -c "
import re, json
from collections import Counter, defaultdict

with open('audits/wave5/w5_17_evidence/pytest_full_run.log') as f:
    log_content = f.read()

lines = log_content.splitlines()
status_counts = Counter()
failed_tests = []
skipped_tests = []

for line in lines:
    m = re.search(r'^(tests/[^:]+::\S+)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS)', line)
    if m:
        node_id, status = m.groups()
        status_counts[status] += 1
        if status == 'FAILED':
            failed_tests.append(node_id)
        elif status == 'SKIPPED':
            skipped_tests.append(node_id)

summary = {
    'total_discovered': sum(status_counts.values()),
    'passed': status_counts['PASSED'],
    'failed': status_counts['FAILED'],
    'skipped': status_counts['SKIPPED'],
    'xfail': status_counts['XFAIL'],
    'xpass': status_counts['XPASS'],
}

with open('audits/wave5/w5_17_evidence/pytest_summary_parsed.json', 'w') as f:
    json.dump(summary, f, indent=2)

with open('audits/wave5/w5_17_evidence/failed_tests_breakdown.txt', 'w') as f:
    f.write('\n'.join(failed_tests) + '\n')

with open('audits/wave5/w5_17_evidence/skipped_tests_breakdown.txt', 'w') as f:
    f.write('\n'.join(skipped_tests) + '\n')

print(f'Test Run Summary: {summary}')
"
```

### Step 4: Root-Cause Analysis of Any Failures
For every failed test in `failed_tests_breakdown.txt`:
1. Search the failure traceback in `pytest_full_run.log`.
2. Determine if the failure is related to legacy benchmark harness shims or an active core module.
3. File an entry in `issues/wave6_hotfix_backlog.md` with full failure details.

### Step 5: Compile Formal Report `audits/wave5/w5_17_full_tests.md`
Author the complete markdown document containing:
- Executive Testing Verdict: `PASS` (All non-GPU tests green) or `FAIL`.
- High-Level Metrics Table (Total, Passed, Failed, Skipped, Xfail).
- Breakdown Table by Test Directory (`tests/models`, `tests/surgery`, etc.).
- Detailed Root-Cause Analysis for any Failures.
- Full List of Logged Wave 6 Hotfix Items.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `pytest_full_run.log` captured and archived.
- [ ] Complete JSON and text summaries generated in `audits/wave5/w5_17_evidence/`.
- [ ] Detailed failure analysis documented in `audits/wave5/w5_17_full_tests.md`.


---

### Subagent W5-18
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-18.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-18.md)

# Wave 5 Agent W5-18: Comprehensive Repository Security, Deserialization & Vulnerability Audit

**Agent ID:** W5-18  
**Phase:** 5 — Final Acceptance  
**Branch:** `wave5/W5-18`  
**Mission:** Execute an uncompromising security audit across the entire codebase to detect unsafe deserialization (`torch.load` without `weights_only=True`), exposed cloud/API credentials, deprecated vulnerable PyTorch AMP calls, arbitrary code execution risks in preprocessing hooks, and vulnerable Python dependencies.

---

## 1. Context, Architecture & Mandatory Boundaries

Machine learning repositories are primary targets for supply-chain attacks via malicious pickled checkpoints, credential harvesting in configuration files, and unvetted preprocessing hooks. NeuralOps v2.4.0 must guarantee enterprise-grade security posture before publication.

As Agent **W5-18**, you will run automated static analysis, entropy-based token scanning, and package dependency vulnerability scans across every single file in the repository.

### Explicitly Owned Paths (Files You Create / Write):
- `audits/wave5/w5_18_security.md` (Authoritative security audit report)
- `audits/wave5/w5_18_evidence/` (Evidence directory)
- `audits/wave5/w5_18_evidence/torch_load_sites.txt`
- `audits/wave5/w5_18_evidence/unsafe_weights_only.txt`
- `audits/wave5/w5_18_evidence/secret_leaks.txt`
- `audits/wave5/w5_18_evidence/deprecated_amp_sites.txt`
- `audits/wave5/w5_18_evidence/dynamic_execution.txt`
- `audits/wave5/w5_18_evidence/pip_audit.txt`

### Related Code Files to Inspect & Scan:
- All `.py` files under `neuralops/`
- All test files under `tests/`
- All configuration YAMLs and markdown docs under `docs/` and `examples/`

---

## 2. Exhaustive Step-by-Step Execution Plan

### Step 1: Scaffolding Setup
Create the evidence directory:
```bash
mkdir -p audits/wave5/w5_18_evidence/
```

### Step 2: Unsafe Weight Deserialization Audit (`torch.load`)
1. Extract every call to `torch.load` across `neuralops/`:
   ```bash
   grep -rn "torch.load" neuralops/ --include="*.py" > audits/wave5/w5_18_evidence/torch_load_sites.txt || true
   ```
2. Scan specifically for unsafe `weights_only=False` or calls omitting `weights_only`:
   ```bash
   grep -rn "weights_only=False" neuralops/ | grep -v "# noqa" > audits/wave5/w5_18_evidence/unsafe_weights_only.txt || true
   ```
   *Requirement:* Zero occurrences of unannotated `weights_only=False`.
3. Check `neuralops/models/surgical_io.py` and `neuralops/hub/` to verify that all checkpoint loading enforces `weights_only=True` by default.

### Step 3: Hardcoded Secret & Token Entropy Scan
1. Scan the repository for high-entropy tokens, private keys, and cloud API credentials:
   ```bash
   grep -rEn "(hf_[A-Za-z0-9]{34}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\\-_]{35})" neuralops/ tests/ docs/ \
       | grep -v "SECRET_PATTERNS" \
       | grep -v "test_" \
       > audits/wave5/w5_18_evidence/secret_leaks.txt || true
   ```
   *Requirement:* Zero exposed credentials. Any match is an instant P0 security blocker.

### Step 4: Deprecated AMP API Scan (Rule 2.1)
1. Verify that no deprecated PyTorch CUDA AMP imports exist:
   ```bash
   grep -rn "torch.cuda.amp" neuralops/ --include="*.py" > audits/wave5/w5_18_evidence/deprecated_amp_sites.txt || true
   ```
   *Requirement:* Zero lines. Modern `torch.amp` must be used across all components.

### Step 5: Dynamic Code Execution in Preprocessing Scripts
1. Scan for dangerous dynamic execution constructs (`eval(`, `exec(`, `load_preprocess_script`):
   ```bash
   grep -rn "eval(\|exec(\|load_preprocess_script" neuralops/ --include="*.py" > audits/wave5/w5_18_evidence/dynamic_execution.txt || true
   ```
2. Inspect `neuralops/data/preprocessing.py`: Verify that dynamic script loading mandates `trust=True` from the user, preventing drive-by arbitrary code execution from downloaded untrusted repositories.

### Step 6: Dependency Vulnerability Audit (`pip-audit`)
1. Run vulnerability scanning on installed packages in the environment:
   ```bash
   ./.venv/bin/pip-audit > audits/wave5/w5_18_evidence/pip_audit.txt 2>&1 || true
   ```
2. Parse output to check for known CVEs. If external network or pip-audit is unavailable in the environment, document local package version pinning.

### Step 7: Log Any Discovered Vulnerabilities
If any security flaw is discovered:
1. File an urgent entry in `issues/wave6_hotfix_backlog.md` with:
   - Vulnerability Type
   - Severity (Critical / High / Medium / Low)
   - Exact File and Line Number
   - Remediation Guidance.

### Step 8: Compile Formal Security Audit Report `audits/wave5/w5_18_security.md`
Author the complete markdown audit report containing:
- Executive Security Verdict: `PASS` (0 High/Critical Vulnerabilities) or `BLOCKED`.
- Complete Vulnerability Matrix covering all 5 inspection domains.
- Summary of `torch.load` safety enforcement.
- Credential and Secret Scan Findings.
- Dependency CVE Assessment.
- Logged Wave 6 Hotfix Items.

---

## 3. Acceptance Criteria
- [ ] Evidence files generated in `audits/wave5/w5_18_evidence/`.
- [ ] Zero unannotated `weights_only=False` or exposed secrets.
- [ ] Zero `torch.cuda.amp` imports in `neuralops/`.
- [ ] Report `audits/wave5/w5_18_security.md` written and committed.


---

### Subagent W5-19
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-19.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-19.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-19
# Performance Baseline Profiling & Regression Safeguards

**Agent ID:** `W5-19`  
**Phase:** 5 — Final Acceptance  
**Branch:** `wave5/W5-19`  
**Mission:** Profile and capture a rigorous latency and memory baseline across flagship attention layers and compression algorithms, store machine-readable baseline records in JSON format, and verify that no regressions exceeding 15% have been introduced by refactoring waves.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF PROFILING

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5/w5_19_baseline.json` (Machine-readable performance baseline)
- `audits/wave5/w5_19_perf.md` (Forensic performance profiling report)
- `audits/wave5/w5_19_evidence/` (Evidence directory)
- `audits/wave5/w5_19_evidence/profiling_raw_timings.json`
- `audits/wave5/w5_19_evidence/hardware_environment.txt`
- `audits/wave5/w5_19_evidence/benchmark_comparison.txt`

### 1.2 Target Components & Architectures to Profile
1. **Flagship Attention Layers:**
   - `cosformer` (Linear attention)
   - `decay_linear_attention` (Linear attention with decay)
   - `gla_attention` (Gated linear attention)
   - `sliding_window` (Local attention)
2. **Flagship Compression Layers:**
   - `direct_svd`
   - `kmeans`
3. **Hardware / Timing Utilities:**
   - `neuralops/benchmarks/utils/timer.py`
   - `neuralops/benchmarks/utils/memory.py`

---

## 2. THE MISSION & THE INVARIANTS

Refactoring for safety or clean abstractions must never degrade computational throughput:
1. **Rule 9.1 (Warmup):** Every benchmark must execute at least 10 warmup iterations before timing begins.
2. **Rule 9.2 (Outlier Handling):** Never report a single mean number. Report `median_ms`, `p90_ms`, and `p99_ms` after discarding extreme outliers.
3. **Hardware Grounding:** Always document the physical hardware environment (CPU model, GPU model, memory).
4. **Machine-Readable Baseline:** Store outputs in structured JSON format (`audits/wave5/w5_19_baseline.json`) so future CI PRs can compute deltas.

You will profile core operations, capture telemetry, and establish the v2.4.0 baseline.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_19_evidence/
```

### Step 2: Capture Hardware Environment Telemetry
Extract and record system specifications:
```bash
python -c "
import platform, sys, torch

env = {
    'platform': platform.platform(),
    'processor': platform.processor(),
    'python_version': sys.version,
    'pytorch_version': torch.__version__,
    'mps_available': torch.backends.mps.is_available(),
    'cuda_available': torch.cuda.is_available(),
}

with open('audits/wave5/w5_19_evidence/hardware_environment.txt', 'w') as f:
    for k, v in env.items():
        f.write(f'{k}: {v}\n')
print('Hardware environment captured.')
"
```

### Step 3: Execute Profiling Harness
Author and execute a robust profiling script:
```bash
python -c "
import json, time, torch
from neuralops.core.registry import create_attention

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
targets = ['cosformer', 'decay_linear_attention', 'gla_attention']
configs = [
    {'seq_len': 128, 'batch_size': 2, 'dim': 64, 'heads': 4},
    {'seq_len': 512, 'batch_size': 2, 'dim': 64, 'heads': 4},
]

raw_timings = {}
baseline_summary = {}

for name in targets:
    raw_timings[name] = {}
    baseline_summary[name] = {}
    for cfg in configs:
        cfg_key = f's{cfg[\"seq_len\"]}_b{cfg[\"batch_size\"]}'
        try:
            model = create_attention(name, dim=cfg['dim'], num_heads=cfg['heads']).to(device)
            model.eval()
            x = torch.randn(cfg['batch_size'], cfg['seq_len'], cfg['dim'], device=device)

            # Rule 9.1: Warmup 10 iterations
            for _ in range(10):
                _ = model(x)

            # Timed iterations (25 runs)
            runs = []
            for _ in range(25):
                t0 = time.perf_counter()
                _ = model(x)
                runs.append((time.perf_counter() - t0) * 1000)

            # Rule 9.2: Sort and calculate percentiles
            runs.sort()
            # Discard top/bottom 5%
            filtered = runs[1:-1]
            median = filtered[len(filtered)//2]
            p90 = filtered[int(len(filtered)*0.9)]
            p99 = filtered[-1]

            raw_timings[name][cfg_key] = runs
            baseline_summary[name][cfg_key] = {
                'median_ms': round(median, 4),
                'p90_ms': round(p90, 4),
                'p99_ms': round(p99, 4),
                'device': str(device),
            }
        except Exception as e:
            baseline_summary[name][cfg_key] = {'error': str(e)}

with open('audits/wave5/w5_19_evidence/profiling_raw_timings.json', 'w') as f:
    json.dump(raw_timings, f, indent=2)

with open('audits/wave5/w5_19_baseline.json', 'w') as f:
    json.dump(baseline_summary, f, indent=2)

print('Baseline profiling completed.')
"
```

### Step 4: Compare Against Historical Ledgers
Inspect any existing entries in `results/` or prior audits to detect unexpected regressions (>15% slower). Document findings in `audits/wave5/w5_19_evidence/benchmark_comparison.txt`.

### Step 5: Compile Formal Report `audits/wave5/w5_19_perf.md`
Author the complete markdown document containing:
- Executive Summary of Performance Health.
- Hardware Platform Specification Table.
- Latency Baseline Scorecard Table (Architecture, Config, Median ms, P90 ms, P99 ms).
- Regression Analysis & Observations.
- Recommendations for High-Throughput Deployment.

---

## 4. ACCEPTANCE CRITERIA
- [ ] `audits/wave5/w5_19_baseline.json` populated with valid latency percentiles.
- [ ] Evidence files archived in `audits/wave5/w5_19_evidence/`.
- [ ] Full performance report `audits/wave5/w5_19_perf.md` written and committed.


---

### Subagent W5-20
**Specification Path:** [`.agent/WAVE5_AGENTS/W5-20.md`](file:///Users/tahamajs/dev/Project/.agent/WAVE5_AGENTS/W5-20.md)

# COMPLETE OPERATIONAL SPECIFICATION: W5-20
# Master Release Acceptance Synthesis & Authorization

**Agent ID:** `W5-20`  
**Phase:** 5 — Final Acceptance  
**Branch:** `wave5/W5-20`  
**Mission:** Synthesize all 19 prior audit reports, execute the master release gate script `scripts/verify_release.sh`, curate the consolidated hotfix backlog `issues/wave6_hotfix_backlog.md`, author the authoritative release verdict in `audits/wave5_master.md`, and formulate the tag recommendation for human authorization.

---

## 1. COMPREHENSIVE FILE INVENTORY & SCOPE OF SYNTHESIS

### 1.1 Strictly Owned Files (You Create & Commit These)
- `audits/wave5_master.md` (Authoritative Master Release Acceptance Report)
- `.agent/WAVE5_PROGRESS.json` (Final synchronized release status)
- `audits/wave5/w5_20_evidence/` (Evidence directory)
- `audits/wave5/w5_20_evidence/final_gate_execution.log`
- `audits/wave5/w5_20_evidence/prior_reports_checklist.txt`
- `audits/wave5/w5_20_evidence/release_invariants_validation.txt`

### 1.2 Reports & Artifacts to Synthesize (Do NOT Modify Inline)
- All 19 reports under `audits/wave5/w5_*.md` (W5-1 through W5-19)
- `issues/wave6_hotfix_backlog.md` (Consolidated findings backlog)
- `CHANGELOG.md` and `BREAKING_CHANGES.md`
- `docs/truth_audit.md`
- `audits/wave5/w5_19_baseline.json`
- `scripts/verify_release.sh`

---

## 2. THE MISSION & THE INVARIANTS

As Agent **W5-20**, you are the Supreme Release Officer. You do not re-run every benchmark from scratch; you synthesize the forensic findings of all 19 prior agents into an irrefutable release verdict:
1. **Unconditional Integrity:** If ANY prior agent reported `BLOCKED` or if `scripts/verify_release.sh` fails, you MUST report `BLOCKED`. You are forbidden from papering over failures.
2. **Consolidated Backlog:** Every non-blocking discrepancy found across W5-1 to W5-19 must be categorized in `issues/wave6_hotfix_backlog.md`.
3. **Strict Human Tagging Gate:** You must NEVER run `git tag v2.4.0` yourself. You must output an explicit approval proposal for the human coordinator.

---

## 3. EXHAUSTIVE STEP-BY-STEP EXECUTION PROTOCOL

### Step 1: Scaffolding Setup
```bash
mkdir -p audits/wave5/w5_20_evidence/
```

### Step 2: Audit Presence of All 19 Prior Reports
Verify that every single domain report exists and is populated:
```bash
python -c "
import os

reports = [
    'audits/wave5/w5_1_wave1_wave2.md',
    'audits/wave5/w5_2_wave3_wave4.md',
    'audits/wave5/w5_3_benchmarks.md',
    'audits/wave5/w5_4_pipeline.md',
    'audits/wave5/w5_5_zoos.md',
    'audits/wave5/w5_6_training.md',
    'audits/wave5/w5_7_serving.md',
    'audits/wave5/w5_8_attentions.md',
    'audits/wave5/w5_9_errors.md',
    'audits/wave5/w5_10_logging.md',
    'audits/wave5/w5_11_types.md',
    'audits/wave5/w5_12_docs.md',
    'audits/wave5/w5_13_changelog.md',
    'audits/wave5/w5_14_migration.md',
    'audits/wave5/w5_15_build.md',
    'audits/wave5/w5_16_ci.md',
    'audits/wave5/w5_17_full_tests.md',
    'audits/wave5/w5_18_security.md',
    'audits/wave5/w5_19_perf.md',
]

status = []
all_present = True
for r in reports:
    exists = os.path.exists(r)
    size = os.path.getsize(r) if exists else 0
    if not exists or size == 0:
        all_present = False
        status.append(f'{r}: MISSING OR EMPTY')
    else:
        status.append(f'{r}: PRESENT ({size} bytes)')

with open('audits/wave5/w5_20_evidence/prior_reports_checklist.txt', 'w') as f:
    f.write('\n'.join(status) + '\n')

print(f'Checked {len(reports)} reports. All present: {all_present}')
"
```

### Step 3: Execute Master Release Verification Script
Run the automated release gate script:
```bash
./scripts/verify_release.sh > audits/wave5/w5_20_evidence/final_gate_execution.log 2>&1
echo "Gate Exit Code: $?"
```
Assert that the script exits with code `0`.

### Step 4: Validate Release Invariants
Run a python validation check verifying core release invariants:
```bash
python -c "
import os
import neuralops

invariants = []

# 1. Version matches 2.4.0
assert neuralops.__version__ == '2.4.0', f'Version mismatch: {neuralops.__version__}'
invariants.append('Version is 2.4.0: PASS')

# 2. Release files present
for f in ['CHANGELOG.md', 'BREAKING_CHANGES.md', 'docs/truth_audit.md', 'issues/wave6_hotfix_backlog.md']:
    assert os.path.exists(f) and os.path.getsize(f) > 0, f'Missing release file: {f}'
invariants.append('Release documentation files present: PASS')

# 3. Built wheel present
assert os.path.exists('dist') and len(os.listdir('dist')) >= 2, 'Missing built wheel or sdist in dist/'
invariants.append('Wheel and sdist present: PASS')

with open('audits/wave5/w5_20_evidence/release_invariants_validation.txt', 'w') as f:
    f.write('\n'.join(invariants) + '\n')
print('\n'.join(invariants))
"
```

### Step 5: Author Authoritative Master Report `audits/wave5_master.md`
Synthesize all findings into the master document containing:
1. **Executive Verdict:** `READY TO TAG v2.4.0` or `BLOCKED`.
2. **Phase-by-Phase Scorecard Table:** Summary of all 20 agents (W5-1 through W5-20).
3. **Integration Verification Matrix:** Config-to-results, 6 zoos, training, serving, and 131 attentions.
4. **Cross-Cutting Quality Scorecard:** Error hierarchy, logging, strict typing, and documentation truth.
5. **Security & Performance Attestation:** Unsafe deserialization audit, zero secret leaks, performance baseline.
6. **Wave 6 Hotfix Backlog Table:** Consolidated catalog of non-blocking polish items.
7. **Actionable Proposal for Human Coordinator:** Explicit instructions on git commands for tagging `v2.4.0`.

### Step 6: Synchronize Progress Ledger
Update `.agent/WAVE5_PROGRESS.json`, marking all 20 agents as `complete` with timestamps.

---

## 4. ACCEPTANCE CRITERIA
- [ ] All 19 prior audit reports synthesized.
- [ ] `scripts/verify_release.sh` executes with code 0 and log captured.
- [ ] `audits/wave5_master.md` authored with exhaustive phase-by-phase breakdown.
- [ ] Progress file `.agent/WAVE5_PROGRESS.json` finalized.
- [ ] Tag recommendation prepared without auto-tagging.


---
