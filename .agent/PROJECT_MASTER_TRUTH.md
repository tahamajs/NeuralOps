# NEURALOPS — PROJECT MASTER TRUTH FILE
# `.agent/PROJECT_MASTER_TRUTH.md`

**Version:** 1.0  
**Effective Date:** 2026-11-15  
**Owner:** Human Coordinator (Taha Majlesi)  
**Status:** BINDING — All agents, all waves, all phases must read this file first.

> **MANDATORY:** Every subagent spawned in this repository — regardless of wave, role, or scope —
> MUST read this file at the start of its session, before reading its own task spec.
> This is the single source of truth for the project. If any other document contradicts this
> file, THIS FILE TAKES PRECEDENCE.

---

## 0. WHAT THIS FILE IS

This is the **master truth document** for the NeuralOps project. It contains:

1. **Project identity** — what NeuralOps is, what it does, what version it's at.
2. **Repository structure** — every important directory and what lives inside it.
3. **Canonical rules** — the non-negotiable laws that govern all code, tests, and commits.
4. **Wave architecture** — the multi-wave remediation plan and what each wave does.
5. **Shared invariants** — what every agent must check and enforce, regardless of its assigned task.
6. **Cross-file mandate** — explicit permission and obligation to fix related files.
7. **Progress state** — where the project stands right now.

No agent should need to guess about project intent. Everything is here.

---

## 1. PROJECT IDENTITY

| Field | Value |
|---|---|
| **Project Name** | NeuralOps (`SVD_linear_Attention`) |
| **Current Version** | `2.4.0` (defined in `neuralops/_version.py`) |
| **CLI Entry** | `neuralops --version` → must output `NeuralOps 2.4.0` |
| **Package Root** | `neuralops/` |
| **Test Root** | `tests/` |
| **Examples Root** | `examples/` |
| **Docs Root** | `docs/` |
| **Audits Root** | `audits/` |
| **Issues/Backlog** | `issues/wave6_hotfix_backlog.md` |
| **Git Default Branch** | `main` |
| **PyPI target** | `neuralops==2.4.0` |

### 1.1 — What NeuralOps Is

NeuralOps is a **production-grade deep learning systems framework** for:
- **131 canonical attention mechanisms** (from linear and linear-recurrent to flash/sliding/sparse variants)
- **72+ compression engines** (SVD, quantization INT4/FP8/MX, structured pruning, LoRA, QLoRA, DoRA)
- **58 Triton JIT GPU kernels** + **140 native CUDA kernel packages**
- **Model Surgery system** (`neuralops.surgery`) — in-place surgical replacement of attention, MLP, and embedding layers
- **Multi-zoo HuggingFace Hub** — `Hooshaai/bert-*`, `Hooshaai/qwen-*`, TinyLlama, LLaVA-VLM loaded via `neuralops.hub`
- **Benchmarking framework** (`neuralops.benchmarks`) — `config_v2`, `cli_v2`, `results_v2`
- **Training pipeline** — finetuning, knowledge distillation (6-pillar KD), recovery, online TTT
- **Serving pipeline** — continuous batching, paged KV-cache, CUDA Graph decode runner

---

## 2. REPOSITORY LAYOUT (CANONICAL MAP)

```
.agent/                             ← All orchestration and planning files
  PROJECT_MASTER_TRUTH.md           ← THIS FILE (read first, always)
  MASTER_PROMPT.md                  ← Original 25 P0 / 30 P1 bug list
  WAVE5_EXECUTE.md                  ← Wave 5 orchestration spec (20 agents)
  WAVE5_AGENTS/W5-{1..20}.md       ← Individual Wave 5 agent specs
  WAVE5_PROGRESS.json               ← Live Wave 5 progress tracker
  WAVE6_EXECUTE.md                  ← Wave 6 orchestration spec (26 agents)
  WAVE6_AGENTS/W6-{1..26}.md       ← Individual Wave 6 agent specs
  WAVE6_PROGRESS.json               ← Live Wave 6 progress tracker

AGENTS.md                           ← HARD RULES for all code (read this too)

neuralops/                          ← Library source code (1808 Python files)
  _version.py                       ← Version = "2.4.0" (do not hand-edit)
  benchmarks/
    config_v2/                      ← Pydantic v2 config engine
    cli_v2/                         ← New CLI (neuralops subcommands)
    results_v2/                     ← Atomic ledger (JSONL + SQLite + Parquet)
    suites_v2/                      ← Benchmark suite definitions
    registry.py                     ← Lazy benchmark registry
    executor.py                     ← Subprocess-isolated executor (frozen)
  surgery/
    director.py                     ← UniversalSurgeryDirector (canonical)
  trainer/
    distillation/                   ← MultiLevelKDTrainer, 6-pillar losses
  serving/                          ← Batching, streaming, paged cache
  hub/                              ← HuggingFace Hub loader

tests/                              ← All tests (1049 Python files)
  unit/                             ← No GPU, no network, <100ms each
  contract/                         ← Schema and contract validation
  integration/                      ← End-to-end pipeline tests
  examples/                         ← Subprocess tests for each example
  hardware/                         ← @pytest.mark.gpu tests only

examples/                           ← 35 demonstration scripts (NN_*.py)
  _shared/                          ← Shared utilities (TinyClassifier, etc.)

docs/                               ← All documentation
audits/                             ← Audit reports from all waves
  wave5/                            ← Wave 5 audit evidence
  wave6/                            ← Wave 6 audit evidence

issues/                             ← Bug backlogs
  wave6_hotfix_backlog.md           ← All bugs found during verification
```

---

## 3. THE THREE LAWS (SUPREME RULES — NEVER VIOLATE)

These override everything else, including individual wave specs:

### LAW 1 — BACKWARD COMPATIBILITY
Every change must leave existing public APIs, configs, and CLI flags working exactly as before. If you must break something, the old API must emit a `DeprecationWarning` pointing to the new one, and a migration snippet must exist in `docs/migrations/`.

### LAW 2 — REVERSIBILITY
Every commit must be independently revertable via `git revert` without cascading breakage. This means:
- One logical change per commit.
- No change that depends on a future, uncommitted change.
- No sweeping refactors that touch 10+ files in one commit.

### LAW 3 — ISOLATION
New code goes in a new module or directory. Old code is touched only when:
(a) it has a proven bug that must be fixed, OR
(b) a new module has been proven and callers are being migrated one at a time.

---

## 4. HARD TECHNICAL RULES (FROM AGENTS.MD — ENFORCED FOR ALL AGENTS)

Read `AGENTS.md` in full. The most critical rules, summarized:

### 4.1 — PyTorch API Rules
```python
# ✅ CORRECT
from torch.amp import autocast, GradScaler
with autocast("cuda", dtype=torch.float16): ...
scaler = GradScaler("cuda", enabled=True)

# ❌ FORBIDDEN
from torch.cuda.amp import autocast, GradScaler  # deprecated API
```

```python
# ✅ CORRECT — GPU timing
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record(); fn(); end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)

# ❌ FORBIDDEN
t0 = time.time(); fn(); elapsed = time.time() - t0  # useless for GPU
```

```python
# ✅ CORRECT — L2 cache flush before timing
def flush_l2(device: torch.device) -> None:
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    buf = torch.empty(l2_bytes // 4, dtype=torch.int32, device=device)
    buf.zero_(); torch.cuda.synchronize(device)

# ❌ FORBIDDEN
buf = torch.empty(256 * 1024 * 1024 // 4, ...)  # hardcoded 256MB
```

```python
# ✅ CORRECT — Device selection
device = resolve_device(config.device)   # from config or arg

# ❌ FORBIDDEN
device = torch.device("cuda:0")          # hardcoded device index
```

### 4.2 — Pydantic v2 Rules (config_v2/)
```python
# ✅ CORRECT
class MyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    items: tuple[str, ...] = Field(default_factory=tuple)

# ❌ FORBIDDEN
class MyConfig:
    items = []           # mutable default
    items: list[str]     # list in frozen model
```

### 4.3 — Result Record Rules
```python
# ✅ CORRECT
return CanonicalRecord(
    status=BenchmarkStatus.SUCCESS,
    metrics={"latency_ms": 1.2},
    ...
)

# ❌ FORBIDDEN
return {"status": "PASS", "latency": 1.2}   # raw dict
return {"status": "SUCCESS", ...}            # raw string status
```

### 4.4 — Import-Time Side Effect Rules
```python
# ✅ CORRECT
_REGISTRY: dict | None = None
def get_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY

# ❌ FORBIDDEN
BenchmarkRegistry.populate_canonical_benchmarks()   # at module level
```

### 4.5 — Error Handling Rules
```python
# ✅ CORRECT
except (EOFError, ConnectionResetError):
    logger.error("Worker crashed: %s", e)
    return BenchmarkResult(status=BenchmarkStatus.HARDWARE_ERROR, ...)

# ❌ FORBIDDEN
except Exception: pass                 # silent swallow
except Exception: return 0.0           # hiding errors as zero
```

### 4.6 — No main() in Library Files
- `def main()` and `if __name__ == "__main__":` are forbidden in `neuralops/benchmarks/` except in `cli_v2/main.py`.
- All entrypoints are registered via `pyproject.toml`.

---

## 5. THE WAVE ARCHITECTURE — WHAT EACH WAVE DOES

### Wave 1 (Complete)
**Scope:** 10 agents, 45 P0/P1 bugs across core library.
**Mission:** Fix the most critical correctness and safety bugs identified in the initial audit.
**Evidence:** 10 merged branches, `pytest tests/` green.

### Wave 2 (Complete)
**Scope:** 20 agents, 20 subsystems.
**Mission:** Deep per-subsystem remediation (attention zoo, compression, serving, training, CLI, results, etc.).
**Evidence:** `audits/wave2_master.md`, branches merged to main.

### Wave 3 (Complete or Skipped)
**Scope:** 20 agents.
**Mission:** Cross-cutting consistency and integration between subsystems.
**Evidence:** Partial — some branches merged.

### Wave 4 (Complete)
**Scope:** 20 agents in `WAVE4_AGENTS/`.
**Mission:** Surgery director hardening, benchmark CPU/GPU parity, multi-zoo training expansion.
**Evidence:** `audits/wave4_master.md`, branches in `WAVE4_AGENTS/`.

### Wave 5 (PENDING — Next to Execute)
**Scope:** 20 agents (`W5-1` to `W5-20`).
**Mission:** Verification, integration, cross-cutting audit, and release engineering for `v2.4.0`.
**Key Rule:** Wave 5 does NOT fix bugs inline. It VERIFIES and REPORTS.
**Files:**
- Master spec: `.agent/WAVE5_EXECUTE.md`
- Agent specs: `.agent/WAVE5_AGENTS/W5-{1..20}.md`
- Progress: `.agent/WAVE5_PROGRESS.json`

**Wave 5 Phases:**
| Phase | Agents | Focus |
|---|---|---|
| Phase 1 | W5-1, W5-2, W5-3 | Forensic audit of Waves 1–4 |
| Phase 2 | W5-4 to W5-8 | End-to-end integration tests |
| Phase 3 | W5-9 to W5-13 | Cross-cutting quality (types, logging, errors, docs) |
| Phase 4 | W5-14 to W5-18 | Release engineering (wheel, migration guide, security) |
| Phase 5 | W5-19, W5-20 | Baseline capture + master release synthesis |

### Wave 6 (In Progress — 20/26 Complete)
**Scope:** 26 agents (`W6-1` to `W6-26`) remediating `examples/`.
**Mission:** Make all 35 example scripts honest, tested, and passing with real library APIs.
**Files:**
- Master spec: `.agent/WAVE6_EXECUTE.md`
- Agent specs: `.agent/WAVE6_AGENTS/W6-{1..26}.md`
- Progress: `.agent/WAVE6_PROGRESS.json`

**Current Status (as of 2026-11-15):**
- ✅ Complete: W6-1, W6-2, W6-3, W6-4, W6-5, W6-6, W6-7, W6-8, W6-9, W6-10, W6-11, W6-12, W6-13, W6-14, W6-21, W6-22, W6-23, W6-24, W6-25, W6-26
- 🔄 Pending: W6-15, W6-16, W6-17, W6-18, W6-19, W6-20

---

## 6. CROSS-FILE REMEDIATION MANDATE (READ CAREFULLY)

This is not optional. Every agent must follow this.

### 6.1 — Agents Are NOT Spectators
When an agent finds that a **related file** outside its primary ownership list is broken, incomplete, or causes tests to fail, the agent is **explicitly authorized and obligated to fix that related file**.

Examples of "related file" that MUST be fixed:
- An import that the owned file depends on is broken → fix the import or the imported module.
- A helper, adapter, bridge, or decorator used by the owned file doesn't work → fix it.
- A test that tests the owned file fails because a downstream module has a bug → trace and fix.
- A config schema referenced by the owned file is missing a required field → add the field.

### 6.2 — Boundaries
Agents must NOT touch:
- Files owned by another currently-running agent (check `.agent/WAVE5_PROGRESS.json` or `.agent/WAVE6_PROGRESS.json`).
- Files in `_legacy/` or `legacy_scripts/` without a migration test.
- `neuralops/_version.py` unless explicitly assigned to do so.
- `AGENTS.md` itself.

### 6.3 — Document Every Related-File Fix
In the agent's audit report, include a dedicated section:
```markdown
## Remediated Related Files
| File | Reason Touched | Change Summary |
|---|---|---|
| `neuralops/foo/bar.py` | Used by owned file, was missing method | Added `baz()` with docstring |
```

---

## 7. GIT DISCIPLINE — NON-NEGOTIABLE

### 7.1 — Branch Per Agent
Every subagent MUST work on its own branch:
```bash
git checkout -b wave5/W5-<N>     # for Wave 5 agents
git checkout -b wave6/W6-<N>     # for Wave 6 agents
```
**Forbidden:** Any direct commit to `main` by a subagent.

### 7.2 — Commit Format
```
<scope>(wave<X>/<AGENT_ID>): <imperative summary, <=72 chars>

<optional body: what changed and why>

Refs: #<issue if applicable>
Tests: <test files added or modified>
```

Examples:
```
audit(wave5/W5-1): verify Wave 1 & 2 forensic lineage, write evidence files
fix(wave6/W6-13): eliminate hardcoded pass_fail in examples 18 & 19
release(wave5/W5-14): author v2.3→v2.4 migration guide
```

### 7.3 — Merge Protocol (Orchestrator Only)
Only the orchestrating agent merges branches into `main`:
```bash
git checkout main
git merge --no-ff wave5/W5-<N> -m "merge(wave5): integrate W5-<N> — <description>"
pytest tests/ -x -q -m "not hub and not gpu"    # must be green
ruff check neuralops/ tests/                     # must be clean
```
After every merge:
1. Update `.agent/WAVE5_PROGRESS.json` or `.agent/WAVE6_PROGRESS.json` → `status: "complete"`.
2. Run the baseline suite.
3. Only then proceed to the next agent.

### 7.4 — Forbidden Actions
- ❌ `git add .` — always use specific files
- ❌ `git push --force`
- ❌ Committing `.pt`, `.safetensors`, `.parquet`, or any artifact file
- ❌ Squashing another agent's commit history
- ❌ Committing to `main` directly (subagents)

---

## 8. TESTING STANDARDS — WHAT EVERY AGENT MUST FOLLOW

### 8.1 — Test Pyramid
```
tests/unit/        → no GPU, no network, <100ms each
tests/contract/    → schema and API contract validation
tests/integration/ → end-to-end pipeline tests (synthetic models)
tests/examples/    → subprocess tests for examples/NN_*.py
tests/hardware/    → @pytest.mark.gpu only
```

### 8.2 — Every Example Must Have a Test
If you create or modify `examples/NN_*.py`, you MUST create `tests/examples/test_NN.py`:
```python
import subprocess, sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "NN_name.py"

def test_NN_runs_clean():
    r = subprocess.run([sys.executable, str(EXAMPLE)],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"Exit {r.returncode}:\n{r.stderr}"
```

### 8.3 — Example Honesty Laws (Wave 6 / All Future Waves)
```python
# ❌ FORBIDDEN in examples/
pass_fail("Overall Status", True, "...")       # hardcoded PASS
torch.randn(...)   # for fabricating "real" evaluation data (not inputs)
try:
    import something
except ImportError:
    flag = False   # silent skip

# ✅ CORRECT
all_passed = p1 and p2 and p3
pass_fail("Overall Status", all_passed, "passed" if all_passed else "FAILED")
return 0 if all_passed else 1
```

### 8.4 — Required Test Commands After Every Change
```bash
# Always run before committing:
pytest tests/unit tests/contract -x -q              # unit baseline
ruff check neuralops/ tests/ examples/              # lint
python examples/NN_*.py                             # example runs clean
pytest tests/examples/test_NN.py -v                # example test passes
```

---

## 9. CODE QUALITY STANDARDS

### 9.1 — Type Hints
Every public function must have full type hints:
```python
def run_benchmark(config: BenchmarkConfig, device: str) -> CanonicalRecord: ...
```

### 9.2 — Docstrings (Google Style)
```python
def run_benchmark(config: BenchmarkConfig, device: str) -> CanonicalRecord:
    """Run a single benchmark and return the canonical result record.

    Args:
        config: Validated benchmark configuration.
        device: Target device string ('cpu', 'cuda', 'cuda:0').

    Returns:
        CanonicalRecord with status, metrics, and telemetry.

    Raises:
        BenchmarkTimeoutError: If the benchmark exceeds config.timeout_s.
    """
```

### 9.3 — Logging (No Naked print() in Library Code)
```python
# ✅ CORRECT (library code)
import logging
logger = logging.getLogger(__name__)
logger.info("Running benchmark: %s on %s", name, device)

# ❌ FORBIDDEN in library code
print(f"Running benchmark: {name} on {device}")
```
`print()` is allowed only in: `neuralops/cli/**`, `scripts/**`, `tests/**`, and `examples/**`.

### 9.4 — File Size Limits
- Soft limit: 500 lines per file.
- Hard limit: 800 lines. Split if exceeded.
- Function hard limit: 100 lines. Extract helpers if exceeded.

---

## 10. WHAT TO DO WHEN BLOCKED

If any agent cannot complete its task:

1. **Never silently skip.** Report BLOCKED.
2. Commit what was done so far:
   ```bash
   git add <specific-files>
   git commit -m "wip(wave5/W5-N): partial — blocked on <exact reason>"
   git push -u origin wave5/W5-N
   ```
3. Report to the orchestrator with:
   - What was completed
   - What is blocking (exact error or missing dependency)
   - Suggested resolution
4. Add any discovered bug to `issues/wave6_hotfix_backlog.md`.
5. Mark the failing test `@pytest.mark.xfail(strict=False, reason="<issue>")`.

---

## 11. WHAT EVERY AGENT MUST PRODUCE

Regardless of wave or task, every agent's deliverables:

| Deliverable | Required? | Location |
|---|---|---|
| Owned output files | ✅ Always | As specified in the agent's spec §1.1 |
| Evidence directory | ✅ Always | `audits/wave5/w5_<N>_evidence/` |
| Test results log | ✅ Always | Inside evidence directory |
| Related-file fixes | ✅ When needed | Documented in audit report |
| Bug reports | ✅ When found | `issues/wave6_hotfix_backlog.md` |
| Progress update | ✅ Always | `.agent/WAVE5_PROGRESS.json` or `.agent/WAVE6_PROGRESS.json` |
| Formal report `.md` | ✅ Always | As specified in the agent's spec |
| COMPLETE message | ✅ Always | To orchestrator on completion |

The COMPLETE message format:
```
COMPLETE: <AGENT_ID> at <git SHA>
Branch: wave5/<AGENT_ID>
Owned files produced: <list>
Related files fixed: <list or "none">
Tests run: <pytest command and results>
Findings: <any bugs or issues logged in backlog>
```

---

## 12. CURRENT PROJECT STATE (2026-11-15)

```
Version:    2.4.0 (neuralops/_version.py ✅)
Git branch: main
Last commit: e1714d3bcb — fix(wave6/W6-5): honest rewrite examples 05 & 06

Wave 6:  20 / 26 agents COMPLETE
         Pending: W6-15, W6-16, W6-17, W6-18, W6-19, W6-20
         (Examples 20, consistency audit, readme, check script, CI, verdict)

Wave 5:  0 / 20 agents COMPLETE  ← NEXT WAVE TO EXECUTE
         All 20 agent specs fully authored and ready in .agent/WAVE5_AGENTS/

Release:  NOT YET TAGGED
          v2.4.0 tag will be created after Wave 5 W5-20 passes all gates.
```

---

## 13. DOCUMENT READING ORDER FOR EVERY AGENT

Before starting any task, every agent MUST read in this exact order:

1. ✅ **THIS FILE** — `PROJECT_MASTER_TRUTH.md` (you are here)
2. ✅ **`AGENTS.md`** — the hard code rules (19,294 bytes of non-negotiable rules)
3. ✅ **Your wave's execute file** — `WAVE5_EXECUTE.md` or `WAVE6_EXECUTE.md`
4. ✅ **Your agent's spec file** — `.agent/WAVE5_AGENTS/W5-<N>.md`
5. ✅ **Your owned files** — read top to bottom before touching a single byte
6. ✅ **Relevant tests** — the tests that cover your owned files
7. ✅ **Progress file** — check what other agents in your wave have already done

---

*This file is maintained by the orchestrating agent. Update Section 12 after each wave completes.*
