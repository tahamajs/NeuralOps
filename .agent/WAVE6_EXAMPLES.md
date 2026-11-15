# NEURALOPS WAVE 6 — EXAMPLES REMEDIATION (20 AGENTS)

**Document version:** 1.0
**Effective date:** 2026-11-15
**Mission owner:** Human coordinator
**Executor:** You (the orchestrating agent)
**Scope:** Every file under `examples/` and `examples/_shared/`, plus
related tests, docs, and CI scripts.

---

## 0. WHAT WAVE 6 IS AND WHY IT EXISTS

`examples/` contains 20 Python demonstration files (`01_*.py` through
`20_*.py`) plus a shared utility package (`examples/_shared/`). These
examples are supposed to demonstrate the NeuralOps library to users.

They have been audited. The audit revealed:

1. **Fake evaluation data.** `20_long_context_niah_eval.py` uses
   `torch.randn` to fabricate "attention distributions". It has no model,
   no tokenizer, no NIAH. `08_grpo_reasoning.py` hardcodes candidate
   response strings and uses `torch.ones()` as target labels.

2. **Hardcoded PASS.** Almost every example ends with
   `pass_fail("Overall Example N Status", True, "...")`. Even when every
   sub-check fails, the example exits with code 0. This makes "the example
   runs" meaningless.

3. **Reimplementation of library features.** `04_knowledge_distillation.py`
   and `13_six_pillar_knowledge_distillation.py` reimplement the 6-pillar
   KD loss inline instead of using `neuralops.trainer.distillation`.
   `10_advanced_compression_zoo.py` imports
   `HessianNullSpaceSVDContinuation` but never calls it.

4. **Fragile path assumptions.** Many examples use paths like
   `model.transformer.distilbert.transformer.layer[0].attention._wrapped_attention`.
   Any refactor of the model class breaks the example silently.

5. **Silent skips hiding missing APIs.** Examples wrap imports in
   `try/except ImportError` and set a flag. If the API doesn't exist, the
   example prints a comment and exits 0. The user never learns that the
   example doesn't demonstrate what it claims.

6. **Misleading function names.** `prune_mlp_intermediate` claims to be
   "ADMM" but is L2-norm channel pruning. `compress_hnc_svd` is plain SVD
   with an unused import.

7. **Duplicate examples.** `04` and `13` both claim "6-pillar KD" with
   different implementations. One must be removed.

**Wave 6 is a remediation wave, not a feature wave.** Its mission is:

- **Verify** each example actually does what its docstring claims.
- **Fix** every example that can be honestly fixed using existing library
  APIs and real data.
- **Delete** (or convert to `NotImplementedError`) every example that
  cannot be honestly demonstrated with the current library.
- **Report** every finding truthfully in `docs/examples_audit.md`.

**A deleted example is better than a fake example.** If the library does
not yet provide the API an example needs, that example is a lie to users.
Wave 6's job is to stop that lie, not to maintain the illusion.

---

## 1. THE DEFINITION OF DONE

Wave 6 is complete when ALL of the following are true:

### 1.1 — Every remaining example is honest
- [ ] For every `examples/NN_*.py` that still exists after Wave 6, running
      `python examples/NN_*.py` completes with exit code 0 on CPU.
- [ ] No file under `examples/` (excluding `tests/`) contains
      `torch.randn` for the purpose of fabricating "real" data (NIAH,
      serving, KD outputs, etc.). Synthetic **inputs** to a kernel
      benchmark are allowed but must be clearly labeled in the docstring.
- [ ] No file under `examples/` contains `pass_fail(..., True, ...)` for
      the final overall status. The overall status MUST be computed from
      the actual sub-check results.
- [ ] No file under `examples/` contains `try/except ImportError` that
      silently skips a missing library API. Every failed import must
      produce a `RuntimeError` with a clear message, or the example must
      be deleted.
- [ ] No file under `examples/` reimplements a feature that exists in
      `neuralops/`. If a feature doesn't exist, raise `NotImplementedError`
      rather than creating an inline implementation.

### 1.2 — Every remaining example has a test
- [ ] For every `examples/NN_*.py` that still exists, there is a
      corresponding `tests/examples/test_NN.py` that runs the example in
      a subprocess and asserts exit code 0.
- [ ] `pytest tests/examples/ -v` passes.

### 1.3 — Deletions are documented
- [ ] `docs/examples_audit.md` exists and lists:
      - For each example: KEPT / REWRITTEN / DELETED / CONVERTED
      - For each DELETED: the reason
      - For each REWRITTEN: what was fake and what replaced it
      - For each CONVERTED: what `NotImplementedError` now replaces the
        fake code

### 1.4 — The shared utilities are real
- [ ] `examples/_shared/` utilities have their own tests in
      `tests/examples/test_shared.py`.
- [ ] `_shared.TinyClassifier`, `_shared.pass_fail`, `_shared.assert_parity`,
      `_shared.measure_latency`, `_shared.tiny_inputs` all behave as
      their docstrings claim (verified by tests).

### 1.5 — CI enforces the standard
- [ ] `scripts/check_examples_fake.sh` exists and returns exit code 0
      against a clean tree.
- [ ] `.github/workflows/examples.yml` runs the fake-detector on every PR
      and fails the CI if any fake pattern is reintroduced.

### 1.6 — Documentation
- [ ] `examples/README.md` explains what each example demonstrates, what
      hardware it needs, and what dependencies it requires.
- [ ] No claim in `examples/README.md` is unsupported by the actual code.

### 1.7 — Cross-cutting consistency (audit only)
- [ ] `audits/wave6/consistency.md` documents cross-example inconsistencies
      (e.g. example A uses `pass_fail(True, ...)` while example B computes
      it correctly). Fixes are reported, not necessarily applied by the
      auditor.

---

## 2. PRE-FLIGHT (MANDATORY)

Before spawning any agent, run these checks. STOP if any fails.

### 2.1 — Repository state

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout main
git pull --ff-only origin main

test -z "$(git status --porcelain)" || (echo "DIRTY TREE"; exit 1)
```

### 2.2 — Baseline test suite

```bash
pytest tests/unit tests/contract -x -q || (echo "BASELINE RED"; exit 1)
```

If `tests/unit` doesn't exist, use `pytest tests/ -x -q -m "not hub and not gpu"`.

### 2.3 — Enumerate current examples

```bash
echo "=== Examples present ==="
ls examples/[0-9][0-9]_*.py 2>/dev/null | sort
echo "=== Count ==="
ls examples/[0-9][0-9]_*.py 2>/dev/null | wc -l
```

Expected: 20 files.

### 2.4 — Create scaffolding

```bash
mkdir -p .agent/WAVE6_AGENTS
mkdir -p audits/wave6
mkdir -p tests/examples
mkdir -p scripts
```

### 2.5 — Create progress file

Write `.agent/WAVE6_PROGRESS.json`:

```json
{
  "wave": 6,
  "started_at": "2026-11-15T00:00:00Z",
  "last_updated": "2026-11-15T00:00:00Z",
  "phase": 1,
  "agents": {
    "W6-1":  {"status": "pending", "branch": "wave6/W6-1",  "worktree": null},
    "W6-2":  {"status": "pending", "branch": "wave6/W6-2",  "worktree": null},
    "W6-3":  {"status": "pending", "branch": "wave6/W6-3",  "worktree": null},
    "W6-4":  {"status": "pending", "branch": "wave6/W6-4",  "worktree": null},
    "W6-5":  {"status": "pending", "branch": "wave6/W6-5",  "worktree": null},
    "W6-6":  {"status": "pending", "branch": "wave6/W6-6",  "worktree": null},
    "W6-7":  {"status": "pending", "branch": "wave6/W6-7",  "worktree": null},
    "W6-8":  {"status": "pending", "branch": "wave6/W6-8",  "worktree": null},
    "W6-9":  {"status": "pending", "branch": "wave6/W6-9",  "worktree": null},
    "W6-10": {"status": "pending", "branch": "wave6/W6-10", "worktree": null},
    "W6-11": {"status": "pending", "branch": "wave6/W6-11", "worktree": null},
    "W6-12": {"status": "pending", "branch": "wave6/W6-12", "worktree": null},
    "W6-13": {"status": "pending", "branch": "wave6/W6-13", "worktree": null},
    "W6-14": {"status": "pending", "branch": "wave6/W6-14", "worktree": null},
    "W6-15": {"status": "pending", "branch": "wave6/W6-15", "worktree": null},
    "W6-16": {"status": "pending", "branch": "wave6/W6-16", "worktree": null},
    "W6-17": {"status": "pending", "branch": "wave6/W6-17", "worktree": null},
    "W6-18": {"status": "pending", "branch": "wave6/W6-18", "worktree": null},
    "W6-19": {"status": "pending", "branch": "wave6/W6-19", "worktree": null},
    "W6-20": {"status": "pending", "branch": "wave6/W6-20", "worktree": null}
  }
}
```

Once Section 2 passes, proceed to Section 3.

---

## 3. THE GIT PROTOCOL

Identical to previous waves. This section is short because it is repeated
verbatim in the spawn template.

### 3.1 — Worktree creation (orchestrator does this once)

```bash
cd "$(git rev-parse --show-toplevel)/.."
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
REPO_ABS=$(pwd)

AGENTS=(W6-1 W6-2 W6-3 W6-4 W6-5 W6-6 W6-7 W6-8 W6-9 W6-10
        W6-11 W6-12 W6-13 W6-14 W6-15 W6-16 W6-17 W6-18 W6-19 W6-20)

for AGENT in "${AGENTS[@]}"; do
    BRANCH="wave6/${AGENT}"
    WORKTREE="${REPO_NAME}_${AGENT//-/_}"
    WORKTREE_ABS="${REPO_ABS}/${WORKTREE}"
    if git -C "$REPO_NAME" rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
        echo "[SKIP] $AGENT exists"
        continue
    fi
    git -C "$REPO_NAME" branch "$BRANCH" main || exit 1
    git -C "$REPO_NAME" worktree add "$WORKTREE_ABS" "$BRANCH" || exit 1
    echo "[OK] $AGENT at $WORKTREE_ABS"
    # Update progress file
    python3 -c "
import json
p = json.load(open('.agent/WAVE6_PROGRESS.json'))
p['agents']['${AGENT}']['worktree'] = '${WORKTREE_ABS}'
json.dump(p, open('.agent/WAVE6_PROGRESS.json', 'w'), indent=2)
"
done
```

Verify: `ls -d ../${REPO_NAME}_W6_* | wc -l` must be 20.

### 3.2 — Commit protocol

Every agent commits after each file is fixed. Format:

```
fix(wave6/<AGENT>): <short description>

Body:
- Bug: <what was fake/broken>
- Fix: <what replaced it>
- Evidence: <exact command output, first 5 lines>
- Test: <test file::test_name>

Files: <changed files>
```

Types: `fix`, `test`, `docs`, `chore`, `delete`, `wip`.

The `delete` type is used when an example is fully removed:
```
delete(wave6/<AGENT>): remove examples/NN_*.py (fake NIAH)

Reason: The example claimed to demonstrate NIAH but used torch.randn
to fabricate attention distributions. No real model, no real tokenizer.
Documented in docs/examples_audit.md.
```

Commit frequency: every 30-60 minutes minimum. Never hold more than
2000 lines of diff before committing.

### 3.3 — Push protocol

```bash
git push -u origin wave6/<AGENT>
```

No force-push. If push fails, STOP and report.

### 3.4 — Report protocol

On success:
```
COMPLETE: <AGENT> at <SHA>
Branch: wave6/<AGENT>
Examples touched: <list of NN>
Status per example: FIXED / DELETED / CONVERTED
Tests added: <count>
Findings logged for Wave 7: <count>
```

On failure:
```
BLOCKED: <AGENT> at <SHA>: <reason>
Branch: wave6/<AGENT>
What was done: <list>
What is blocking: <specific>
Suggestion: <what a human should do>
```

---

## 4. THE SPAWN TEMPLATE (VERBATIM)

```
═══════════════════════════════════════════════════════════════════════
SUBTASK: <AGENT_ID>
═══════════════════════════════════════════════════════════════════════

You are Subagent <AGENT_ID> in Wave 6.

Working directory: <WORKTREE_ABS>
Git branch: wave6/<AGENT_ID>
Task file: .agent/WAVE6_AGENTS/<AGENT_ID>.md

═══════════════════════════════════════════════════════════════════════
BEFORE ANY WORK (steps 1-6)
═══════════════════════════════════════════════════════════════════════

STEP 1. Change to your worktree:
    cd <WORKTREE_ABS>

STEP 2. Confirm branch:
    git rev-parse --abbrev-ref HEAD
    # MUST output "wave6/<AGENT_ID>". If not, STOP and report.

STEP 3. Confirm worktree is clean:
    git status --porcelain
    # MUST be empty. If not, STOP and report.

STEP 4. Read reference documents:
    cat AGENTS.md
    cat .agent/WAVE6_EXECUTE.md
    cat .agent/WAVE6_AGENTS/<AGENT_ID>.md

STEP 5. Read the examples assigned to you END TO END. Read the docstring,
        read every function, read the `main()` block. Note where the code
        doesn't match the docstring.

STEP 6. Run the baseline audit on your examples:
    for f in <your assigned examples>; do
        echo "=== $f ==="
        echo "-- torch.randn:"
        grep -n "torch.randn" "$f" || true
        echo "-- pass_fail True:"
        grep -n "pass_fail.*True" "$f" || true
        echo "-- except ImportError:"
        grep -n "except ImportError" "$f" || true
        echo "-- duplicate pillar functions:"
        grep -n "^def pillar\|^def loss_\|^def compute_.*attn" "$f" || true
        echo "-- fragile paths:"
        grep -n "\.transformer\.\|\.layer\[\|_wrapped_attention" "$f" || true
        echo "-- run time:"
        timeout 180 python "$f" 2>&1 | tail -15
        echo "exit: $?"
    done

        Record every finding. This is your baseline.

═══════════════════════════════════════════════════════════════════════
YOUR MISSION
═══════════════════════════════════════════════════════════════════════

Your task file (.agent/WAVE6_AGENTS/<AGENT_ID>.md) specifies:
  - The exact files you own (NO OTHERS)
  - The specific bugs to fix in each file
  - The specific tests to add
  - The acceptance criteria

You MUST follow the file ownership STRICTLY. If you find a bug in a file
that isn't yours, log it in .agent/WAVE6_AGENTS/<AGENT_ID>_CROSS.md.
Do NOT fix it.

═══════════════════════════════════════════════════════════════════════
THE FOUR LAWS OF WAVE 6 (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════════════

LAW 1 — NO FAKE DATA FOR REAL EVALUATIONS
If an example claims to demonstrate NIAH, serving, KD, or model
evaluation, it MUST use a real model and real input. If the library
provides the APIs, use them. If it doesn't, raise NotImplementedError.

LAW 2 — NO HARDCODED PASS
`pass_fail("Overall Example X Status", True, "...")` is FORBIDDEN.
The overall status MUST be computed:
    all_passed = p1 and p2 and p3
    pass_fail("Overall Status", all_passed, "..." if all_passed else "...")
    return 0 if all_passed else 1

LAW 3 — NO SILENT SKIP ON MISSING APIS
If an import fails, the example must FAIL LOUDLY:
    raise RuntimeError(f"This example requires <module>, which is not
                        available: {e}. Either implement the module or
                        delete this example.")
Silent try/except ImportError that sets a flag and skips is FORBIDDEN.

LAW 4 — NO REIMPLEMENTATION OF LIBRARY FEATURES
If `neuralops.trainer.distillation.MultiLevelKD` exists, the KD example
must use it. If it doesn't exist, the example must raise
NotImplementedError. Defining `def pillar1_logit_kl()` inline and
claiming "this demonstrates the library's 6-pillar KD" is FORBIDDEN.

═══════════════════════════════════════════════════════════════════════
PER-FILE WORKFLOW
═══════════════════════════════════════════════════════════════════════

For each example you own:

STEP A — CLASSIFY
Read the file and answer:
  - What does the docstring claim this example demonstrates?
  - What does the code actually do?
  - Do the two match? If not, this is a "lie" example.
  - Can it be fixed with existing library APIs?
    → YES: fix it
    → NO: delete it (or raise NotImplementedError)
  - Is there a real library feature it should demonstrate?

STEP B — DECIDE
You must choose ONE of these verdicts for each example:

  (1) REWRITE — the example can be fixed using real library APIs and
      real data. Rewrite it.

  (2) DELETE — the example cannot be honestly demonstrated with the
      current library. Delete it. Document the reason.

  (3) CONVERT — the example should be kept as a "placeholder" but
      cannot be honestly implemented. Replace its body with
      `raise NotImplementedError(...)` and a clear docstring.

Document your choice with the reasoning in the commit message.

STEP C — IMPLEMENT
For REWRITE:
  1. Use `examples/_shared/` utilities wherever possible
  2. Use real library APIs (import from neuralops)
  3. Use real data (real model, real tokenizer, or real library call)
  4. Compute all_passed from real sub-checks
  5. Return `0 if all_passed else 1`

For DELETE:
  1. `git rm examples/NN_*.py`
  2. `git rm tests/examples/test_NN.py` (if exists)
  3. Document in docs/examples_audit.md (coordinate with W6-16)
  4. Commit with `delete(wave6/<AGENT_ID>)` prefix

For CONVERT:
  1. Keep the file
  2. Replace `main()` body with:
     ```python
     def main() -> int:
         section("Example NN: <title>")
         kv("Status", "NOT IMPLEMENTED")
         raise NotImplementedError(
             "<specific reason>. This example requires <feature>, which is "
             "not yet available in NeuralOps. See <related alternative> for "
             "the closest available demonstration."
         )
     ```
  3. Add a test that asserts the NotImplementedError is raised.

STEP D — TEST
For every example you touch:
  1. Add or update `tests/examples/test_NN.py`:
     ```python
     import subprocess
     import sys
     from pathlib import Path
     import pytest

     EXAMPLE = Path(__file__).parent.parent.parent / "examples" / "NN_*.py"

     def test_example_runs_clean():
         result = subprocess.run(
             [sys.executable, str(EXAMPLE)],
             capture_output=True, text=True, timeout=180,
         )
         assert result.returncode == 0, \
             f"Example NN failed with exit {result.returncode}:\n{result.stderr}"

     def test_example_has_no_fake_patterns():
         source = EXAMPLE.read_text()
         assert "torch.randn" not in source, \
             "Example uses torch.randn for data — must use real data"
         assert "pass_fail(\"Overall" not in source or \
                "pass_fail(\"Overall Example NN Status\", True," not in source, \
             "Example hardcodes PASS for overall status"
     ```

  2. Run:
     ```bash
     python examples/NN_*.py
     echo "Exit: $?"  # MUST be 0
     pytest tests/examples/test_NN.py -v
     # MUST pass
     ```

STEP E — COMMIT
```bash
git add examples/NN_*.py tests/examples/test_NN.py
git commit -m "fix(wave6/<AGENT_ID>): NN <verb> <what>

Bug: <what was fake>
Fix: <what replaced it>
Evidence: <python examples/NN_*.py exited 0; pytest test_NN.py passed 2/2>
Files: examples/NN_*.py, tests/examples/test_NN.py"
```

```bash
git push -u origin wave6/<AGENT_ID>
```

═══════════════════════════════════════════════════════════════════════
HARD RULES (violation = automatic rejection)
═══════════════════════════════════════════════════════════════════════

1. NEVER touch a file outside your Owned Paths.
2. NEVER keep `torch.randn` for evaluation data.
3. NEVER keep `pass_fail(..., True, ...)` for overall status.
4. NEVER keep `try/except ImportError` that silently skips.
5. NEVER reimplement a library feature that exists.
6. NEVER claim "complete" without running `python examples/NN.py` and
   showing exit 0.
7. NEVER delete an example without documenting the reason in your
   commit message AND in `.agent/WAVE6_AGENTS/<AGENT_ID>_DELETIONS.md`.
8. NEVER use `git add .` — always specific files.
9. NEVER force-push.
10. NEVER return to the orchestrator without a COMPLETE or BLOCKED report.

═══════════════════════════════════════════════════════════════════════
IF BLOCKED (>20 minutes on one item)
═══════════════════════════════════════════════════════════════════════

```bash
git add <files>
git commit -m "wip(wave6/<AGENT_ID>): blocked on <reason>"
git push -u origin wave6/<AGENT_ID>
```

Report BLOCKED per Section 3.4 of WAVE6_EXECUTE.md.

═══════════════════════════════════════════════════════════════════════
IF SERVER RESTARTS
═══════════════════════════════════════════════════════════════════════

1. Read .agent/WAVE6_PROGRESS.json
2. git log wave6/<AGENT_ID> -5
3. Resume from next uncommitted file
4. Do NOT start over

═══════════════════════════════════════════════════════════════════════
BEGIN NOW.
═══════════════════════════════════════════════════════════════════════
```

---

## 5. THE 20 AGENT SPECIFICATIONS

### PHASE 1 — Foundation (2 agents, must complete before Phase 2)

#### W6-1 — `examples/_shared/` Foundation

**Owned Paths:**
- `examples/_shared/**` (all files)
- `tests/examples/test_shared.py` (create)

**Mission:** Verify that every utility imported by the examples is real,
tested, and behaves as its docstring claims.

**Files to inspect (or create if missing):**
- `examples/_shared/__init__.py`
- `examples/_shared/tinymodel.py` (if exists)
- `examples/_shared/output.py` (if exists)
- `examples/_shared/bench.py` (if exists)

**Utilities to verify:**

1. `TinyClassifier` — must be a real `nn.Module` with:
   - `forward()` method
   - Trainable parameters
   - `count_parameters()` returns `sum(p.numel() for p in self.parameters())`
   - `count_trainable_parameters()` returns
     `sum(p.numel() for p in self.parameters() if p.requires_grad)`
   - Accepts `n_layers` and `seed` kwargs
   - Uses a real HuggingFace model or a real hand-written transformer

2. `tiny_inputs()` — return real tensors of documented shape. If it uses
   `torch.randn`, that's OK for inputs but document it in the docstring.

3. `pass_fail(name, condition, detail)` — must print "PASS" if condition
   is True, "FAIL" if False. Must accept `condition` as `bool`, not just
   truthy.

4. `kv(key, value)` — must print formatted key-value.

5. `section(title)` — must print section separator.

6. `table(rows, headers)` — must print formatted table.

7. `assert_parity(actual, expected, atol, label)` — must raise
   `AssertionError` if `(actual - expected).abs().max() > atol`.

8. `measure_latency(fn, warmup, iters)` — must actually run `fn` and
   return real elapsed time.

9. `get_backend_preference()` — return `"cuda"` / `"mps"` / `"cpu"`
   based on `torch.cuda.is_available()` etc.

10. `gpu_name()` — return `torch.cuda.get_device_name(0)` or `"CPU"`.

11. `toy_dataloader()`, `toy_calibration_loader()` — must return real
    `DataLoader` with actual batches, not `torch.randn`.

12. `synthetic_tokens()` — OK to use `torch.randint` for token IDs
    (that's what tokens are).

13. `skip_if_missing(package, message)` — return `True` if package is
    missing, `False` otherwise. Note: if you use this in examples, the
    example must still pass — skipping is a valid outcome but the overall
    status must reflect the skip.

**Specific bugs to fix:**

- If `pass_fail` accepts any truthy value (e.g. `pass_fail("X", "non-empty string", ...)`) and prints PASS, that's a bug. It must enforce `isinstance(condition, bool)`.
- If `assert_parity` doesn't actually compare (e.g. always returns True), rewrite it.
- If `TinyClassifier` is not a real model, either write a real one or delete `TinyClassifier` and update all examples to use a real HF model.

**Tests to add** (`tests/examples/test_shared.py`):

```python
import io
import sys
from contextlib import redirect_stdout
import pytest
import torch
from examples._shared import (
    TinyClassifier, pass_fail, kv, section, table,
    assert_parity, measure_latency, get_backend_preference, gpu_name,
)


def test_tiny_classifier_is_real_module():
    m = TinyClassifier()
    assert isinstance(m, torch.nn.Module)
    assert m.count_parameters() > 0
    assert m.count_trainable_parameters() > 0
    x = torch.randint(0, 1000, (2, 8))
    out = m(input_ids=x)
    assert out is not None


def test_pass_fail_prints_pass():
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass_fail("test_check", True, "detail")
    assert "PASS" in buf.getvalue()


def test_pass_fail_prints_fail():
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass_fail("test_check", False, "detail")
    assert "FAIL" in buf.getvalue()


def test_assert_parity_catches_mismatch():
    a = torch.zeros(3)
    b = torch.ones(3)
    with pytest.raises(AssertionError):
        assert_parity(a, b, atol=1e-3, label="test")


def test_assert_parity_passes_on_match():
    a = torch.randn(3)
    b = a.clone()
    assert_parity(a, b, atol=1e-3, label="test")


def test_measure_latency_returns_positive():
    def fn():
        return torch.randn(100, 100) @ torch.randn(100, 100)
    lat = measure_latency(fn, warmup=1, iters=2)
    assert lat > 0.0


def test_get_backend_preference_returns_known_string():
    backend = get_backend_preference()
    assert backend in ("cuda", "mps", "cpu")


def test_gpu_name_returns_string():
    name = gpu_name()
    assert isinstance(name, str)
    assert len(name) > 0
```

**Acceptance criteria:**
- [ ] Every utility listed above is verified by a test
- [ ] Every utility that fails its test is either fixed or removed with
      coordinated updates to all examples that use it
- [ ] `pytest tests/examples/test_shared.py -v` passes (≥8 tests)
- [ ] Commit message includes evidence for each utility

**Note:** If W6-1 changes a utility's signature or behavior, it must
NOTIFY the orchestrator (commit a note to
`.agent/WAVE6_AGENTS/W6-1_NOTICE.md`) so W6-3..W6-14 know to update
their examples. This is the only cross-agent communication channel.

---

#### W6-2 — Baseline Audit Framework

**Owned Paths:**
- `audits/wave6/baseline_audit.md`
- `audits/wave6/baseline_raw.json`

**Mission:** Produce a complete baseline audit of all 20 examples BEFORE
any fixes. This is the "before" snapshot that will be compared against
"after" in W6-20.

**Tasks:**

1. For each example `examples/NN_*.py` (NN = 01..20):

   ```bash
   python -c "
   import re, json
   from pathlib import Path

   path = Path('examples/NN_*.py')
   src = path.read_text()

   findings = {
       'file': str(path),
       'has_torch_randn': 'torch.randn' in src,
       'randn_lines': [i+1 for i,l in enumerate(src.splitlines())
                       if 'torch.randn' in l],
       'has_hardcoded_pass': bool(re.search(
           r'pass_fail\(\"Overall[^\"]*\",\s*True', src)),
       'hardcoded_pass_lines': [i+1 for i,l in enumerate(src.splitlines())
                                if re.search(r'pass_fail\(\"Overall[^\"]*\",\s*True', l)],
       'has_silent_import': 'except ImportError' in src,
       'import_error_lines': [i+1 for i,l in enumerate(src.splitlines())
                              if 'except ImportError' in l],
       'has_duplicate_pillars': bool(re.search(r'^def pillar[0-9]', src, re.M)),
       'pillar_defs': re.findall(r'^def (pillar[0-9]\w*)', src, re.M),
       'fragile_paths': re.findall(r'(\.transformer\.\w+|_wrapped_attention|\.layer\[\d+\])', src),
       'line_count': len(src.splitlines()),
   }
   print(json.dumps(findings, indent=2))
   "
   ```

2. Run each example to see baseline exit code:
   ```bash
   for f in examples/[0-9][0-9]_*.py; do
       echo "=== $f ==="
       timeout 180 python "$f" > /dev/null 2>&1
       echo "exit: $?"
   done
   ```

3. Aggregate into `audits/wave6/baseline_raw.json`:
   ```json
   {
     "timestamp": "...",
     "examples": [ <findings for each> ]
   }
   ```

4. Write `audits/wave6/baseline_audit.md` — human-readable summary:
   - Total examples: 20
   - Examples with `torch.randn` for fake data: N
   - Examples with hardcoded PASS: N
   - Examples with silent imports: N
   - Examples with duplicate library code: N
   - Baseline exit code 0: N/20 (this number should be high — that's
     the problem, since exit 0 doesn't prove anything works)

**Acceptance:**
- [ ] `baseline_raw.json` exists with all 20 examples
- [ ] `baseline_audit.md` summarizes key metrics
- [ ] Report clearly states "Exit code 0 in baseline does NOT indicate
      success; see the audit reasons"

---

### PHASE 2 — Per-Example Fixes (12 agents, run in parallel after Phase 1)

Every agent in this phase follows the same pattern:
1. Run baseline audit (from W6-2) on their assigned examples
2. Classify each: REWRITE / DELETE / CONVERT
3. Implement the fix
4. Add tests
5. Commit + push

The specific bugs for each example are listed below.

---

#### W6-3 — Examples 01, 02

**Owned Paths:**
- `examples/01_attention_surgery.py`
- `examples/02_model_compression.py`
- `tests/examples/test_01.py`
- `tests/examples/test_02.py`

**Example 01 specific bugs:**

1. **`demonstrate_weight_transplantation` is trivially true.** Current:
   ```python
   m_copied = TinyClassifier(seed=100).to(device)
   m_fresh = TinyClassifier(seed=100).to(device)
   neuralops.replace_attention(m_copied, ..., copy_weights=True)
   neuralops.replace_attention(m_fresh, ..., copy_weights=False)
   params_copied = torch.cat([p.flatten() for p in m_copied.parameters()])
   params_fresh = torch.cat([p.flatten() for p in m_fresh.parameters()])
   weights_differ = not torch.equal(params_copied, params_fresh)
   pass_fail(..., weights_differ, ...)
   ```
   This only proves "two models have different weights". It does NOT
   verify that `copy_weights=True` actually preserves the original
   weights.

   **Fix:**
   ```python
   base = TinyClassifier(seed=100).to(device)
   # Find the first attention layer's q_proj BEFORE surgery
   from neuralops.surgery.scanner import UniversalArchitectureScanner
   specs = UniversalArchitectureScanner.scan(base)
   assert specs, "No attention targets found"
   orig_q = specs[0].extracted_qkv[0].clone()  # W_q
   
   surg = neuralops.replace_attention(base, attention="cosformer", copy_weights=True)
   # Find the new attention layer
   new_specs = UniversalArchitectureScanner.scan(surg)
   new_q = new_specs[0].extracted_qkv[0]
   diff = (orig_q - new_q).abs().max().item()
   pass_fail("Weight transplantation preserved q_proj",
             diff < 1e-6, f"max diff = {diff:.2e}")
   ```

2. **Hardcoded PASS at end:**
   ```python
   pass_fail("Overall Example 01 Status", True, "All surgery and hybrid checks passed")
   return 0
   ```
   Fix:
   ```python
   all_passed = p1 and p2 and p3
   pass_fail("Overall Example 01 Status", all_passed,
             "All checks passed" if all_passed else "One or more checks failed")
   return 0 if all_passed else 1
   ```

**Example 02 specific bugs:**

1. **`compute_spectral_metrics` computes metrics on ORIGINAL weights, not
   compressed.** Current:
   ```python
   _, S, _ = torch.linalg.svd(W, full_matrices=False)  # W is original
   # Compute energy retention from S
   ```
   This doesn't test the compressed model at all.

   **Fix:** Compute reconstruction error by comparing output of original
   and compressed layers:
   ```python
   def test_compression_output_parity(original_layer, compressed_layer, rank):
       x = torch.randn(4, original_layer.in_features)
       orig_out = original_layer(x)
       comp_out = compressed_layer(x)
       rel_err = (orig_out - comp_out).norm() / orig_out.norm()
       return rel_err.item()
   ```

2. **Hardcoded PASS:** same fix as above.

**Acceptance criteria:**
- [ ] Both examples can be run: `python examples/01_*.py && python examples/02_*.py`
- [ ] Both examples exit with code `0` (all sub-checks pass) or `1`
      (some fail — then the test should not require it to pass, only to
      complete)
- [ ] `grep "torch.randn" examples/01_*.py examples/02_*.py` returns
      nothing except in weight init
- [ ] `grep "pass_fail.*True" examples/01_*.py examples/02_*.py` returns
      nothing for overall status
- [ ] `tests/examples/test_01.py` and `test_02.py` exist and pass

---

#### W6-4 — Examples 03, 04

**Owned Paths:**
- `examples/03_lora_finetuning.py`
- `examples/04_knowledge_distillation.py`
- `tests/examples/test_03.py`
- `tests/examples/test_04.py`

**Example 03 specific bugs:**

1. **`LoRALinear.from_linear` signature:** verify it accepts
   `(linear_layer, lora_rank=int, freeze_base=bool)`. If not, the
   example crashes. Check the actual signature in
   `neuralops/core/peft/lora.py` and update the call.

2. **Merge parity test:** verify it actually catches a real difference.
   The test should:
   ```python
   model_unmerged = deepcopy(model)
   out1 = model_unmerged(inputs).logits
   for m in model_unmerged.modules():
       if isinstance(m, LoRALinear):
           m.merge()
   out2 = model_unmerged(inputs).logits
   assert (out1 - out2).abs().max() < 1e-5
   ```

**Example 04 specific bugs:**

1. **Reimplements 6-pillar KD.** Current:
   ```python
   def pillar1_logit_kl(...): ...
   def pillar2_hidden_cosine(...): ...
   def pillar3_attention_js(...): ...
   def pillar4_gradient_alignment(...): ...
   def pillar5_fisher_curvature(...): ...
   def pillar6_activation_covariance(...): ...
   ```
   The docstring claims "6-Pillar Knowledge Distillation". But the
   library has `neuralops.trainer.distillation.multi_level_kd.MultiLevelKD`
   (if it exists — verify).

   **Fix options (in order of preference):**
   - (A) Use `MultiLevelKD` from the library. Rewrite the example around
     it.
   - (B) If `MultiLevelKD` doesn't exist, raise `NotImplementedError`:
     ```python
     def main():
         section("Example 04: 6-Pillar Knowledge Distillation")
         raise NotImplementedError(
             "MultiLevelKD is not available in the current NeuralOps build. "
             "This example requires neuralops.trainer.distillation."
             "multi_level_kd.MultiLevelKD. Use examples/13 if it is available."
         )
     ```
   - (C) Delete the example entirely.

2. **`pillar4_gradient_alignment` computes gradients on wrong params.**
   Current:
   ```python
   t_params = [p for p in teacher_model.transformer.classifier.parameters()]
   t_grads = torch.autograd.grad(loss_t, t_params, ...)
   ```
   This computes gradients only on the classifier, not the full model.
   If fixing option (A), this is deleted. If (B), deleted. If (C),
   deleted.

**Acceptance criteria:**
- [ ] Example 03 runs cleanly
- [ ] Example 04 is either (A) using real MultiLevelKD, (B) raising
      NotImplementedError, or (C) deleted
- [ ] No `def pillar[0-9]` remains in example 04 (if it uses the library)
- [ ] Tests for each example
- [ ] Both examples exit with meaningful status

---

#### W6-5 — Examples 05, 06

**Owned Paths:**
- `examples/05_fla_style_benchmarking.py`
- `examples/06_export_and_push.py`
- `tests/examples/test_05.py`
- `tests/examples/test_06.py`

**Example 05 specific bugs:**

1. **`measure_vram_if_available`** — verify it returns `None` on CPU
   (not `0.0`). The current code says:
   ```python
   vram_str = f"{base_vram:.1f} MB" if base_vram is not None else "0.0 MB (CPU)"
   ```
   If `measure_vram_if_available()` returns `0.0` on CPU, the else branch
   never fires. Verify.

2. **`compute_flops_estimate`** — verify it returns a real number, not a
   hardcoded constant. If it's hardcoded, either compute it from actual
   model structure or rename to `compute_flops_estimate_heuristic` and
   document it.

3. **Mamba latency exception handling** — the code special-cases Mamba
   (`ratio > 10.0` → PASS_WITH_WARNING). This is dishonest. If Mamba is
   slow on CPU, that's a real finding. Either:
   - Remove the special case and let the test fail
   - Or raise a `NotImplementedError` for Mamba on CPU

**Example 06 specific bugs:**

1. **`export_and_verify_gguf` is a no-op.** Current:
   ```python
   def export_and_verify_gguf(temp_dir):
       if skip_if_missing("gguf", "..."):
           return True, 0, "SKIPPED: gguf missing"
       # When gguf is installed, demonstrate GGUF metadata writing
       return True, 1024, "PASS"
   ```
   This returns "PASS" without doing anything. Fix:
   - Either actually export GGUF if `gguf` is available
   - Or always return "SKIPPED: not implemented in this example"

2. **SafeTensors reload parity** — verify `assert_parity` catches real
   mismatch. If the exported file is the same model, parity is trivial.
   Test it by intentionally corrupting the file first:
   ```python
   # Test:
   import json
   state_dict["transformer.classifier.weight"] += 0.1  # corrupt
   # Then verify assert_parity fails
   ```

**Acceptance criteria:**
- [ ] Example 05 runs, exit 0 or 1 based on real timing
- [ ] Example 06 runs, exit 0 or 1 based on real export success
- [ ] No silent skips that hide missing features
- [ ] No hardcoded PASS
- [ ] Tests for each

---

#### W6-6 — Examples 07, 08

**Owned Paths:**
- `examples/07_custom_attention.py`
- `examples/08_grpo_reasoning.py`
- `tests/examples/test_07.py`
- `tests/examples/test_08.py`

**Example 07 specific bugs:**

1. **`MyCausalLinearAttention` ignores `attention_mask`.** When
   `attention_mask` is provided, masked positions should be zeroed in
   `phi_k`:
   ```python
   if attention_mask is not None:
       mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
       phi_k = phi_k * mask_expanded
   ```
   Current code ignores the mask entirely. This means padding tokens
   contaminate the KV state.

2. **`@neuralops.register_custom_attention("my_causal_linear")`** —
   verify this decorator exists. If it doesn't, the example must raise
   `NotImplementedError` or be converted to use a different registration
   path.

3. **`@COMPRESSOR_REGISTRY.register("my_svd_variant")`** — same.

**Example 08 specific bugs:**

1. **COMPLETELY FAKE.** The current example hardcodes:
   ```python
   candidate_responses = [
       "Let me think. Step 1: Analyze. Answer: Option A is correct.",
       "The problem requires careful deduction. The result is unclear.",
       "Short solution. Answer: 42.",
       "Thinking aloud. The final conclusion appears to be 42 but with caveats.",
   ]
   ```
   These are not generated by any model. The reward is a substring check.

   Then:
   ```python
   target_actions = torch.ones(k_samples, dtype=torch.long, device=device)
   action_log_probs = log_probs.gather(1, target_actions.unsqueeze(1)).squeeze(1)
   surrogate_loss = - (adv_weights * action_log_probs).mean()
   ```
   This is not GRPO. It's a made-up loss on random logits.

   **Fix options:**
   - (A) Use a real model + real tokenizer to generate candidates. The
     model can be `hf-internal-testing/tiny-random-gpt2` (small, works
     offline after first download). Actually generate tokens, decode
     them, score them, compute advantages, do a real policy gradient
     step.
   - (B) Delete the example entirely. Document in
     `docs/examples_audit.md`: "GRPO requires a language model. TinyClassifier
     is not suitable. Deleted."
   - (C) Convert to a pure-math test that computes GRPO advantages without
     claiming to be a training example.

   **Preferred:** (B) or (C).

**Acceptance criteria:**
- [ ] Example 07 respects `attention_mask`
- [ ] Example 08 is either rewritten honestly (A), deleted (B), or
      converted to math test (C)
- [ ] If deleted, documented in `audits/wave6/w6_6_DELETIONS.md`
- [ ] Tests for each remaining example

---

#### W6-7 — Examples 09, 10

**Owned Paths:**
- `examples/09_multi_stage_model_surgery.py`
- `examples/10_advanced_compression_zoo.py`
- `tests/examples/test_09.py`
- `tests/examples/test_10.py`

**Example 09 specific bugs:**

1. **Verify each `surgery.apply(...)` operation exists:**
   ```python
   from neuralops.surgery.registry import SURGERY_REGISTRY
   for op in ["attention.replace", "architecture.reduce_mlp",
              "architecture.drop_layer", "weights.quantize"]:
       assert op in SURGERY_REGISTRY, f"Op '{op}' not registered"
   ```
   If an op is missing, the current code catches the exception and
   continues. That's a lie. Replace with `NotImplementedError`.

2. **`stages_log` collects "Forward Status" but the last item is always
   `True`** because of the hardcoded `pass_fail(..., True, ...)`. Fix.

**Example 10 specific bugs:**

1. **`compress_hnc_svd` imports `HessianNullSpaceSVDContinuation` but
   never uses it.** Current:
   ```python
   from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation
   hnc = HessianNullSpaceSVDContinuation(...)
   # Then... does nothing with hnc
   u, s, vh = torch.linalg.svd(weight, full_matrices=False)
   u_r = u[:, :target_rank]
   # ...
   ```
   This is plain SVD pretending to be HNC-SVD. Either:
   - Call `hnc.compress(weight)` (verify method exists)
   - Rename function to `compress_svd_baseline` and remove HNC claim
   - Delete this method from the example and document why

2. **Same problem may exist for "ADMM"** in other examples — verify all
   function names match their actual implementation.

**Acceptance criteria:**
- [ ] Example 09 only claims ops that actually work
- [ ] Example 10 function names match actual algorithms
- [ ] No unused imports
- [ ] Tests for each

---

#### W6-8 — Examples 11, 12

**Owned Paths:**
- `examples/11_quantization_fp8_int4_mx.py`
- `examples/12_structured_pruning_and_token_shedding.py`
- `tests/examples/test_11.py`
- `tests/examples/test_12.py`

**Example 11 specific bugs:**

1. **FP8 SQNR threshold of 10.0 dB is too permissive.** Real FP8
   quantization achieves 20-40 dB SQNR on typical weights. Change
   threshold to `sqnr_fp8 > 20.0`.

2. **Part 4 modifies the model in place** — after that example ends, the
   model is quantized. Should either reset after or work on a copy.

**Example 12 specific bugs:**

1. **`prune_mlp_intermediate` claims "ADMM" but implements L2 norm
   pruning.** Fix:
   - Either rename to `prune_mlp_by_norm` and update docstring/claims
   - Or implement actual ADMM (would need 20+ iterations of the
     augmented Lagrangian — significant work)
   - Or raise `NotImplementedError` and remove the ADMM claim

2. **`apply_24_sparsity` and `TokenSheddingWrapper`** — verify these
   actually work as claimed (2:4 exact sparsity, sink token protection).
   Test with specific example.

**Acceptance criteria:**
- [ ] Example 11 thresholds are honest (FP8 > 20dB)
- [ ] Example 12 either does real ADMM or doesn't claim ADMM
- [ ] Tests for each

---

#### W6-9 — Examples 13, 14

**Owned Paths:**
- `examples/13_six_pillar_knowledge_distillation.py`
- `examples/14_advanced_optimizers_muon_stiefel.py`
- `tests/examples/test_13.py`
- `tests/examples/test_14.py`

**Example 13 specific bugs:**

1. **Duplicate of Example 04.** Both claim "6-pillar KD" with different
   implementations. **One must be deleted.**

   Coordinate: if W6-4 keeps example 04, then W6-9 deletes example 13
   (or vice versa). Document the decision in the commit message.

   **Preferred:** Delete 13, keep 04 rewritten correctly. Reason: 04
   comes first in the sequence, users will hit it first.

2. **`loss_pillar4_grad_alignment`** — same problem as example 04:
   computes gradients on unrelated parameters.

3. **Hardcoded `pass_fail(..., True, ...)`** — fix.

**Example 14 specific bugs:**

1. **Verify imports exist:**
   ```python
   from neuralops.trainer.optimizers.muon import Muon, zeropower_via_newtonschulz5
   from neuralops.trainer.optimizers.stiefel import (
       project_stiefel_gradient, stiefel_qr_retraction,
   )
   ```
   If either doesn't exist, the example must raise `NotImplementedError`.

2. **The Muon optimizer may not exist** — check if `Muon` class exists
   in the library. If not, this example is a lie.

3. **The Riemannian Stiefel optimizer may not exist** — check if
   `project_stiefel_gradient` and `stiefel_qr_retraction` exist.

4. **`pass_fail(..., True, ...)`** — fix.

**Acceptance criteria:**
- [ ] Example 13 is deleted OR merged with 04
- [ ] Example 14 only claims APIs that exist
- [ ] No hardcoded PASS
- [ ] Tests for each remaining example

---

#### W6-10 — Example 15

**Owned Paths:**
- `examples/15_serving_vllm_sglang_trtllm.py`
- `tests/examples/test_15.py`

**Specific bugs:**

1. **Verify each import:**
   ```python
   from neuralops.integrations.vllm_backend import (
       NeuralOpsAttentionBackend, NeuralOpsImpl, NeuralOpsRecurrentSpec,
       register as register_vllm,
   )
   from neuralops.integrations.sglang_backend import (
       NeuralOpsLinearAttnBackend, register as register_sglang,
   )
   from neuralops.integrations.tensorrt_llm import (
       TensorRTLLMPluginBuilder, TRTPluginSpec,
   )
   from neuralops.integrations.tensorrt_llm_plugin import neuralops_attention
   ```
   If ANY of these doesn't exist, either:
   - The example must raise `NotImplementedError` for that section
   - Or the section must be deleted entirely

2. **`is_package_installed`** — verify it works.

3. **The status "PASS (offline synthesis)"** is misleading. If TRT-LLM
   isn't installed, "offline synthesis" doesn't prove anything works.

4. **Verify the example still runs to completion on a CPU-only host
   without any of these packages.** It should exit 0 with clear
   "SKIPPED" messages, or exit 1 if any integration is genuinely broken.

**Acceptance criteria:**
- [ ] Each integration is either real (imports work) or explicitly
      documented as missing
- [ ] Example runs to completion on CPU-only
- [ ] Exit code is meaningful (0 only if all claimed features work,
      1 if some are broken/missing)
- [ ] Test asserts correct exit code

---

#### W6-11 — Example 16

**Owned Paths:**
- `examples/16_distributed_parallelism.py`
- `tests/examples/test_16.py`

**Specific bugs:**

1. **`simulate_layer_parallel_transformer` uses fragile paths:**
   ```python
   distil = model.transformer.distilbert
   emb = distil.embeddings(input_ids)
   h0_out = distil.transformer.layer[0](emb, attn_mask=attn_mask)
   h0 = h0_out[0] if isinstance(h0_out, tuple) else h0_out
   h1_out = distil.transformer.layer[1](h0, attn_mask=attn_mask)
   h1 = h1_out[0] if isinstance(h1_out, tuple) else h1_out
   pooled = h1[:, 0]
   pooled = model.transformer.pre_classifier(pooled)
   ```
   This assumes the exact structure of `TinyClassifier`. Any refactor
   breaks it.

   **Fix:** Either:
   - Wrap `TinyClassifier` in a stable interface:
     ```python
     def run_layers(model, input_ids, layer_indices):
         """Run a subset of layers, return activations."""
         # Use model.get_layer(i) or model.layers[i]
     ```
   - Or use `model.forward` with hooks to extract intermediate activations:
     ```python
     activations = {}
     def hook(name):
         def fn(module, inp, out):
             activations[name] = out
         return fn
     model.transformer.distilbert.transformer.layer[0].register_forward_hook(
         hook("layer0"))
     model(input_ids=input_ids)
     # Use activations
     ```
   - Or just delete the example (it's a simulation, not a real distributed
     test).

2. **The "parity check" with `assert_parity`** — verify it catches real
   mismatches. If `simulate_layer_parallel` is buggy, does the test fail?

**Acceptance criteria:**
- [ ] Example 16 either uses stable module access or is deleted
- [ ] No fragile path assertions
- [ ] Parity test actually catches mismatches
- [ ] Test for the example

---

#### W6-12 — Example 17

**Owned Paths:**
- `examples/17_recipe_dsl_and_snapshot_rollback.py`
- `tests/examples/test_17.py`

**Specific bugs:**

1. **`Recipe(... kind="SurgeryRecipe" ...)`** — verify `Recipe` accepts
   `kind="SurgeryRecipe"`. The API might expect `"Recipe"` or a
   different string.

2. **`snapshot.restore(surgery.model)`** — verify it returns a new model,
   not None. If `ModelSnapshot.restore` mutates in place and returns
   None, this line is wrong.

3. **`plan_recipe(recipe, model=model)`** — verify the signature. Maybe
   `plan_recipe(recipe)` only.

4. **`surgery.apply("attention.replace", target="cosformer")`** — verify
   the op exists and accepts `target="cosformer"`.

**Fix:** Test each API call in isolation first. If any doesn't work,
either fix the API call or raise `NotImplementedError`.

**Acceptance criteria:**
- [ ] Every API call in the example actually works
- [ ] The example exits 0 if all APIs work, or raises
      `NotImplementedError` if any is missing
- [ ] Test asserts the correct exit code

---

#### W6-13 — Examples 18, 19

**Owned Paths:**
- `examples/18_three_level_kernel_dispatch.py`
- `examples/19_spectral_and_hessian_diagnostics.py`
- `tests/examples/test_18.py`
- `tests/examples/test_19.py`

**Example 18 specific bugs:**

1. **Verify imports:**
   ```python
   from neuralops import force_kernel_backend
   from neuralops.ops.cuda import (
       cosformer_linear_attention, get_dispatch_stats, reset_dispatch_stats,
   )
   ```
   If any doesn't exist, example must raise `NotImplementedError`.

2. **The example claims to test 3 backend levels.** If CUDA is not
   available (typical case), only Level 3 (PyTorch) runs. That's honest,
   but the docstring should not claim "3-level hierarchy verified" if
   only 1 level runs.

3. **The final `pass_fail(..., True, ...)`** — fix.

**Example 19 specific bugs:**

1. **`interp = "Sharp loss curvature direction"`** — this is fake
   interpretation. The code doesn't know what "sharp" means. Fix:
   ```python
   interp = f"eigenvalue {ev:.6f}"  # just the number
   ```
   Or add a real threshold with justification:
   ```python
   interp = "high curvature" if ev > 1.0 else "moderate curvature"
   # But document what "1.0" means
   ```

2. **`pass_fail(..., True, ...)`** — fix.

**Acceptance criteria:**
- [ ] Example 18 only runs sub-checks that work
- [ ] Example 19 doesn't make fake claims
- [ ] No hardcoded PASS
- [ ] Tests for each

---

#### W6-14 — Example 20 (Verdict: Delete or Fix)

**Owned Paths:**
- `examples/20_long_context_niah_eval.py`
- `tests/examples/test_20.py`
- `audits/wave6/w6_14_verdict.md`

**Specific bugs:**

The example is **completely fake**:

1. `evaluate_synthetic_niah` uses `torch.randn` for "attention scores".
   No model, no tokenizer, no NIAH.

2. `simulate_h2o_eviction` uses `torch.randn` and `topk` to fake cache
   eviction.

3. The example claims: "Evaluates retrieval across context lengths:
   [128, 256, 512, 1024] tokens" — but there's no retrieval.

**Verdict decision (do this now):**

**Option A — Rewrite with real model:**
- Use `hf-internal-testing/tiny-random-gpt2` or similar
- Real tokenizer
- Real NIAH protocol: insert passkey in a haystack of random text
- Real retrieval: ask the model "What is the passkey?" and check output
- This requires internet access on first run (HuggingFace cache)

If the library provides `neuralops.evaluation.niah_passkey.NeedleInAHaystackEvaluator`,
use it. Otherwise, write real NIAH from scratch (this is significant work).

**Option B — Delete:**
Document in `audits/wave6/w6_14_verdict.md`:
```markdown
# Example 20 Verdict: DELETED

Reason: The example claimed to demonstrate NIAH and H2O cache eviction
but used torch.randn to fabricate data. It contains no model, no
tokenizer, no retrieval.

To honestly demonstrate NIAH, a real language model and tokenizer are
required. `TinyClassifier` (used elsewhere in examples) is not a language
model. Deleting this example is better than keeping a lie.

If you need NIAH, use:
    from neuralops.evaluation.niah_passkey import NeedleInAHaystackEvaluator
Or implement it manually with `transformers.AutoModelForCausalLM`.

Alternative example to demonstrate long-context behavior:
See `examples/05_fla_style_benchmarking.py` for kernel-level long-seq
benchmarking (though it only goes up to seq_len=32, extend it if needed).
```

**Option C — Convert to NotImplementedError:**

Replace `main()` with:
```python
def main() -> int:
    section("Example 20: Long-Context NIAH & H2O Eviction")
    kv("Status", "NOT IMPLEMENTED")
    raise NotImplementedError(
        "This example requires a real language model and tokenizer for "
        "honest NIAH demonstration. See "
        "neuralops.evaluation.niah_passkey.NeedleInAHaystackEvaluator for "
        "the API. The previous torch.randn-based fake has been removed."
    )
```

**Preferred:** Option B (delete). The alternative options require
significant work for questionable value.

**Acceptance criteria:**
- [ ] Decision made and documented in `w6_14_verdict.md`
- [ ] If DELETED: file removed, documented in `docs/examples_audit.md`
      (coordinate with W6-16)
- [ ] If CONVERTED: file raises NotImplementedError, test asserts that
- [ ] If REWRITTEN: real model, real data, real retrieval, real cache
      eviction

---

### PHASE 3 — Integration & Documentation (6 agents)

#### W6-15 — Integration Test for All Examples

**Owned Paths:**
- `tests/examples/test_all_examples.py` (create)
- `examples/README.md` (create or rewrite)

**Mission:** One test that runs every example in a subprocess and asserts
exit code 0. Plus a README explaining each example.

**Test file:**

```python
import subprocess
import sys
from pathlib import Path
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example):
    """Each example must run cleanly in a subprocess."""
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{example.name} failed with exit {result.returncode}:\n"
        f"STDOUT:\n{result.stdout[-1000:]}\n"
        f"STDERR:\n{result.stderr[-1000:]}"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_has_no_fake_patterns(example):
    """Each example must not contain fake patterns."""
    src = example.read_text()
    # Allow torch.randn only in comments or in weight initialization
    for i, line in enumerate(src.splitlines(), 1):
        if "torch.randn" in line and not line.strip().startswith("#"):
            # Exception: `nn.Linear` weight initialization is fine
            if "weight" not in line.lower() and "param" not in line.lower():
                pytest.fail(
                    f"{example.name}:{i} uses torch.randn for non-init: {line.strip()}"
                )
```

**`examples/README.md`:**

```markdown
# NeuralOps Examples

Each example demonstrates a specific subsystem of the NeuralOps library.
All examples run on CPU without external dependencies (unless noted).

| # | Example | Demonstrates | Hardware | Deps |
|---|---|---|---|---|
| 01 | `01_attention_surgery.py` | `neuralops.replace_attention` across 5 families | CPU | — |
| 02 | `02_model_compression.py` | 4 SVD compressors | CPU | — |
| ... | ... | ... | ... | ... |

## Quick start

```bash
python examples/01_attention_surgery.py
```

## Removed examples

The following examples were removed in Wave 6 because they could not be
honestly implemented with the current library API:

- `NN_*.py` — reason (see `docs/examples_audit.md`)
```

**Acceptance:**
- [ ] Test file exists and passes
- [ ] README lists all current examples
- [ ] README lists removed examples

---

#### W6-16 — `docs/examples_audit.md`

**Owned Paths:**
- `docs/examples_audit.md`

**Mission:** Document the full audit result. For each of the original 20
examples, one of these verdicts:

- **KEPT**: no changes needed (unlikely)
- **REWRITTEN**: fake patterns removed, now demonstrates real feature
- **CONVERTED**: replaced with `NotImplementedError`
- **DELETED**: file removed, reason documented

**Format:**

```markdown
# NeuralOps Examples Audit Report

**Date:** 2026-11-15
**Author:** Wave 6 remediation

## Summary

| Verdict | Count |
|---|---|
| KEPT | 0 |
| REWRITTEN | N |
| CONVERTED | N |
| DELETED | N |
| **Total** | 20 |

## Per-Example Detail

### Example 01 — `01_attention_surgery.py`
- **Verdict:** REWRITTEN
- **Original issues:**
  - `demonstrate_weight_transplantation` was trivially true
  - Hardcoded `pass_fail("Overall", True, ...)`
- **Fixes applied:**
  - Test now verifies specific Q/K/V weights are preserved
  - Overall status computed from `p1 and p2 and p3`
- **Tests:** `tests/examples/test_01.py::test_example_runs`

### Example 20 — `20_long_context_niah_eval.py`
- **Verdict:** DELETED
- **Original issues:**
  - Entire NIAH was fake (torch.randn as attention scores)
  - H2O eviction was fake (torch.randn + topk)
  - No model, no tokenizer, no retrieval
- **Fix applied:**
  - File removed. Honest implementation requires a language model.
- **Documented:** See `audits/wave6/w6_14_verdict.md`

(... continue for all 20)
```

**Acceptance:**
- [ ] Report exists
- [ ] Every original example (01..20) is accounted for
- [ ] Every DELETED is documented with reason
- [ ] Every REWRITTEN lists original issues and applied fixes

---

#### W6-17 — Fake Detection Script + CI

**Owned Paths:**
- `scripts/check_examples_fake.sh` (create)
- `.github/workflows/examples.yml` (create)

**Mission:** Automate the fake detection. Any future PR that
reintroduces fake patterns should fail CI.

**`scripts/check_examples_fake.sh`:**

```bash
#!/bin/bash
set -e

EXAMPLES="examples"
FAIL=0

echo "=== Checking examples for fake patterns ==="

# Pattern 1: torch.randn for evaluation data (not weight init)
echo "-- Checking for torch.randn (except weight init)..."
if grep -rn "torch\.randn" "$EXAMPLES" --include="*.py" | \
   grep -v "tests/" | \
   grep -vE "(weight|param|init|embedding)" | \
   grep -v "^Binary"; then
    echo "FAIL: torch.randn found outside weight init"
    FAIL=1
fi

# Pattern 2: pass_fail with literal True for overall status
echo "-- Checking for hardcoded PASS in overall status..."
if grep -rn 'pass_fail("Overall[^"]*", *True' "$EXAMPLES" --include="*.py"; then
    echo "FAIL: hardcoded PASS in overall status"
    FAIL=1
fi

# Pattern 3: silent try/except ImportError
echo "-- Checking for silent import skips..."
if grep -rn -B1 -A1 "except ImportError" "$EXAMPLES" --include="*.py" | \
   grep -q "HAS_\|SKIPPED\|return True"; then
    echo "FAIL: silent import skip pattern detected"
    FAIL=1
fi

# Pattern 4: duplicate pillar reimplementations
echo "-- Checking for reimplemented 6-pillar KD..."
if grep -rn "^def pillar[0-9]" "$EXAMPLES" --include="*.py"; then
    echo "FAIL: 6-pillar KD reimplemented inline"
    FAIL=1
fi

if [ $FAIL -eq 0 ]; then
    echo "=== All checks passed ==="
    exit 0
else
    echo "=== FAILED ==="
    exit 1
fi
```

**`.github/workflows/examples.yml`:**

```yaml
name: Examples

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install
        run: |
          pip install -e .
          pip install pytest
      - name: Check examples for fake patterns
        run: bash scripts/check_examples_fake.sh
      - name: Run all examples
        run: |
          for f in examples/[0-9][0-9]_*.py; do
              echo "=== $f ==="
              python "$f" || exit 1
          done
      - name: Run example tests
        run: pytest tests/examples/ -v
```

**Acceptance:**
- [ ] Script exists, executable
- [ ] Running script on clean tree returns 0
- [ ] Running script after reintroducing a fake pattern returns 1
- [ ] Workflow file exists
- [ ] Workflow has the 3 steps above

---

#### W6-18 — Cross-Example Consistency Audit

**Owned Paths:**
- `audits/wave6/consistency.md`

**Mission:** Compare examples with each other. Report inconsistencies.
Do NOT fix them (fixes belong to specific owners).

**Audit dimensions:**

1. **Overall status format:**
   ```bash
   for f in examples/[0-9][0-9]_*.py; do
       echo "$f:"
       grep -n 'pass_fail("Overall' "$f" || echo "  (no Overall check)"
   done
   ```
   Which examples use `all_passed`, which use `True`?

2. **Section naming:**
   Do examples use consistent section titles ("Part 1:", "Part 2:")?
   Or inconsistent ("Demo:", "Test:", "Phase:")?

3. **`kv()` usage:**
   Do all examples call `kv()` for key-value output? Or some use `print()`?

4. **`table()` usage:**
   Do all examples use `table()` for tabular output? Or some inline
   formatting?

5. **Import order:**
   Do all examples import `neuralops` after `torch`? Or inconsistent?

6. **Docstring format:**
   Do all examples have `Subsystem Demonstrated:` section?

7. **Exit code convention:**
   Do all examples `return 0 if all_passed else 1`? Or some `sys.exit(0)`?

**Report format:**

```markdown
# Cross-Example Consistency Audit

## Dimension 1: Overall status format
- 18 examples use `all_passed = p1 and p2 and p3`
- 2 examples (05, 06) still hardcode `True`
- Action: 05, 06 owners must fix

## Dimension 2: Section naming
- 12 examples use "Part 1:"
- 5 examples use "Stage 1:"
- 3 examples use "Phase 1:"
- Action: Consider standardizing on "Part 1:" in a future wave

(etc.)
```

**Acceptance:**
- [ ] Report covers 7 dimensions
- [ ] Recommendations listed per dimension
- [ ] No file edits (audit only)

---

#### W6-19 — Run All Examples in CI Context

**Owned Paths:**
- `audits/wave6/all_examples_run.md`
- `audits/wave6/all_examples_output/` (captured stdout/stderr per example)

**Mission:** Run every example one final time and capture output.

**Tasks:**

1. For each example:
   ```bash
   mkdir -p audits/wave6/all_examples_output
   for f in examples/[0-9][0-9]_*.py; do
       name=$(basename "$f" .py)
       python "$f" > "audits/wave6/all_examples_output/${name}.log" 2>&1
       echo "$(basename "$f"): exit $?" >> audits/wave6/all_examples_run.md
   done
   ```

2. Produce `all_examples_run.md`:
   ```markdown
   # All Examples Final Run
   
   | # | Example | Exit | Duration | Log |
   |---|---|---|---|---|
   | 01 | 01_attention_surgery.py | 0 | 1.2s | [log](all_examples_output/01_attention_surgery.log) |
   | ... | ... | ... | ... | ... |
   ```

3. Verify:
   ```bash
   grep -c "exit 0" audits/wave6/all_examples_run.md
   ```
   Expected: same as number of remaining examples.

**Acceptance:**
- [ ] All examples exit with code 0
- [ ] Logs captured for each
- [ ] Summary table generated

---

### PHASE 4 — Final Aggregation (1 agent)

#### W6-20 — Master Report + Final Verification

**Owned Paths:**
- `audits/wave6_master.md`

**Mission:** Aggregate all reports, run final verification, report status
to the human coordinator.

**Tasks:**

1. Read all 19 prior reports:
   ```bash
   cat audits/wave6/baseline_audit.md
   cat audits/wave6/w6_14_verdict.md
   cat audits/wave6/consistency.md
   cat audits/wave6/all_examples_run.md
   cat docs/examples_audit.md
   ```

2. **Run the fake detection:**
   ```bash
   bash scripts/check_examples_fake.sh
   echo "Exit: $?"
   ```
   Must be 0.

3. **Run all example tests:**
   ```bash
   pytest tests/examples/ -v > audits/wave6/w6_20_tests.log 2>&1
   tail -20 audits/wave6/w6_20_tests.log
   ```

4. **Run every example:**
   ```bash
   for f in examples/[0-9][0-9]_*.py; do
       python "$f" > /dev/null 2>&1
       if [ $? -ne 0 ]; then
           echo "FAIL: $f"
       fi
   done
   ```

5. **Produce `audits/wave6_master.md`:**

```markdown
# NeuralOps Wave 6 — Master Report

## Summary

- Total original examples: 20
- Kept: N
- Rewritten: N
- Converted to NotImplementedError: N
- Deleted: N
- Currently passing: N
- Currently failing: N

## Per-Example Verdict

| # | File | Verdict | Exit Code | Reason (if removed) |
|---|---|---|---|---|
| 01 | 01_attention_surgery.py | REWRITTEN | 0 | — |
| ... | ... | ... | ... | ... |
| 20 | 20_long_context_niah_eval.py | DELETED | N/A | Fake NIAH (see w6_14_verdict.md) |

## Fake Detection Result

`bash scripts/check_examples_fake.sh` exit code: 0 (PASS)

## Tests

`pytest tests/examples/ -v`: N passed, 0 failed (see w6_20_tests.log)

## Remaining Issues (for Wave 7)

If any example could not be fixed, list it here:

- Example NN: reason. Suggested Wave 7 action.

## Recommendation

- [x] Wave 6 remediation complete. Examples are honest.
- [ ] Additional work needed (see Remaining Issues above)
```

6. **Report to human:**

```
COMPLETE: W6-20 (master report)

Final Status: WAVE 6 COMPLETE
- Examples kept: N/20
- Examples deleted: N/20
- All remaining examples pass `python examples/NN.py`
- Fake detection script passes
- All example tests pass
- Master report: audits/wave6_master.md

Recommendation: MERGE WAVE 6 to main.
```

**Acceptance:**
- [ ] `audits/wave6_master.md` complete
- [ ] Fake detection passes
- [ ] All example tests pass
- [ ] Every remaining example runs to completion
- [ ] Master report clearly states final status

---

## 6. WAIT PROTOCOL

After spawning agents in Phase 1, WAIT for W6-1 and W6-2 to complete
before spawning Phase 2.

```python
import json
import time

PHASE_1 = ["W6-1", "W6-2"]

while True:
    progress = json.load(open('.agent/WAVE6_PROGRESS.json'))
    statuses = {a: v['status'] for a, v in progress['agents'].items()
                if a in PHASE_1}
    if all(s in ('complete', 'blocked', 'failed') for s in statuses.values()):
        break
    print(f"Phase 1: {statuses}")
    time.sleep(180)

# Check W6-1 didn't fail. If it did, Phase 2 is blocked.
if progress['agents']['W6-1']['status'] == 'failed':
    raise RuntimeError("W6-1 (_shared) failed. Cannot proceed to Phase 2.")
```

Then spawn Phase 2 (W6-3 through W6-14, W6-17). Wait for all to complete.

Then spawn Phase 3 (W6-15, W6-16, W6-18).

Then spawn Phase 4 (W6-19, W6-20).

**Do NOT proceed to Phase N+1 if Phase N has failures** (except `blocked`
which is acceptable — it means the agent reported a legitimate blocker).

---

## 7. MERGE PROTOCOL

### 7.1 — Verify all phases complete

```bash
for AGENT in W6-1 W6-2 ... W6-20; do
    BRANCH="wave6/${AGENT}"
    if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
        echo "MISSING: $AGENT"
        continue
    fi
    COUNT=$(git log "$BRANCH" --oneline | wc -l)
    echo "$AGENT: $COUNT commits"
done
```

If any agent has 0 commits and status isn't `blocked`: BLOCK merge.

### 7.2 — Merge order

```bash
git checkout main
git pull

# Phase 1
git merge --no-ff wave6/W6-1 -m "merge: wave6/W6-1 (_shared foundation)"
git merge --no-ff wave6/W6-2 -m "merge: wave6/W6-2 (baseline audit)"

# Phase 2
for A in W6-3 W6-4 W6-5 W6-6 W6-7 W6-8 W6-9 W6-10 W6-11 W6-12 W6-13 W6-14 W6-17; do
    git merge --no-ff "wave6/$A" -m "merge: wave6/$A" || echo "CONFLICT: $A"
done

# Phase 3
for A in W6-15 W6-16 W6-18; do
    git merge --no-ff "wave6/$A" -m "merge: wave6/$A" || echo "CONFLICT: $A"
done

# Phase 4
for A in W6-19 W6-20; do
    git merge --no-ff "wave6/$A" -m "merge: wave6/$A" || echo "CONFLICT: $A"
done
```

After each merge:
```bash
pytest tests/unit tests/contract -x -q
if [ $? -ne 0 ]; then
    git reset --hard HEAD~1
    echo "REVERTED: wave6/$A (test failure)"
fi
```

If conflict: `git merge --abort`, log, continue.

### 7.3 — Final verification

```bash
pytest tests/examples/ -v
bash scripts/check_examples_fake.sh
for f in examples/[0-9][0-9]_*.py; do
    python "$f" > /dev/null 2>&1 || echo "FAIL: $f"
done
```

If all clean: report "WAVE 6 COMPLETE".
If not: report which examples still fail.

---

## 8. FINAL REPORT FORMAT

```markdown
# Wave 6 Final Report

## Agents
- Completed: N/20
- Blocked: N (see below)
- Failed: N

## Examples
- Original: 20
- Kept as-is: N
- Rewritten: N
- Converted to NotImplementedError: N
- Deleted: N
- Currently passing `python examples/NN.py`: N

## Verification
- `scripts/check_examples_fake.sh`: PASS/FAIL
- `pytest tests/examples/`: N passed, N failed
- `pytest tests/unit tests/contract`: N passed, N failed

## Blockers (if any)
- W6-N: reason

## Recommendation
- [x] Ready to merge
- [ ] Needs attention (see Blockers)
```

---

## 9. RESUME PROTOCOL

If the server restarts:

1. Read `.agent/WAVE6_PROGRESS.json`
2. For each agent:
   - If status `complete`: skip
   - If status `blocked`: skip
   - If status `failed`: skip (report)
   - If status `pending` or `running`:
     - Check `git log wave6/<AGENT> -1`
     - If commits exist: mark "likely complete" and continue
     - If no commits: respawn this agent
3. Determine current phase:
   - Phase 1 incomplete: resume Phase 1
   - Phase 1 complete, Phase 2 incomplete: resume Phase 2
   - etc.
4. Continue from Section 6 (wait protocol)

**Never** start over from Phase 1.

---

## 10. ABSOLUTE PROHIBITIONS

1. Do NOT proceed to Phase N+1 without Phase N being `complete` or `blocked`.
2. Do NOT spawn fewer than 20 agents.
3. Do NOT merge a branch with 0 commits (unless blocked).
4. Do NOT accept `torch.randn` for evaluation data.
5. Do NOT accept hardcoded `pass_fail(..., True, ...)`.
6. Do NOT accept silent `try/except ImportError`.
7. Do NOT accept reimplementation of library features.
8. Do NOT accept a "rewritten" example that is still fake in a different
   way. If the rewrite isn't honest, the example must be deleted.
9. Do NOT delete an example without documenting the reason.
10. Do NOT return to the user before Section 8 (Final Report) is produced.

---

## 11. BEGIN

Start with Section 2 (Pre-Flight). Then Section 3 (worktrees). Then
Section 6 Phase 1 spawn. Wait. Then Phase 2. Wait. Then Phase 3. Wait.
Then Phase 4. Then Section 7 (merge). Then Section 8 (final report).

Do NOT stop between phases to ask "should I continue?". The answer is
always yes.

Do NOT return to the user until Section 8 (Final Report) is produced.

BEGIN.
```

---

## Opening Message to Paste

After saving the above as `.agent/WAVE6_EXECUTE.md`, paste this exact message to your orchestrator agent:

```
Execute .agent/WAVE6_EXECUTE.md exactly as written.

You are the orchestrator. Follow Section 6 (Wait Protocol) rigorously:
run Phase 1, wait for it to complete, then Phase 2, wait, then Phase 3,
wait, then Phase 4.

Do NOT return to the user between phases.

Do NOT claim a phase complete until every agent in that phase has
reported `COMPLETE` or `BLOCKED`.

If the server restarts mid-execution, use Section 9 (Resume Protocol).
Do NOT start over from Phase 1.

Return only when Section 8 (Final Report) is produced.

BEGIN WITH SECTION 2 (PRE-FLIGHT) NOW.
```

---

## What Makes This Different

The critical differences from previous waves:

1. **The verdict framework is explicit**: every example gets one of four verdicts (KEPT / REWRITTEN / CONVERTED / DELETED). **Deleting is a valid outcome.** This removes the pressure for agents to fake a "pass".

2. **`scripts/check_examples_fake.sh`** is a hard gate. It uses grep to detect the exact fake patterns. If any agent reintroduces `torch.randn` or `pass_fail(..., True, ...)`, the CI fails.

3. **Each agent's "before you start" audit** forces them to look at the current state before touching anything. They cannot skip the diagnosis.

4. **The four laws** (no fake data, no hardcoded PASS, no silent skip, no reimplementation) are repeated in the spawn template so agents cannot claim they didn't see them.

5. **Phases with hard gates**: W6-1 (foundation) must complete before W6-3..W6-14 begin. This prevents the "rewrite against broken `_shared`" scenario.

6. **Real test evidence required**: each commit message must include the exact command output (`python examples/NN.py` exit 0, `pytest test_NN.py` pass). Agents cannot claim success without evidence.

7. **Explicit deletions documented**: when an example is deleted, the reason is written to a specific file (`w6_N_DELETIONS.md` or `docs/examples_audit.md`). There's no ambiguity about why.

8. **Final report requires N examples still passing**: not "all examples pass", but "N examples still exist and all N pass". If N=10, that's honest. If N=20 but 10 are fakes, that's not allowed.

If this wave still fails, the issue is not the prompt. At that point, the human needs to either:
- Manually rewrite the top 5 most important examples
- Delete the rest
- Ship with "examples are being rewritten in v2.5" as a known limitation

There's no prompt trick that forces an LLM to write real code when its training patterns favor fake code. The prompt can only make fake code detectable and unacceptable. This prompt does that.

اینارو برام کالم کامل کامل برسی میکنی دقیقا ؟
و بعد اینارو اوکی کنی کامل البته فبلش کدارو چک کن که دقیق باشه 
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-15T02:16:00+03:30.
</ADDITIONAL_METADATA>
