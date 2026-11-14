# NEURALOPS WAVE 4 — FINAL REMEDIATION (20 AGENTS)

**Document version:** 1.0
**Effective date:** 2026-11-15
**Mission owner:** Human coordinator
**Executor:** You (the orchestrator agent)
**Prerequisites:** Wave 1 (5 agents) and Wave 2 (20 agents) must be fully merged
into `main` and `pytest tests/unit tests/contract` must be green.

---

## 0. HOW TO READ THIS DOCUMENT

This document is the single source of truth for Wave 4. It contains:

- **Section 1** — What Wave 4 is, and why it exists.
- **Section 2** — Pre-flight conditions you MUST verify before any work.
- **Section 3** — The Git protocol you MUST follow for every single agent.
- **Section 4** — The spawn template you MUST use verbatim for every agent.
- **Section 5** — The full specification of the 20 agents (W4-L1 through W4-R1).
- **Section 6** — The wait protocol (what to do while agents run).
- **Section 7** — The merge protocol (exact order, how to handle conflicts).
- **Section 8** — The final verification protocol.
- **Section 9** — The report format you MUST use at the end.
- **Section 10** — The resume protocol for server restarts.
- **Section 11** — The troubleshooting guide.
- **Section 12** — Absolute prohibitions.

If any instruction in this document conflicts with anything else you've been
told, **this document wins**. If you are unsure about anything, STOP and
report "BLOCKED: <question>" rather than guessing.

---

## 1. WHAT WAVE 4 IS

Wave 1 and Wave 2 fixed and deep-audited approximately 300 of the 500+ files
in `neuralops/`. Wave 4 addresses the remaining ~250 files and the entire
non-code infrastructure (tests, CI, docs, packaging, release).

Wave 4 is the **final remediation wave**. After Wave 4, the library must be
production-ready for a `v2.4.0` release.

Wave 4 covers:

1. **Layers** — the ~91 attention mechanisms that Wave 2 only sampled.
2. **Compression** — the ~37 compressors that Wave 2 only sampled.
3. **Ops** — the ~60 Triton and CUDA kernels that Wave 2 only sampled.
4. **Trainer** — the entire `neuralops/trainer/` subsystem (never audited).
5. **API + Config** — `neuralops/api.py` and `neuralops/config/**` (facades
   and schema, never deep-audited).
6. **Core remaining** — the 40% of `neuralops/core/**` that Wave 2 skipped.
7. **Domain services** — `export/`, `inference/`, `integrations/`,
   `evaluation/`, `ui/`, `cli/`, `utils/`, `results_saver/` — all only
   sampled in Wave 1.
8. **Benchmarks remaining** — `benchmarks/suites/`, `benchmarks/core/`,
   `benchmarks/kd/`, `benchmarks/storage/`, `benchmarks/adapters/`, and
   every persistence file (`saver.py`, `summarizer.py`, `reproduce.py`,
   `index.py`, `hf_sync.py`, `git_sync.py`, `dedup.py`, `dispatcher.py`,
   `paths.py`, `executor.py`, `result.py`, `schema.py`).
9. **Benchmarks CLI** — decompose the 2,922-line `benchmarks/cli.py` into
   `benchmarks/cli_v2/`, and quarantine the 58 legacy scripts.
10. **Infrastructure** — reorganize `tests/**`, add GitHub Actions, add
    pre-commit, audit `docs/**` and `examples/**`, audit `pyproject.toml`,
    and produce release artifacts (`CHANGELOG.md`, `BREAKING_CHANGES.md`,
    `AUDIT_MASTER_FIXES.md`, `VERIFICATION.md`, `_version.py` bump, tag).

Wave 4 is executed as **20 parallel subagents**, each isolated in its own
**Git worktree** and each owning a **disjoint file set**. Two agents NEVER
touch the same file.

Wave 4 does NOT touch files fixed in Wave 1 or Wave 2 unless a Wave 4 agent
discovers a regression. If that happens, the agent reports it as a cross-domain
bug (Section 3.4) — it does NOT fix it.

---

## 2. PRE-FLIGHT (MANDATORY)

You MUST run every check in this section and you MUST confirm each passes
before any agent is spawned. If any check fails, STOP and report to the user:
"PRE-FLIGHT FAILED: <check name>: <reason>".

### 2.1 — Repository state

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout main
git pull --ff-only origin main
```

Verify clean working tree:

```bash
if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: working tree is dirty"
    git status --porcelain
    exit 1
fi
```

Verify you are on `main`:

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "FAIL: on branch $BRANCH, expected main"
    exit 1
fi
```

### 2.2 — Wave 1 and Wave 2 completion

Verify Wave 1 merge commits exist (should be 10 merge commits with
"merge: wave1" in the subject):

```bash
WAVE1_COUNT=$(git log --oneline --grep="merge: wave1" | wc -l)
if [ "$WAVE1_COUNT" -lt 10 ]; then
    echo "FAIL: Wave 1 incomplete — only $WAVE1_COUNT merge commits found"
    exit 1
fi
```

Verify Wave 2 merge commits exist (should be 20):

```bash
WAVE2_COUNT=$(git log --oneline --grep="merge: wave2" | wc -l)
if [ "$WAVE2_COUNT" -lt 20 ]; then
    echo "FAIL: Wave 2 incomplete — only $WAVE2_COUNT merge commits found"
    exit 1
fi
```

### 2.3 — Baseline test suite green

```bash
pytest tests/unit tests/contract -x -q
if [ $? -ne 0 ]; then
    echo "FAIL: baseline tests red"
    exit 1
fi
```

If `tests/unit` or `tests/contract` do not exist yet (because Wave 3 was
skipped), run the full suite instead:

```bash
pytest tests/ -x -q -m "not hub and not gpu"
```

### 2.4 — Disk space and tooling

Verify at least 5 GB free disk space (20 worktrees × ~250 MB each ≈ 5 GB):

```bash
FREE_GB=$(df -BG "$(git rev-parse --show-toplevel)" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 5 ]; then
    echo "FAIL: only ${FREE_GB}GB free, need at least 5GB"
    exit 1
fi
```

Verify required tools:

```bash
for tool in git python pytest ruff mypy; do
    command -v $tool >/dev/null 2>&1 || { echo "FAIL: $tool not found"; exit 1; }
done
```

### 2.5 — Create Wave 4 directory structure

```bash
mkdir -p .agent/WAVE4_AGENTS
mkdir -p audits/wave4
mkdir -p issues/wave5_backlog.md
```

### 2.6 — Create the Wave 4 progress file

Write `.agent/WAVE4_PROGRESS.json` with this exact content:

```json
{
  "wave": 4,
  "started_at": "2026-11-15T00:00:00Z",
  "last_updated": "2026-11-15T00:00:00Z",
  "agents": {
    "W4-L1": {"status": "pending", "branch": "wave4/W4-L1", "commits": 0, "worktree": null},
    "W4-L2": {"status": "pending", "branch": "wave4/W4-L2", "commits": 0, "worktree": null},
    "W4-L3": {"status": "pending", "branch": "wave4/W4-L3", "commits": 0, "worktree": null},
    "W4-L4": {"status": "pending", "branch": "wave4/W4-L4", "commits": 0, "worktree": null},
    "W4-O1": {"status": "pending", "branch": "wave4/W4-O1", "commits": 0, "worktree": null},
    "W4-O2": {"status": "pending", "branch": "wave4/W4-O2", "commits": 0, "worktree": null},
    "W4-C1": {"status": "pending", "branch": "wave4/W4-C1", "commits": 0, "worktree": null},
    "W4-T1": {"status": "pending", "branch": "wave4/W4-T1", "commits": 0, "worktree": null},
    "W4-T2": {"status": "pending", "branch": "wave4/W4-T2", "commits": 0, "worktree": null},
    "W4-T3": {"status": "pending", "branch": "wave4/W4-T3", "commits": 0, "worktree": null},
    "W4-A1": {"status": "pending", "branch": "wave4/W4-A1", "commits": 0, "worktree": null},
    "W4-C2": {"status": "pending", "branch": "wave4/W4-C2", "commits": 0, "worktree": null},
    "W4-E1": {"status": "pending", "branch": "wave4/W4-E1", "commits": 0, "worktree": null},
    "W4-U1": {"status": "pending", "branch": "wave4/W4-U1", "commits": 0, "worktree": null},
    "W4-U2": {"status": "pending", "branch": "wave4/W4-U2", "commits": 0, "worktree": null},
    "W4-B1": {"status": "pending", "branch": "wave4/W4-B1", "commits": 0, "worktree": null},
    "W4-B2": {"status": "pending", "branch": "wave4/W4-B2", "commits": 0, "worktree": null},
    "W4-B3": {"status": "pending", "branch": "wave4/W4-B3", "commits": 0, "worktree": null},
    "W4-I1": {"status": "pending", "branch": "wave4/W4-I1", "commits": 0, "worktree": null},
    "W4-R1": {"status": "pending", "branch": "wave4/W4-R1", "commits": 0, "worktree": null}
  }
}
```

Once every step in Section 2 passes, proceed to Section 3.

---

## 3. THE GIT PROTOCOL (MANDATORY FOR EVERY AGENT)

Every agent operates under the exact same Git protocol. This section defines
that protocol once, and Section 5 re-states the key rules per agent for
emphasis. If the protocol in this section and the rules in Section 5 ever
disagree, Section 5 wins (because it is more specific per agent).

### 3.1 — Worktree isolation (the most important rule)

**Every agent works in its OWN Git worktree. Never share a working directory
between two agents. Never run two agents on the same branch.**

Why: if two agents share a directory, `git checkout` of one branch will
clobber the other agent's uncommitted work. If two agents work on the same
branch, their commits will interleave unpredictably.

The orchestrator creates the worktrees once (Section 3.2). Each agent is told
which worktree is theirs. An agent MUST NOT create additional worktrees,
MUST NOT switch branches, MUST NOT pull from `main`, and MUST NOT push to
`main`.

### 3.2 — Worktree creation (done once, by the orchestrator)

The orchestrator creates all 20 worktrees in a single pass:

```bash
cd "$(git rev-parse --show-toplevel)/.."
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")
REPO_PATH=$(git -C "$REPO_NAME" rev-parse --show-toplevel)

declare -a AGENTS=(W4-L1 W4-L2 W4-L3 W4-L4 W4-O1 W4-O2 W4-C1 W4-T1 W4-T2
                   W4-T3 W4-A1 W4-C2 W4-E1 W4-U1 W4-U2 W4-B1 W4-B2 W4-B3
                   W4-I1 W4-R1)

for AGENT in "${AGENTS[@]}"; do
    BRANCH="wave4/${AGENT}"
    WORKTREE="${REPO_NAME}_${AGENT//-/_}"
    WORKTREE_ABS="$(pwd)/${WORKTREE}"

    if git -C "$REPO_NAME" rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
        echo "[SKIP] $AGENT: branch $BRANCH already exists"
        EXISTING_WT=$(git -C "$REPO_NAME" worktree list --porcelain | \
                      grep -A1 "branch refs/heads/${BRANCH}" | grep "worktree" | \
                      awk '{print $2}')
        if [ -n "$EXISTING_WT" ]; then
            echo "[SKIP] $AGENT: worktree already at $EXISTING_WT"
            continue
        fi
    else
        git -C "$REPO_NAME" branch "$BRANCH" main || {
            echo "[FAIL] $AGENT: could not create branch"
            exit 1
        }
    fi

    git -C "$REPO_NAME" worktree add "$WORKTREE_ABS" "$BRANCH" || {
        echo "[FAIL] $AGENT: could not create worktree"
        exit 1
    }
    echo "[OK] $AGENT: worktree at $WORKTREE_ABS on branch $BRANCH"

    # Update progress file
    python3 -c "
import json
with open('.agent/WAVE4_PROGRESS.json') as f:
    p = json.load(f)
p['agents']['${AGENT}']['worktree'] = '${WORKTREE_ABS}'
with open('.agent/WAVE4_PROGRESS.json', 'w') as f:
    json.dump(p, f, indent=2)
"
done
```

After this step, you should have 20 sibling directories next to the main repo,
named `{REPO_NAME}_W4_L1` through `{REPO_NAME}_W4_R1`.

Verify:

```bash
ls -d ../${REPO_NAME}_W4_* | wc -l   # must output 20
```

If fewer than 20: STOP and report which ones failed.

### 3.3 — Branch naming convention

- Branch name: `wave4/<AGENT_ID>` (e.g. `wave4/W4-L1`)
- Worktree directory: `{REPO_NAME}_<AGENT_ID_with_underscores>` (e.g.
  `NeuralOps_W4_L1`)
- Commit prefix: `fix(wave4/<AGENT_ID>):` for final commits, `wip(wave4/<AGENT_ID>):` for work-in-progress

### 3.4 — Commit protocol (per agent)

Each agent MUST commit frequently. The reason is simple: if the server
restarts (which it has done repeatedly in Wave 1 and Wave 2), uncommitted work
is lost. Committed work is durable.

Commit frequency rules:

- **After completing each file** in the agent's Owned Paths list: commit.
- **After completing each bug fix** (even if the file is not fully audited):
  commit.
- **Before going idle** for any reason: commit.
- **Never** go more than 60 minutes without a commit.
- **Never** accumulate more than 2000 lines of diff before committing.

Commit message format:

```
<prefix>(wave4/<AGENT_ID>): <short description>

<optional body with details>

Files: <comma-separated list of touched files>
Tests: <comma-separated list of test files added or modified>
Bugs fixed: <comma-separated list of bug IDs from the task file>
```

Where `<prefix>` is one of:

- `fix` — a bug is fixed and tested
- `test` — tests are added with no code changes
- `docs` — documentation-only change
- `wip` — work in progress, not yet complete
- `refactor` — refactoring with no behavioral change
- `build` — build/CI/packaging changes
- `chore` — non-code changes (renaming, moving files)

Example good commits:

```
fix(wave4/W4-L1): handle missing optional bias in GLA

- If bias=False, forward() no longer crashes on None.bias
- Added test_attention_handles_missing_bias in tests/layers/
- Closes bug P0-L1-3 from WAVE4_AGENTS/W4-L1.md

Files: neuralops/layers/gla.py, tests/layers/test_gla.py
Tests: tests/layers/test_gla.py
Bugs fixed: P0-L1-3
```

```
wip(wave4/W4-L2): audit of sliding_window.py in progress

- Added regression test for future-token leakage (currently failing)
- Fix in next commit

Files: neuralops/layers/sliding_window.py, tests/layers/test_sliding_window.py
Tests: tests/layers/test_sliding_window.py
Bugs fixed: none yet
```

Example bad commits (will be rejected by the orchestrator at review time):

```
fix some stuff                         # too vague
wip                                    # no description
Update layers                          # no agent ID, no description
fix(benchmarks/SA-B1): fixed registry  # wrong prefix — this is Wave 4, not Wave 3
```

Rules for `git add`:

- **ALWAYS** use `git add <specific file>` or `git add <specific directory>`.
- **NEVER** use `git add .`.
- **NEVER** use `git add -A`.
- **NEVER** commit files outside your Owned Paths (Section 5 of your task
  file lists them).

If a stray file appears in `git status` that you did not intend to change,
investigate before adding. It may be a leftover from another agent's worktree
(shouldn't happen, but verify).

### 3.5 — Test protocol (per agent)

For every bug you fix:

1. **Reproduce the bug** in `/tmp/repro_<bugid>.py` before making any code
   change. Confirm it produces the buggy behavior.
2. **Write the regression test** in `tests/<domain>/test_<filename>.py`
   BEFORE fixing the bug. Run it once — it MUST fail.
3. **Fix the bug** in the source file.
4. **Run the regression test again** — it MUST pass.
5. **Run the full domain test suite** — it MUST pass.

If step 5 fails, you've introduced a regression. Fix it before committing.

Test naming convention:

- `test_<module>_<bug_summary>.py` (e.g. `test_gla_missing_bias.py`)
- For multiple related bugs in the same module, group them:
  `test_gla_regressions.py`

Test markers (already defined in `pyproject.toml` if Wave 3 is present;
otherwise add them yourself as part of your commit):

- `@pytest.mark.fast` — under 100ms
- `@pytest.mark.slow` — over 1s
- `@pytest.mark.gpu` — requires CUDA
- `@pytest.mark.hub` — requires network
- `@pytest.mark.mps` — requires Apple Silicon

If `pyproject.toml` lacks these markers, add them:

```toml
[tool.pytest.ini_options]
markers = [
    "fast: unit tests < 100ms",
    "slow: integration tests > 1s",
    "gpu: requires CUDA GPU",
    "hub: requires network access",
    "mps: requires Apple Silicon",
]
```

### 3.6 — Lint protocol (per agent)

Before every commit:

```bash
ruff check <changed files>
ruff format --check <changed files>
```

If the agent's work includes type annotations:

```bash
mypy --strict <changed files>
```

If lint or format fails:

- **Try to fix** (10 minutes max).
- **If still failing**: commit with `wip()` prefix and note the failures in
  the commit body. Do NOT `# type: ignore` or `# noqa` unless there's a
  legitimate reason documented in a comment.

### 3.7 — Push protocol (per agent)

After the agent's final commit:

```bash
git push -u origin wave4/<AGENT_ID>
```

If push fails:

- **Network error**: retry up to 3 times.
- **Rejected (non-fast-forward)**: this means someone else pushed to your
  branch. STOP and report "BLOCKED: branch divergence on wave4/<AGENT_ID>".
  Do NOT force-push.
- **Permission error**: STOP and report "BLOCKED: push permission denied".

### 3.8 — Reporting protocol (per agent)

After a successful push, the agent reports back to the orchestrator with
EXACTLY this format:

```
COMPLETE: <AGENT_ID> at <SHA>
Branch: wave4/<AGENT_ID>
Commits: <N>
Files audited: <N>
Files changed: <N>
Tests added: <N>
Bugs fixed: <comma-separated list of IDs>
Bugs logged for Wave 5: <comma-separated list of IDs>
```

Example:

```
COMPLETE: W4-L1 at a3b4c5d6e7f
Branch: wave4/W4-L1
Commits: 12
Files audited: 23
Files changed: 8
Tests added: 9
Bugs fixed: P0-L1-1, P0-L1-3, P1-L1-2, P1-L1-5, P2-L1-1, P2-L1-4
Bugs logged for Wave 5: P2-L1-7, P2-L1-8
```

If the agent cannot complete (Section 5.11 for the blocking protocol):

```
BLOCKED: <AGENT_ID> at <SHA>: <reason>
Branch: wave4/<AGENT_ID>
Commits: <N>
What was completed: <list>
What is blocking: <detailed description>
Suggestion: <what a human should do to unblock>
```

Example:

```
BLOCKED: W4-L3 at 8f9e0d1c2b3: cannot reproduce bug P1-L3-4
Branch: wave4/W4-L3
Commits: 7
What was completed: 21 of 23 files deep-audited; 6 bugs fixed
What is blocking: P1-L3-4 claims that paged_attention leaks memory on MPS,
  but I don't have an MPS device to reproduce. The code looks correct by
  inspection but I cannot verify.
Suggestion: Run on an Apple Silicon machine, or accept the fix as
  best-effort based on code inspection.
```

---

## 4. THE SPAWN TEMPLATE (VERBATIM)

For EVERY one of the 20 agents, you MUST send exactly this message to the
subagent, with `<ANGLE_BRACKET>` placeholders replaced:

```
═══════════════════════════════════════════════════════════════════════
SUBTASK: <AGENT_ID>
═══════════════════════════════════════════════════════════════════════

You are Subagent <AGENT_ID>.

Your working directory: <WORKTREE_ABS>
Your Git branch: wave4/<AGENT_ID>
Your task file: .agent/WAVE4_AGENTS/<AGENT_ID>.md

═══════════════════════════════════════════════════════════════════════
BEFORE YOU DO ANYTHING ELSE (steps 1-5, in this order)
═══════════════════════════════════════════════════════════════════════

STEP 1: Change to your worktree.
    cd <WORKTREE_ABS>

STEP 2: Confirm you are on the correct branch.
    git rev-parse --abbrev-ref HEAD
    # MUST output: wave4/<AGENT_ID>
    # If it outputs anything else, STOP and report:
    #   "ERROR: wrong branch (<actual>), expected wave4/<AGENT_ID>"
    # Do NOT proceed. Do NOT switch branches. Report and stop.

STEP 3: Confirm the worktree is clean.
    git status --porcelain
    # MUST output nothing.
    # If it outputs files, STOP and report:
    #   "ERROR: worktree dirty with files: <list>"
    # Someone else's work is in your directory. Report and stop.

STEP 4: Read the three reference documents, in this order.
    cat AGENTS.md
    cat .agent/WAVE4_EXECUTE.md
    cat .agent/WAVE4_AGENTS/<AGENT_ID>.md

    The third file is your full task specification. Read it end to end
    before you touch any code. Everything you need to know about which
    files you own, which bugs to fix, and which tests to add is in that
    file.

STEP 5: Confirm you understand your Owned Paths.
    # From your task file, extract the list of files/directories you are
    # allowed to touch. Write them here for your own reference:
    #
    # Owned Paths:
    #   <path 1>
    #   <path 2>
    #   ...
    #
    # ANYTHING outside this list is OFF LIMITS. If you find a bug outside
    # your Owned Paths, you MUST NOT fix it. Instead, append to:
    #   .agent/WAVE4_AGENTS/<AGENT_ID>_CROSS.md
    # with this format:
    #   ## <timestamp>
    #   File: <file path>
    #   Line: <line number or "unknown">
    #   Symptom: <what's wrong>
    #   Suspected bug ID: <P0/P1/P2 if guessable>
    #   Suggested fix: <optional>

Only after STEP 5 succeeds, proceed to the EXECUTION section below.

═══════════════════════════════════════════════════════════════════════
EXECUTION
═══════════════════════════════════════════════════════════════════════

For EACH file listed in your task file's "Files to Audit" section:

PHASE A — READ AND INSPECT
  1. Read the entire file (no skimming).
  2. Note the class names, function signatures, and any docstrings
     that claim specific behavior.
  3. Note the imports — especially torch, torch.nn, torch.nn.functional,
     and any neuralops imports.

PHASE B — SMOKE TEST
  1. Run a minimal import:
       python -c "import <module>"
  2. Instantiate the main class with realistic dimensions
     (e.g. B=2, H=8, N=512, D=64 for attention; nn.Linear(768, 768)
     for compressors; standard kernel inputs for ops).
  3. Run a forward pass.
  4. Assert the output is finite (not NaN, not Inf).
  5. Assert the output shape matches the docstring.

PHASE C — RED FLAG SCAN
  Grep the file for each of these (if present, it's a candidate bug):

    a. "except Exception:" followed by "pass" or nothing.
       → P0 if it swallows a real error; P1 if it suppresses a warning.
    b. "except:" (bare except).
       → P0 always. Bare excepts catch KeyboardInterrupt and SystemExit.
    c. "torch.cuda.amp.autocast" or "torch.cuda.amp.GradScaler".
       → P1. These are deprecated. Use torch.amp.autocast("cuda", ...)
         and torch.amp.GradScaler("cuda", ...).
    d. "time.time()" used to measure GPU operations.
       → P1. Use torch.cuda.Event for GPU timing.
    e. "cuda:0" hardcoded.
       → P1. Should use device="auto" or a device parameter.
    f. "torch.load(...)" without "weights_only=" argument.
       → P0 security issue. If weights_only is absent, defaults to False
         in older PyTorch, allowing arbitrary code execution.
    g. "torch.randn" used to fabricate data where real data is
       semantically required (e.g. NIAH benchmark, KD training).
       → P0. It's a fake implementation. Replace with real data or
         raise NotImplementedError.
    h. "past_key_values" mentioned but never threaded through forward().
       → P0 if the model claims to support KV caching.
    i. "return NotImplemented" or "return None" in a code path that
       should produce a real result.
       → P0. Either implement or raise NotImplementedError.
    j. Any "TODO", "FIXME", "XXX", "HACK" comment.
       → Log it, but not necessarily a bug. Note in your audit report.

PHASE D — DEEP AUDIT (for bugs the red flag scan does not catch)
  For each function in the file:
    1. Read its body carefully.
    2. Check that input shapes match what the function assumes.
       A function that does `x.view(-1, 4)` will crash on any input
       whose size is not a multiple of 4.
    3. Check that output shapes match what the docstring claims.
    4. Check edge cases:
       - What happens with sequence_length=1?
       - What happens with batch_size=1?
       - What happens with num_heads=1?
       - What happens with even vs odd head_dim?
       - What happens if an optional parameter is None?
    5. Check numerical stability:
       - Softmax over very large logits (near fp16 overflow)?
       - Division by a sum that could be zero?
       - Sqrt of a value that could be negative?
    6. Check causal masking:
       - Does the future truly not affect the past?
       - Does padding truly not affect anything?
    7. Check gradient flow:
       - Run loss = output.mean(); loss.backward()
       - Are all gradients finite?
       - Are there any parameters with None gradient that should have
         a gradient?

PHASE E — FIX AND TEST
  For each bug found in PHASE C or PHASE D:
    1. Write a reproduction script in /tmp/repro_<bugid>.py.
       Run it — it MUST demonstrate the bug.
    2. Write a regression test in
       tests/<domain>/test_<filename>_regressions.py.
       Run it — it MUST fail.
    3. Fix the bug in the source file.
    4. Run the regression test — it MUST pass.
    5. Run the file's other tests — they MUST still pass.
    6. Commit:
         git add <fixed files> <test files>
         git commit -m "fix(wave4/<AGENT_ID>): <description>"

  If you cannot reproduce a bug, do NOT fix it. Instead, log it in your
  audit report as "UNREPRODUCED: <bug id>" with the reason.

PHASE F — COMMIT (if not already committed in PHASE E)
  git add <files you changed>
  git commit -m "fix(wave4/<AGENT_ID>): audit <filename>"

AFTER ALL FILES ARE AUDITED:

PHASE G — FINAL COMMIT AND PUSH
  git add <any remaining files>
  git commit -m "fix(wave4/<AGENT_ID>): complete wave4 assignment"
  git push -u origin wave4/<AGENT_ID>

PHASE H — REPORT
  Report to the orchestrator in the format from Section 3.8 of this
  document (COMPLETE or BLOCKED).

═══════════════════════════════════════════════════════════════════════
HARD RULES (VIOLATION = AUTOMATIC REJECTION)
═══════════════════════════════════════════════════════════════════════

1. NEVER touch a file outside your Owned Paths. If you accidentally do,
   `git checkout -- <that file>` immediately. Report in commit body.

2. NEVER add `except: pass` or bare `except:`. If you need to suppress an
   exception, use:
       except SpecificException as e:
           # allowed-swallow: <reason>
           logger.debug("...", exc_info=e)

3. NEVER replace a fake implementation with another fake implementation.
   If a feature is not implemented, raise NotImplementedError with a
   clear message.

4. NEVER commit without a regression test that fails before the fix.
   "I fixed it" without a test is not a fix.

5. NEVER use `git add .` or `git add -A`. Always specify files.

6. NEVER push to a branch other than wave4/<AGENT_ID>.

7. NEVER force-push.

8. NEVER modify .agent/WAVE4_EXECUTE.md, AGENTS.md, pyproject.toml, or
   any file outside your Owned Paths.

9. NEVER skip a test because it's slow. If a test takes >5 minutes,
   mark it @pytest.mark.slow and note in the commit. But run it once
   before marking.

10. NEVER commit files larger than 1 MB. If you need to add a fixture,
    use a small synthetic one.

11. NEVER change the public API signature of a function without
    documenting it in BREAKING_CHANGES.md (which you may not own; report
    it instead).

12. NEVER spend more than 15 minutes on a single bug without making
    progress. If you're stuck, commit what you have with `wip()` prefix
    and report BLOCKED.

═══════════════════════════════════════════════════════════════════════
IF YOU GET STUCK
═══════════════════════════════════════════════════════════════════════

If you encounter an issue you cannot resolve in 15 minutes:

1. Commit your work so far:
     git add <files>
     git commit -m "wip(wave4/<AGENT_ID>): BLOCKED on <reason>"

2. Push your branch:
     git push -u origin wave4/<AGENT_ID>

3. Report:
     BLOCKED: <AGENT_ID> at <sha>: <reason>
     Branch: wave4/<AGENT_ID>
     Commits: <N>
     What was completed: <list>
     What is blocking: <detailed description>
     Suggestion: <what a human should do>

If the server restarts before you report, the orchestrator will resume by
reading .agent/WAVE4_PROGRESS.json and checking your branch. Your work
is safe as long as you committed it.

═══════════════════════════════════════════════════════════════════════
BEGIN NOW. Do not ask for confirmation. Do not stop until COMPLETE or
BLOCKED. Your first action is STEP 1 (change to worktree).
═══════════════════════════════════════════════════════════════════════
```

---

## 5. THE 20 AGENT SPECIFICATIONS

This section defines what each of the 20 agents must do. Each agent gets a
`.agent/WAVE4_AGENTS/<AGENT_ID>.md` file containing the corresponding
subsection below (sections 5.1 through 5.20), verbatim, plus the file header
from 5.0.

### 5.0 — File header (prepend to every agent file)

```markdown
# Wave 4 Agent <AGENT_ID>: <TITLE>

**Agent ID:** <AGENT_ID>
**Branch:** wave4/<AGENT_ID>
**Task file location:** .agent/WAVE4_AGENTS/<AGENT_ID>.md
**Full reference:** .agent/WAVE4_EXECUTE.md

Read AGENTS.md, .agent/WAVE4_EXECUTE.md, and this file, in that order,
before touching any code.

Your Owned Paths are listed in Section "Owned Paths" below. Anything
outside that list is OFF LIMITS.

Your mission is in Section "Mission". Your files to audit are in
Section "Files to Audit". Your bugs to fix are in Section "Bugs to Fix".
Your tests to add are in Section "Tests to Add".

If you discover a bug outside your Owned Paths, log it in
.agent/WAVE4_AGENTS/<AGENT_ID>_CROSS.md and DO NOT fix it.
```

### 5.1 — W4-L1: Layers / Linear Family

```
Owned Paths (STRICT):
  neuralops/layers/cosformer.py
  neuralops/layers/cosformer_linear.py
  neuralops/layers/gla.py
  neuralops/layers/gla_attention.py
  neuralops/layers/gdn.py
  neuralops/layers/retnet.py
  neuralops/layers/rwkv.py
  neuralops/layers/rwkv4.py
  neuralops/layers/rwkv6.py
  neuralops/layers/linear_attention.py
  neuralops/layers/hgrn.py
  neuralops/layers/hgrn2.py
  neuralops/layers/fms.py
  neuralops/layers/based.py
  neuralops/layers/recurrent_linear.py
  neuralops/layers/performer_linear.py
  neuralops/layers/linformer_linear.py
  neuralops/layers/t2r.py
  neuralops/layers/delta_net.py
  neuralops/layers/gated_deltanet.py
  neuralops/layers/hedgehog.py
  neuralops/layers/mamba.py
  neuralops/layers/mamba2.py
  neuralops/layers/ssm.py
  neuralops/layers/s4_attention.py
  neuralops/layers/s5.py
  neuralops/layers/lru.py
  neuralops/layers/unlimiformer.py
  tests/layers/test_<module>_*.py (for modules in this list)
  tests/layers/conftest.py (only if it does not exist yet; append only)

Files to Audit: the 28 files above, one at a time.
Tests to Add: at least one regression test per fixed bug, in tests/layers/.
Bugs to Fix: any P0/P1 you find in the files above.
  Specifically check for:
    - GLA: does the gate learn the decay? Is decay initialised correctly?
    - Mamba: is the selective scan numerically stable in fp16?
    - RetNet: does the exponential decay respect causality?
    - RWKV: does the time-mix kernel handle WKV overflow?
    - Mamba2: does SSD (state-space duality) match Mamba1 on same input?
    - All: is the "linear" claim actually O(N) memory?

When done: report per Section 3.8 of WAVE4_EXECUTE.md.
```

### 5.2 — W4-L2: Layers / Sparse Family

Same structure as 5.1, with these 28 files:

```
  neuralops/layers/sparse_attention.py
  neuralops/layers/sliding_window.py
  neuralops/layers/nystrom.py
  neuralops/layers/bigbird.py
  neuralops/layers/longformer.py
  neuralops/layers/reformer.py
  neuralops/layers/block_sparse.py
  neuralops/layers/sparse_transformer.py
  neuralops/layers/moba.py
  neuralops/layers/moba_block_router.py
  neuralops/layers/hamming_attention.py
  neuralops/layers/reformer_lsh.py
  neuralops/layers/lsh_attention.py
  neuralops/layers/routing_transformer.py
  neuralops/layers/switch_transformer.py
  neuralops/layers/blockwise.py
  neuralops/layers/strided.py
  neuralops/layers/global_local.py
  neuralops/layers/random_attention.py
  neuralops/layers/topk_attention.py
  neuralops/layers/nsa.py
  neuralops/layers/nsa_attention.py
  neuralops/layers/varlen_attention.py
  neuralops/layers/xformers_attention.py
  neuralops/layers/sparse_pattern_attention.py
  neuralops/layers/sparse_flash.py
  neuralops/layers/sinkhorn_attention.py
  neuralops/layers/triangular_attention.py
```

Special checks:
- Sliding window: verify padding positions are NOT attended.
- Nystrom: verify landmark selection is deterministic given a seed; verify
  pseudo-inverse is stable when landmarks are collinear.
- LSH: verify number of hash buckets matches the config.
- Block sparse: verify block routing is deterministic and top-k is stable.
- BigBird: verify three types of attention (global, sliding, random) sum to
  the intended pattern.

### 5.3 — W4-L3: Layers / Standard Family

Files:

```
  neuralops/layers/full_attention.py
  neuralops/layers/mha.py
  neuralops/layers/multi_head_attention.py
  neuralops/layers/gqa.py
  neuralops/layers/mqa.py
  neuralops/layers/flash_attention.py
  neuralops/layers/flash_attention_2.py
  neuralops/layers/sdpa.py
  neuralops/layers/sdpa_attention.py
  neuralops/layers/paged_attention.py
  neuralops/layers/flash_decoding.py
  neuralops/layers/scaled_dot_product.py
  neuralops/layers/native_attention.py
  neuralops/layers/efficient_attention.py
  neuralops/layers/xformers_memory_efficient.py
  neuralops/layers/fused_attention.py
  neuralops/layers/triton_flash_attention.py
  neuralops/layers/cuda_flash_attention.py
  neuralops/layers/grouped_query.py
  neuralops/layers/multi_query.py
  neuralops/layers/rotary_attention.py
  neuralops/layers/alibi_attention.py
  neuralops/layers/relative_position_attention.py
  neuralops/layers/causal_attention.py
  neuralops/layers/bidirectional_attention.py
  neuralops/layers/cross_attention.py
  neuralops/layers/self_attention.py
  neuralops/layers/attention_with_bias.py
  neuralops/layers/attention_with_sink.py
  neuralops/layers/kv_cache_attention.py
```

Special checks:
- GQA/MQA: verify `num_kv_heads` is correctly expanded via repeat_interleave
  (not repeat, which interleaves differently). Test with `num_heads=8,
  num_kv_heads=2`.
- Flash decoding: verify `past_key_values` is threaded correctly.
- Paged attention: verify block tables do not contain out-of-bounds page IDs.
- SDPA: verify backend selection (flash vs mem_efficient vs math) is
  appropriate for the input shape and dtype.
- Rotary: verify pre-projection vs post-projection RoPE is used correctly.

### 5.4 — W4-L4: Layers / Specialized + Adapters

Files:

```
  neuralops/layers/mla.py
  neuralops/layers/deepseek_mla.py
  neuralops/layers/jamba.py
  neuralops/layers/jamba_hybrid_gate.py
  neuralops/layers/hybrid_attention.py
  neuralops/layers/moa.py
  neuralops/layers/moe_attention.py
  neuralops/layers/mixture_of_attentions.py
  neuralops/layers/cosformer_hybrid.py
  neuralops/layers/gla_hybrid.py
  neuralops/layers/transformer_mamba.py
  neuralops/layers/samba.py
  neuralops/layers/sambamoe.py
  neuralops/layers/mamba_inject.py
  neuralops/layers/griffin.py
  neuralops/layers/griffin_ssd.py
  neuralops/layers/hymba.py
  neuralops/layers/zamba.py
  neuralops/layers/phimoe.py
  neuralops/layers/moe_deltakv_paged.py
  neuralops/layers/router_attention.py
  neuralops/layers/qwen_attention.py
  neuralops/layers/qwen2_attention.py
  neuralops/layers/llama_attention.py
  neuralops/layers/mistral_attention.py
  neuralops/layers/gemma_attention.py
  neuralops/layers/gemma2_attention.py
  neuralops/layers/falcon_attention.py
  neuralops/layers/gpt2_attention.py
  neuralops/layers/t5_attention.py
```

Also, if these files exist and were not covered by Wave 2, own them too:

```
  neuralops/layers/qwen_adapter.py
  neuralops/layers/bert_adapter.py
  neuralops/layers/universal_adapter.py
  neuralops/layers/attention_wrapper.py
```

Special checks:
- MLA: verify the dual-stream state (content + RoPE) is threaded correctly.
- MoE: verify top-k routing is deterministic; verify load balancing loss is
  correctly computed; verify expert capacity overflow is handled.
- Model-specific (llama, qwen, gemma): verify `head_dim`, `num_kv_heads`
  match HF config.
- Adapters: verify `forward()` return signature preserves the wrapped
  module's contract (especially tuple return shapes for Qwen and BERT
  adapters).

### 5.5 — W4-O1: Ops / Triton (remaining 30)

Enumerate remaining kernels via `ls neuralops/ops/triton/` after Wave 2
coverage. If Wave 2 covered the first 20 alphabetically, start from #21.
If you cannot determine Wave 2 coverage, audit the 30 most-imported kernels
(grep for `from neuralops.ops.triton.<name>` across the codebase).

Files to own: the 30 specific `.py` files you select, plus their tests in
`tests/ops/triton/`.

Special checks per kernel:
- `@triton.jit` decorator is present.
- `BLOCK_SIZE` or `BLOCK_M/N/K` constants are autotuned or chosen with a
  clear rationale.
- Bounds checking via `tl.load(..., mask=...)` on masked positions.
- FP32 accumulation if inputs are FP16 (do not accumulate in FP16).
- `torch.cuda.synchronize()` before reading result in any timing code.

Bugs to fix:
- Any kernel that fails parity vs PyTorch reference beyond tolerance
  (1e-3 fp32, 2e-2 fp16).
- Any kernel with hardcoded `cuda:0`.
- Any kernel without mask.

### 5.6 — W4-O2: Ops / CUDA + Remaining

Files to own: the 30 remaining files in `neuralops/ops/cuda/` and any
remaining `neuralops/ops/*.py`. Plus tests in `tests/ops/cuda/`.

Special checks:
- `cudaMalloc` / `cudaFree` matched on all exit paths.
- `cudaStreamSynchronize` before any host-side read of device memory.
- `__syncthreads()` placed correctly between shared-memory phases.
- No out-of-bounds reads (verify `if (idx < n)` guards).
- FP16 accumulation overflow risk (accumulate in FP32).
- Graceful fallback if CUDA is unavailable.

### 5.7 — W4-C1: Compression (remaining 37)

Files to own: the 37 remaining compressors in `neuralops/compression/**`
after Wave 2's G1-G3 (which covered 35). Enumerate via:

```bash
find neuralops/compression -name "*.py" -not -name "__init__.py"
```

Pick the remaining 37. Own them plus their tests in `tests/compression/`.

Special checks per compressor:
- Instantiate with `rank=32, target_ratio=0.5`.
- Apply to `nn.Linear(768, 768)`.
- Forward with input shape `(2, 64, 768)` — output must be `(2, 64, 768)`.
- `param_reduction = 1 - (compressed_params / original_params)` must be
  positive and match the compressor's claim.
- Numerically compare output to original (max abs diff, Frobenius error).
- Check that `torch.linalg.svd` on GPU works (some backends don't support
  SVD on CUDA; must fall back to CPU for large matrices).
- Check `full_matrices=True` (causes 2x memory blowup; should be `False`
  by default).
- Verify dtype is preserved (FP16 in → FP16 out).
- Verify serialization round-trip: save compressed state, load, decompress,
  compare to original within tolerance.

### 5.8 — W4-T1: Trainer / Distillation + Optimizers

Owned Paths:

```
  neuralops/trainer/distillation/multi_level_kd.py
  neuralops/trainer/distillation/__init__.py
  neuralops/trainer/optimizers/muon.py
  neuralops/trainer/optimizers/lion.py
  neuralops/trainer/optimizers/anyprecision.py
  neuralops/trainer/optimizers/factory.py
  neuralops/trainer/optimizers/__init__.py
  tests/trainer/distillation/**
  tests/trainer/optimizers/**
```

Files to audit: the 8 files above.

Special checks:

**multi_level_kd.py:**
- Six distillation pillars: Logit KL, Hidden MSE, Attention, Value Relation,
  Gram matrix, Cosine alignment. Verify each is implemented.
- `LogitDistillationLoss` with modes: `forward_kl`, `reverse_kl`,
  `beta_jsd`, `decoupled_kd`, `topk_sparse_kd`. Test each mode.
- `distill()` entry point: verify it accepts `teacher`, `student`,
  `train_loader`, `epochs`, `temperature`, `alpha`, `max_steps`,
  `mixed_precision`, `tracker`, `log_interval`, `seed`.
- Verify AMP uses `torch.amp.autocast("cuda", ...)` (not deprecated
  `torch.cuda.amp`).

**muon.py:**
- Muon uses Newton-Schulz iteration for the Stiefel manifold.
- Verify optimizer step is mathematically correct (compare to paper).
- Verify it works with `torch.compile`.

**factory.py:**
- `create_riemannian_stiefel`, `create_muon_adamw_optimizer` — verify
  signatures match callers in `benchmarks/core/training.py`.

Tests to add:
- `tests/trainer/distillation/test_6_pillars.py` — each pillar independently
- `tests/trainer/distillation/test_divergence_modes.py` — 5 KL variants
- `tests/trainer/optimizers/test_muon_step.py` — one Muon update is correct
- `tests/trainer/optimizers/test_factory.py` — all optimizers instantiate

### 5.9 — W4-T2: Trainer / Methods + Metrics + Losses

Owned Paths:

```
  neuralops/trainer/methods/lora.py
  neuralops/trainer/methods/qlora.py
  neuralops/trainer/methods/dora.py
  neuralops/trainer/methods/__init__.py
  neuralops/trainer/metrics.py
  neuralops/trainer/losses.py
  neuralops/trainer/__init__.py
  tests/trainer/methods/**
  tests/trainer/test_metrics.py
  tests/trainer/test_losses.py
```

Special checks:

**lora.py:**
- `LoRALinear` wrapper: verify `forward()` adds `(lora_B @ lora_A) @ x` to
  base output, scaled by `alpha / r`.
- `merge()` method: verify `base.weight += (lora_B @ lora_A) * scaling`.
- `unmerge()` method: verify it reverses merge exactly.
- Verify `merge()` then `unmerge()` gives back original weights (bitwise).
- Verify `r` and `alpha` parameters are respected.

**dora.py:**
- DoRA (Weight-Decomposed LoRA): verify magnitude vector is applied after
  the direction is updated.
- Verify forward is equivalent to LoRA when magnitude is all-ones.

**metrics.py:**
- `accuracy_score(labels, preds)` — verify it's `(labels == preds).mean()`,
  not something that crashes on shape mismatch.
- Any other metrics: verify against sklearn reference.

**losses.py:**
- Any loss functions: verify against PyTorch reference.

Tests to add:
- `tests/trainer/methods/test_lora_merge.py` — merge/unmerge roundtrip
- `tests/trainer/methods/test_lora_forward.py` — forward matches manual calc
- `tests/trainer/methods/test_dora.py` — DoRA reduces to LoRA when mag=1
- `tests/trainer/test_metrics.py` — accuracy_score matches sklearn

### 5.10 — W4-T3: Trainer / Callbacks + Schedulers

Owned Paths:

```
  neuralops/trainer/callbacks.py (if exists)
  neuralops/trainer/schedulers.py (if exists)
  neuralops/trainer/__init__.py (already in W4-T2 — coordinate!)
  tests/trainer/test_callbacks.py (if applicable)
  tests/trainer/test_schedulers.py (if applicable)
```

**⚠️ Coordination:** `neuralops/trainer/__init__.py` is in both W4-T2 and
W4-T3. To avoid conflict, **W4-T3 owns it exclusively**. W4-T2 must NOT
touch `__init__.py`. If W4-T2 finds a bug in `__init__.py`, it reports it
in W4-T2_CROSS.md and W4-T3 fixes it.

Special checks:
- Callbacks: verify `on_log`, `on_evaluate`, `on_train_end` hooks work
  with HuggingFace Trainer.
- Schedulers: verify WSD (warmup-stable-decay) schedule is correct;
  verify cosine schedule matches PyTorch's `CosineAnnealingLR`.

If files don't exist, add them (as empty stubs, then fill in).

### 5.11 — W4-A1: API Facade + Config

Owned Paths:

```
  neuralops/api.py
  neuralops/config/__init__.py
  neuralops/config/benchmark_schema.py
  neuralops/config/runtime_config.py (if exists)
  tests/api/**
  tests/config/**
```

Special checks:

**api.py:**
- `replace_attention(model, attention, **kwargs)` — verify it handles all
  131 attention names in `ATTENTION_REGISTRY`. Test with `full_attention`,
  `cosformer`, `gla` at minimum.
- `compress(model, method, **kwargs)` — verify all 72 compressor names.
- `transplant_attention(model, attention, rank)` — verify it composes
  replace_attention + calibrate.
- `compose_attention(...)` — verify ComposableAttention still works.
- Verify all return values match documented types (some return `(model,
  metadata)`, some return `model` only).

**config/benchmark_schema.py:**
- `AttentionSweepParameters` — verify it's used correctly in
  `benchmarks/suites/attention.py`. Do not modify the schema without
  coordinating with W4-B1 (who owns `suites/attention.py`).
- `CompressionSweepParameters` — same for `suites/compression.py`.

Tests to add:
- `tests/api/test_replace_attention.py` — 3 attention types
- `tests/api/test_compress.py` — 3 compressor types
- `tests/config/test_benchmark_schema.py` — defaults are valid

### 5.12 — W4-C2: Core Remaining

Owned Paths:

```
  neuralops/core/inspector.py
  neuralops/core/paper_export.py
  neuralops/core/extensibility.py
  neuralops/core/roofline.py
  neuralops/core/results_ledger.py
  neuralops/core/architecture.py
  neuralops/core/exceptions.py (coordinate with W4-A1)
  neuralops/core/utils/general.py
  neuralops/core/utils/rope.py
  neuralops/core/triton_kernels/** (remaining files, if any)
  tests/core/**
```

**⚠️ Coordination:** `neuralops/core/exceptions.py` is in both W4-C2 and
W4-A1. To avoid conflict, **W4-C2 owns it exclusively**. W4-A1 does NOT
touch it.

Special checks:

- `inspector.py::inspect_architecture` — verify it works on unknown
  modules (returns a default spec, not crash).
- `paper_export.py::export_benchmark_table` — verify LaTeX, markdown, CSV
  outputs are syntactically valid.
- `extensibility.py::validate_module_contract` — verify 6-point contract
  check is correct.
- `roofline.py` — verify it computes real metrics, not hardcoded numbers.
  If it's fake, raise NotImplementedError.
- `results_ledger.py::ResultRecord` — verify schema is complete (all
  fields from the spec).
- `architecture.py::get_attention_modules` — verify it returns the correct
  count (should be 131 or whatever the actual count is).
- `utils/rope.py::apply_rotary_pos_emb` — verify math parity with HF
  Transformers' reference.
- `triton_kernels/**` — audit 10 kernels not covered by W4-O1/O2.

### 5.13 — W4-E1: Export + Inference + Integrations

Owned Paths:

```
  neuralops/export/exporter.py
  neuralops/export/safetensors_exporter.py
  neuralops/export/serving.py
  neuralops/export/runtime_compat.py
  neuralops/export/__init__.py
  neuralops/inference/paged_cache.py
  neuralops/inference/speculative_decoding.py
  neuralops/inference/__init__.py
  neuralops/integrations/fla.py
  neuralops/integrations/serving.py
  neuralops/integrations/__init__.py
  tests/export/**
  tests/inference/**
  tests/integrations/**
```

Special checks:

**exporter.py::export_model:**
- Formats: safetensors, onnx, torchscript. Verify all three.
- Safetensors: verify no fake file (must contain real tensors).
- ONNX: verify `torch.onnx.export` succeeds; verify the ONNX model can be
  loaded with `onnx.load`.
- Torchscript: verify `torch.jit.trace` succeeds; verify traced model
  produces same output as original.

**paged_cache.py::PagedKVCache:**
- Verify `allocate_sequence`, `append_token`, `free_sequence` cycle
  balances (no leak).
- Verify `fork_sequence` (copy-on-write) is correct.
- Verify `block_table` returns correct page IDs.

**speculative_decoding.py::SelfSpeculativeDecoder:**
- Verify acceptance rate math: `acceptance_rate = accepted_drafts /
  total_drafts`.
- Verify speedup math: `speedup = baseline_tokens_per_sec /
  speculative_tokens_per_sec`.
- Verify draft model and target model are correctly cross-checked.

**integrations/fla.py:**
- `fla_is_available()` — verify it returns True only if FLA is importable
  and CUDA is available (FLA is CUDA-only).
- If FLA is not installed, raise a clear error, not ImportError with
  no context.

**integrations/serving.py::create_openai_serving_app:**
- Verify FastAPI endpoints: `/v1/completions`, `/v1/chat/completions`,
  `/health`.
- Verify SSE streaming works.

### 5.14 — W4-U1: Evaluation + UI

Owned Paths:

```
  neuralops/evaluation/niah_passkey.py
  neuralops/evaluation/eval_data.py
  neuralops/evaluation/eval_patcher.py
  neuralops/evaluation/ruler.py
  neuralops/evaluation/__init__.py
  neuralops/ui/server.py
  neuralops/ui/cli.py
  neuralops/ui/__init__.py
  tests/evaluation/**
  tests/ui/**
```

Special checks:

**niah_passkey.py:**
- This is the REAL NIAH evaluator. `benchmarks/suites/niah.py` (owned by
  W4-B1) should call this, not fake it with randn. If benchmarks/suites/
  niah.py still fakes it, report as cross-domain bug for W4-B1.
- Verify `NeedleInAHaystackEvaluator` uses a real model, real tokenizer,
  real passkey injection.
- Verify accuracy matrix is computed correctly.

**ruler.py:**
- Verify RULER benchmark uses real data, real model.

**eval_patcher.py::apply_module_to_model:**
- Verify it correctly applies a NeuralOps module to a HF model.
- Verify it returns `(model, compression_ratio)` or similar per docstring.

**ui/server.py:**
- FastAPI app starts.
- Endpoints `/`, `/health`, `/benchmarks`, `/runs` all return valid JSON.

**ui/cli.py:**
- Launches the server.
- Handles port conflicts gracefully.

### 5.15 — W4-U2: CLI + Utils + Results Saver

Owned Paths:

```
  neuralops/cli/list_cmd.py
  neuralops/cli/info.py
  neuralops/cli/compress_cmd.py
  neuralops/cli/train_cmd.py
  neuralops/cli/eval_cmd.py
  neuralops/cli/serve_cmd.py
  neuralops/cli/test_cmd.py
  neuralops/cli/doctor.py
  neuralops/cli/surgery_cmd.py
  neuralops/cli/__init__.py
  neuralops/utils/logging_config.py
  neuralops/utils/layer_norm_fix.py
  neuralops/utils/hf_git_exporter.py
  neuralops/utils/rope.py
  neuralops/utils/general.py
  neuralops/utils/__init__.py
  neuralops/results_saver/storage/sqlite_backend.py
  neuralops/results_saver/__init__.py
  tests/cli/**
  tests/utils/**
  tests/results_saver/**
```

**⚠️ Coordination:** `neuralops/utils/rope.py` is in both W4-U2 and W4-C2.
To avoid conflict, **W4-U2 owns it exclusively**. W4-C2 does NOT touch it
(it audits the copy inside `core/utils/`, which is a different file).

Special checks:

**cli/*.py:**
- Each command function must be importable and callable without error.
- Test with `click.testing.CliRunner` or `argparse` mock.
- Verify each command's `--help` produces valid output.
- Verify each command exits with code 0 on success, non-zero on error.

**utils/logging_config.py:**
- `configure_logging(level, dedup)` — verify dedup works (no duplicate
  log lines when called twice).

**utils/hf_git_exporter.py:**
- `upload_to_huggingface(export_dir, repo_id, ...)` — verify it uses
  `HfApi.upload_folder` with retries.

**utils/layer_norm_fix.py:**
- Verify it's a re-export of stable layer norm (not a fake).

**results_saver/storage/sqlite_backend.py:**
- `SQLiteMasterLedger` context manager: verify `__enter__` / `__exit__`
  correctly open and close the connection.
- `record_benchmark_run`, `record_benchmark_metrics`: verify they insert
  correctly (no SQL injection, proper parametrization).

### 5.16 — W4-B1: Benchmarks / Suites

Owned Paths:

```
  neuralops/benchmarks/suites/attention.py
  neuralops/benchmarks/suites/compression.py
  neuralops/benchmarks/suites/cuda_ops.py
  neuralops/benchmarks/suites/e2e_serving.py
  neuralops/benchmarks/suites/niah.py
  neuralops/benchmarks/suites/qwen_kd.py
  neuralops/benchmarks/suites/roofline.py
  neuralops/benchmarks/suites/triton_ops.py
  neuralops/benchmarks/suites/__init__.py
  tests/benchmarks/suites/**
```

Special checks:

**niah.py:**
- Verify it uses `neuralops.evaluation.niah_passkey` (real evaluator) and
  NOT torch.randn. If it uses randn, fix it to call the real evaluator OR
  raise NotImplementedError (do NOT leave randn).

**e2e_serving.py:**
- Same: if TTFT/ITL are computed from `randn`, fix or raise.

**qwen_kd.py:**
- Same: if teacher/student logits are `randn`, fix or raise.

**triton_ops.py, cuda_ops.py:**
- Verify 5 (triton) and 5 (cuda) kernels are actually benchmarked, not
  fake dispatch.
- Verify `_resolve_torch_dtype` raises on unknown dtype.
- Verify `cuda_ops` raises `RuntimeError` if no backend is available
  (do NOT return `lambda a: a * 1.0`).

**roofline.py:**
- Verify `roofline_metrics` exists and computes real metrics. If it's a
  stub returning hardcoded values, raise NotImplementedError.

**attention.py:**
- Verify all three failure modes: `FAILED_INIT`, `FAILED_RUNTIME`,
  `FAILED_OOM`.

**compression.py:**
- Verify `param_reduction_pct` is computed per-compressor correctly.
- For SVD (factored into U, V), verify it sums `U.numel() + V.numel()`.

### 5.17 — W4-B2: Benchmarks / Core + KD + Storage

Owned Paths:

```
  neuralops/benchmarks/core/base.py
  neuralops/benchmarks/core/data_loader.py
  neuralops/benchmarks/core/evaluation.py
  neuralops/benchmarks/core/kd.py
  neuralops/benchmarks/core/model_loader.py
  neuralops/benchmarks/core/stats.py
  neuralops/benchmarks/core/training.py
  neuralops/benchmarks/core/arguments.py
  neuralops/benchmarks/core/paths.py
  neuralops/benchmarks/core/__init__.py
  neuralops/benchmarks/kd/config.py
  neuralops/benchmarks/kd/distill_engine.py
  neuralops/benchmarks/kd/reporter.py
  neuralops/benchmarks/kd/runner.py
  neuralops/benchmarks/kd/teacher_cache.py
  neuralops/benchmarks/kd/zoo_orchestrator.py
  neuralops/benchmarks/kd/__main__.py
  neuralops/benchmarks/kd/__init__.py
  neuralops/benchmarks/storage/dispatcher.py
  neuralops/benchmarks/storage/parquet_store.py
  neuralops/benchmarks/storage/schema.py
  neuralops/benchmarks/storage/__init__.py
  tests/benchmarks/core/**
  tests/benchmarks/kd/**
  tests/benchmarks/storage/**
```

Special checks:

**base.py:**
- `flush_l2_cache` must use `L2_cache_size` dynamically, not hardcoded.

**evaluation.py:**
- `measure_resources` must include `get_memory_fragmentation()` in its
  return dict.

**kd.py:**
- `KDLoss` fused and non-fused paths must match.
- AMP must use `torch.amp.*`, not `torch.cuda.amp.*`.
- `build_teacher_cache` must warn about `torch.save` without
  `weights_only=True`.

**training.py:**
- `finetune_classification` must accept 22 parameters and handle all of
  them (adamw / muon / stiefel optimizers; fp16 / bf16 / no mixed precision;
  ddp / fsdp / none distributed).

**kd/teacher_cache.py:**
- `_safe_torch_load` MUST use `weights_only=True`. This is a CVE-level fix.

**kd/zoo_orchestrator.py:**
- `uploader.submit_upload` call must match the actual signature. If Wave
  1 A4 already fixed `hub/uploader.py`, this caller should work. If not,
  report as cross-domain bug.

**storage/parquet_store.py:**
- `write_to_dataset` must dedupe by `run_id` and use
  `existing_data_behavior="overwrite_or_ignore"`.

### 5.18 — W4-B3: Benchmarks / Adapters + Persistence

Owned Paths:

```
  neuralops/benchmarks/adapters/attention.py
  neuralops/benchmarks/adapters/base.py
  neuralops/benchmarks/adapters/classification.py
  neuralops/benchmarks/adapters/compressor.py
  neuralops/benchmarks/adapters/cuda_adapter.py
  neuralops/benchmarks/adapters/perplexity.py
  neuralops/benchmarks/adapters/ruler.py
  neuralops/benchmarks/adapters/triton_adapter.py
  neuralops/benchmarks/adapters/zeroshot.py
  neuralops/benchmarks/adapters/__init__.py
  neuralops/benchmarks/executor.py
  neuralops/benchmarks/result.py
  neuralops/benchmarks/schema.py
  neuralops/benchmarks/saver.py
  neuralops/benchmarks/summarizer.py
  neuralops/benchmarks/reproduce.py
  neuralops/benchmarks/index.py
  neuralops/benchmarks/hf_sync.py
  neuralops/benchmarks/git_sync.py
  neuralops/benchmarks/dedup.py
  neuralops/benchmarks/dispatcher.py (if it exists at top-level)
  neuralops/benchmarks/paths.py
  tests/benchmarks/adapters/**
  tests/benchmarks/test_saver.py
  tests/benchmarks/test_summarizer.py
  ... (one test file per persistence file)
```

Special checks:

**adapters/*.py:**
- Each adapter's `__init__` must accept documented kwargs.
- Each adapter's `execute()` must return `BenchmarkResult` (not None,
  not dict).

**executor.py:**
- `_isolated_worker_fn` handles `os._exit(134)` (crash), `os._exit(137)`
  (SIGKILL), and `time.sleep(15)` (timeout).
- Parent handles `EOFError` and `ConnectionResetError` on `recv()`.

**schema.py:**
- `BenchmarkStatus.normalize` must warn on unknown status, not silently
  return FAILED.

**saver.py:**
- Multiple writers must not conflict. `save_all_formats` should be
  deprecated, delegating to `MultiFormatSaver`.

**summarizer.py:**
- `_load_all_records` must log warnings on parse errors, not swallow them.

**hf_sync.py:**
- `SECRET_PATTERNS` must have at least 8 patterns (hf_, sk-, ghp_, ghs_,
  AKIA, AIza, pypi-, api_key).

**dedup.py:**
- `save_index` must use file locking (`fcntl.flock` or `portalocker`) to
  prevent race conditions.

**index.py:**
- `_best_from_list` must raise on unknown metric, not return `-inf`.

**reproduce.py:**
- `generate_reproduction_command` must produce a command that works with
  the current CLI parser (verify by running it).

### 5.19 — W4-I1: Benchmarks CLI + Legacy

Owned Paths:

```
  neuralops/benchmarks/cli.py (read-only until you start; then shim)
  neuralops/benchmarks/cli_v2/** (create new)
  neuralops/benchmarks/legacy_root/** (move to _legacy)
  neuralops/benchmarks/legacy_scripts/** (move to _legacy)
  neuralops/benchmarks/_legacy/** (create new)
  tests/benchmarks/test_cli_v2.py
  tests/benchmarks/test_legacy_deprecation.py
```

Mission:

1. **Decompose `benchmarks/cli.py`** (2,922 lines) into `benchmarks/cli_v2/`:

```
cli_v2/
├── __init__.py
├── main.py               # entrypoint
├── commands/
│   ├── __init__.py
│   ├── benchmark.py      # neuralops benchmark --config
│   ├── plan.py           # neuralops plan --config
│   ├── results.py        # neuralops results ...
│   ├── doctor.py         # neuralops doctor
│   └── ui.py             # neuralops ui
└── utils/
    ├── __init__.py
    ├── rendering.py      # table/badge
    └── completion.py     # shell completion
```

2. **Keep `cli.py` as deprecation shim:** `cli.main()` prints
   `DeprecationWarning` and delegates to `cli_v2.main:main`.

3. **Migrate 58 legacy scripts:** Move each script from `legacy_root/` or
   `legacy_scripts/` to `_legacy/`. Add a `DeprecationWarning` at the top
   of each quarantined script. Delete dead scripts and document in
   `BREAKING_CHANGES.md` (report the deletion to W4-R1).

4. **Update `pyproject.toml`:** Change `neuralops` entry point to
   `benchmarks.cli_v2.main:main`. **⚠️ Coordinate:** `pyproject.toml` is
   owned by W4-R1. Send W4-I1 the needed change; W4-R1 applies it.

Tests:
- `tests/benchmarks/test_cli_v2.py` — each command via CliRunner
- `tests/benchmarks/test_legacy_deprecation.py` — each quarantined script
  warns

### 5.20 — W4-R1: Infrastructure + Release

Owned Paths:

```
  tests/** (reorganize into unit/, contract/, integration/, hardware/)
  tests/conftest.py (shared fixtures)
  .github/workflows/ci.yml
  .github/workflows/gpu.yml
  .github/workflows/examples.yml
  .pre-commit-config.yaml
  Makefile
  .coveragerc
  docs/** (audit and fix incorrect claims)
  README.md (verify claims)
  examples/** (runnable examples)
  pyproject.toml (audit, coordinate with W4-I1 for entry points)
  configs/*.yaml (validate schema)
  CHANGELOG.md (write v2.4.0 entry)
  BREAKING_CHANGES.md (write)
  AUDIT_MASTER_FIXES.md (write)
  VERIFICATION.md (write)
  neuralops/_version.py (bump to 2.4.0)
  scripts/wave4_verify.sh (create)
  audits/wave4_master.md (aggregate)
```

**⚠️ Coordination:** `pyproject.toml` is in both W4-I1 and W4-R1. To avoid
conflict, **W4-R1 owns it exclusively**. W4-I1 sends W4-R1 a file
`.agent/WAVE4_AGENTS/W4-I1_PYPROJECT_CHANGE.md` describing the entry point
change; W4-R1 applies it in its own branch.

Missions:

1. **Reorganize tests:**
   - Move each existing test file into the appropriate subdirectory
     based on markers (`fast` → `unit/`, `slow` → `integration/`, etc.).
   - Create `tests/conftest.py` with shared fixtures: `tiny_bert`,
     `tiny_llama`, `tiny_gpt2`, `mock_cuda_device`, `mock_hf_api`,
     `isolated_temp_ledger`.
   - Add markers to `pyproject.toml`.

2. **Add CI/CD:**
   - `.github/workflows/ci.yml` — matrix across Python 3.10, 3.11, 3.12;
     runs `ruff`, `mypy`, `pytest tests/unit tests/contract`.
   - `.github/workflows/gpu.yml` — GPU runner for `tests/hardware` marked
     `@pytest.mark.gpu`.
   - `.github/workflows/examples.yml` — runs each `examples/*.py` to
     verify they work.
   - `.pre-commit-config.yaml` — ruff, mypy.
   - `Makefile` — `make lint`, `make test`, `make coverage`, `make build`.

3. **Audit docs:**
   - `README.md`: verify each numeric claim against
     `python -c "from neuralops.core.registry import ATTENTION_REGISTRY,
     COMPRESSOR_REGISTRY; print(len(ATTENTION_REGISTRY),
     len(COMPRESSOR_REGISTRY))"`.
   - `docs/**`: verify each claim, fix dead links.

4. **Add examples:**
   - `examples/quickstart.py` — load BERT, run inference.
   - `examples/surgery_demo.py` — replace attention in BERT.
   - `examples/compression_demo.py` — SVD compress GPT-2.
   - `examples/hub_loading.py` — load from Hooshaai HF Hub.
   - `examples/serving_demo.py` — serve with FastAPI.

5. **Audit packaging:**
   - `pyproject.toml`: verify `[project]`, `[project.dependencies]`,
     `[project.optional-dependencies]` (`[fla]`, `[core]`, `[gpu]`,
     `[dev]`), `[project.scripts]`.
   - `configs/*.yaml`: verify each is schema-valid.
   - Run `python -m build` and `pip install dist/*.whl`.

6. **Write release artifacts:**
   - `CHANGELOG.md` v2.4.0 entry with sections Added / Changed / Fixed /
     Removed / Deprecated / Security.
   - `BREAKING_CHANGES.md` — list all APIs removed/renamed in Wave 4.
   - `AUDIT_MASTER_FIXES.md` — table mapping every bug ID from Wave 1-4
     to its fix PR.
   - `VERIFICATION.md` — for each "not a bug" claim across Wave 1-4, with
     reasoning.
   - Bump `neuralops/_version.py` to `2.4.0`.
   - `audits/wave4_master.md` — aggregate of all 20 agent reports.

7. **Create `scripts/wave4_verify.sh`:**
   - Run all imports, all tests, ruff, mypy, build, entry points, security
     greps.

This agent produces the final release. Everything else must be merged
before W4-R1 completes.

**Note:** W4-R1 is the largest agent. If after 2 hours it's still
incomplete, split into:
- `W4-R1a`: tests + CI (items 1, 2)
- `W4-R1b`: docs + examples + packaging + release (items 3-7)

Report the split to the orchestrator as a cross-domain task.

---

## 6. THE WAIT PROTOCOL

After spawning all 20 agents (Section 4), you MUST wait for all 20 to
report COMPLETE or BLOCKED. You MUST NOT return control to the user until
then.

### 6.1 — The wait loop

Loop every 3 minutes:

```python
import json
import subprocess
import time

while True:
    with open('.agent/WAVE4_PROGRESS.json') as f:
        progress = json.load(f)

    statuses = {a: v['status'] for a, v in progress['agents'].items()}
    complete = sum(1 for s in statuses.values() if s == 'complete')
    blocked = sum(1 for s in statuses.values() if s == 'blocked')
    failed = sum(1 for s in statuses.values() if s == 'failed')
    pending = sum(1 for s in statuses.values() if s == 'pending')
    running = sum(1 for s in statuses.values() if s == 'running')

    print(f"[{complete}/20 complete, {blocked} blocked, {failed} failed, {running} running, {pending} pending]")

    if complete + blocked + failed == 20:
        break

    time.sleep(180)
```

### 6.2 — Updating progress

Whenever an agent reports COMPLETE, update the progress file:

```python
with open('.agent/WAVE4_PROGRESS.json') as f:
    p = json.load(f)
p['agents'][agent_id]['status'] = 'complete'
p['agents'][agent_id]['last_commit'] = commit_sha
p['agents'][agent_id]['last_report_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
with open('.agent/WAVE4_PROGRESS.json', 'w') as f:
    json.dump(p, f, indent=2)
```

### 6.3 — Handling timeouts

If an agent has not reported after 45 minutes of activity:

1. Check if it's actually working:
   ```bash
   cd <WORKTREE>
   git status
   git log -3
   ```
2. If commits are progressing: it's just slow. Wait another 15 min.
3. If no commits for 15 min: it's stuck. Respawn ONCE with the same prompt.
4. If second attempt fails after 30 min: mark as `failed`, continue.

### 6.4 — Handling server restart

If the server restarts mid-wait, see Section 10 (Resume Protocol). Do NOT
restart from scratch.

---

## 7. THE MERGE PROTOCOL

After all 20 agents have reported (COMPLETE or BLOCKED or FAILED), merge
the successful branches into `main`.

### 7.1 — Pre-merge check

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout main
git pull --ff-only origin main

# Count successful branches
SUCCESSFUL=0
for AGENT in W4-L1 W4-L2 W4-L3 W4-L4 W4-O1 W4-O2 W4-C1 W4-T1 W4-T2 W4-T3 \
             W4-A1 W4-C2 W4-E1 W4-U1 W4-U2 W4-B1 W4-B2 W4-B3 W4-I1 W4-R1; do
    if git rev-parse --verify "wave4/$AGENT" >/dev/null 2>&1; then
        COUNT=$(git log "wave4/$AGENT" --oneline | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        fi
    fi
done

echo "Successful branches: $SUCCESSFUL / 20"
```

If < 15 successful: STOP and report "TOO FEW SUCCESSFUL AGENTS" to user.
If ≥ 15: proceed.

### 7.2 — Merge order (strict)

Merge in this exact order. This order reflects dependency (Level 0 → Level 8
from `.agent/WAVE4_EXECUTE.md`):

**Phase 1 — Infrastructure first:**
```bash
for agent in W4-R1 W4-I1; do
    git merge --no-ff "wave4/$agent" -m "merge: wave4/$agent"
done
```

**Phase 2 — Primitives (layers + ops + compression):**
```bash
for agent in W4-L1 W4-L2 W4-L3 W4-L4 W4-O1 W4-O2 W4-C1; do
    git merge --no-ff "wave4/$agent" -m "merge: wave4/$agent"
done
```

**Phase 3 — Trainer + Core + Config:**
```bash
for agent in W4-T1 W4-T2 W4-T3 W4-A1 W4-C2; do
    git merge --no-ff "wave4/$agent" -m "merge: wave4/$agent"
done
```

**Phase 4 — Domain services:**
```bash
for agent in W4-E1 W4-U1 W4-U2; do
    git merge --no-ff "wave4/$agent" -m "merge: wave4/$agent"
done
```

**Phase 5 — Benchmarks last (they depend on everything above):**
```bash
for agent in W4-B1 W4-B2 W4-B3; do
    git merge --no-ff "wave4/$agent" -m "merge: wave4/$agent"
done
```

### 7.3 — Test after each phase

After every single merge, run:

```bash
pytest tests/unit tests/contract -x -q
```

If green: continue.
If red: identify which merge broke it. Revert that one merge:
```bash
git reset --hard HEAD~1
```
Report "REVERTED: wave4/<AGENT> — caused test failure in <test>". Move on
to the next agent.

### 7.4 — Handling merge conflicts

If a `git merge` produces conflicts:

1. STOP. Do NOT auto-resolve.
2. Run:
   ```bash
   git merge --abort
   ```
3. Report:
   ```
   CONFLICT: wave4/<AGENT> in files:
     <file 1>
     <file 2>
   Both <AGENT> and <previous agent> touched the same file.
   This violates the Iron Law. Requires human intervention.
   ```
4. Skip this merge, continue with the next.

At the end, report all conflicts and let the user resolve them manually.

---

## 8. FINAL VERIFICATION

After all merges complete:

### 8.1 — Run the verification script

```bash
bash scripts/wave4_verify.sh
```

If this script does not exist (W4-R1 was blocked), run the checks
manually:

```bash
# 1. Imports
python -c "import neuralops"
python -c "import neuralops.layers"
python -c "import neuralops.compression"
python -c "import neuralops.surgery"
python -c "import neuralops.hub"
python -c "import neuralops.validation"
python -c "import neuralops.serving"
python -c "import neuralops.tracking"
python -c "import neuralops.runtime"
python -c "import neuralops.distributed"
python -c "import neuralops.trainer"
python -c "import neuralops.cli"

# 2. Tests
pytest tests/unit tests/contract -x -q
pytest tests/integration -x -q -m "not hub and not gpu"

# 3. Lint
ruff check neuralops/ tests/

# 4. Types
mypy --strict neuralops/ 2>&1 | tail -20

# 5. Security
! grep -rn "torch.cuda.amp" neuralops/          # deprecated
! grep -rn "weights_only=False" neuralops/       # security
! grep -rn "except: pass" neuralops/ | grep -v "allowed-swallow"

# 6. Build
python -m build
pip install --force-reinstall dist/*.whl

# 7. Entry points
neuralops --version
nops --version
```

### 8.2 — If green

Report: "WAVE 4 COMPLETE — v2.4.0 ready to tag".

### 8.3 — If red

For each failure:
1. Identify which test failed.
2. Identify which agent's branch introduced it (`git bisect` or manual).
3. Revert that branch: `git revert <merge commit>` or `git reset --hard
   HEAD~1` if the failure is on the last merge.
4. Re-run verification.
5. Report: "WAVE 4 NEEDS FIXES — reverted wave4/<AGENT>, remaining tests
   green".

---

## 9. THE REPORT FORMAT

At the very end, report to the user with EXACTLY this structure:

```
WAVE 4 EXECUTION SUMMARY
========================

Wave 4 started: <timestamp>
Wave 4 finished: <timestamp>
Duration: <hours>

Agents: <N>/20 complete
  COMPLETE: <list of agent IDs, one per line with commit SHA>
  BLOCKED:  <list of agent IDs with reasons, one per line>
  FAILED:   <list of agent IDs with reasons, one per line>

Branches merged: <N>/20
  MERGED:   <list>
  CONFLICTS: <list with conflicting files>

Files audited:    <N>
Files changed:    <N>
Tests added:      <N>
P0 bugs fixed:    <N>
P1 bugs fixed:    <N>
P2 bugs fixed:    <N>
Bugs logged for Wave 5: <N> (see issues/wave5_backlog.md)

Verification results:
  pytest:        PASS / FAIL (<counts>)
  ruff:          PASS / FAIL (<count> errors)
  mypy:          PASS / FAIL (<count> errors)
  build:         PASS / FAIL
  entry points:  PASS / FAIL
  security greps: PASS / FAIL

Final status: READY FOR v2.4.0 TAG / NEEDS FIXES

Next steps:
  - If READY: run `git tag v2.4.0` and `git push --tags`
  - If NEEDS FIXES: see "conflicts" and "failed" sections above
```

---

## 10. RESUME PROTOCOL

If the server restarts at any point during Wave 4, the orchestrator MUST
use this protocol on the next invocation. Do NOT restart from scratch.

### 10.1 — Read progress file

```bash
cat .agent/WAVE4_PROGRESS.json
```

For each agent in `progress['agents']`, check its `status`:
- `complete`: skip
- `blocked`: skip (report as is)
- `failed`: skip
- `running`: needs resumption
- `pending`: needs creation + spawn

### 10.2 — Verify branch state

For each agent marked `running` or `pending`:

```bash
AGENT="W4-L1"  # repeat for each
BRANCH="wave4/$AGENT"
WORKTREE="../${REPO_NAME}_${AGENT//-/_}"

# Does branch exist?
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    COMMITS=$(git log "$BRANCH" --oneline | wc -l)
    echo "$AGENT: branch exists, $COMMITS commits"
else
    echo "$AGENT: no branch — needs creation"
fi

# Does worktree exist?
if [ -d "$WORKTREE" ]; then
    echo "$AGENT: worktree exists at $WORKTREE"
else
    echo "$AGENT: no worktree — needs creation"
fi
```

### 10.3 — Classify and act

For each agent:

| Branch exists? | Commits > 0? | Last commit prefix? | Action |
|---|---|---|---|
| Yes | Yes | `fix(wave4/...)` | Mark COMPLETE. Skip. |
| Yes | Yes | `wip(wave4/...)` | Mark PARTIAL. **Respawning would lose work.** Report to user. |
| Yes | No | — | Mark NOT_STARTED. Respawn. |
| No | — | — | Create branch + worktree. Respawn. |

### 10.4 — Resume spawning

For agents marked NOT_STARTED or no branch: spawn them using the SAME
Section 4 template.

For agents marked PARTIAL: report to user with:
```
PARTIAL: <AGENT> at <SHA>: <N> commits, last commit "wip(...)"
Branches: wave4/<AGENT>
Suggestion: Human review required. Either accept as-is or ask agent to
continue. Do NOT auto-respawn — this would discard the partial work.
```

### 10.5 — Continue from Section 6

After respawning NOT_STARTED agents, return to Section 6 (wait protocol).

Do NOT restart from Section 1 (pre-flight). Do NOT recreate branches that
already exist. Do NOT respawn complete agents.

---

## 11. TROUBLESHOOTING

### 11.1 — Agent says "branch is wrong"

If an agent reports "ERROR: wrong branch", it means the worktree was set
up incorrectly. Verify:

```bash
cd <WORKTREE>
git rev-parse --abbrev-ref HEAD
```

If it's not `wave4/<AGENT_ID>`, fix it:

```bash
cd "$(git rev-parse --show-toplevel)"
git worktree remove <WORKTREE> --force
git worktree add <WORKTREE> wave4/<AGENT_ID>
```

Then respawn the agent.

### 11.2 — Agent says "worktree dirty"

If an agent reports "ERROR: worktree dirty with files", another agent's
work leaked in. This should not happen if Section 3.2 was followed
correctly. Clean it:

```bash
cd <WORKTREE>
git checkout -- .
git clean -fd
```

Then respawn.

### 11.3 — Agent commits to wrong branch

If an agent commits to `main` instead of `wave4/<AGENT_ID>`:

```bash
cd "$(git rev-parse --show-toplevel)"
# Verify the commit hash
git log main -1
# If it's the agent's commit:
git reset --hard HEAD~1  # undo on main
# Then check the agent's branch
cd <WORKTREE>
git log -1
# If it's on the right branch, good
# If it's not on the branch at all, the commit is lost
# In that case: instruct agent to redo from clean state
```

### 11.4 — Server restarted mid-merge

If the server restarts during Section 7 (merge):

```bash
cd "$(git rev-parse --show-toplevel)"
git status
```

If there are unresolved merge conflicts:
```bash
git merge --abort
```

Then re-run Section 7 from the beginning, but skip merges that already
succeeded (they're now part of `main` and are no-ops):

```bash
git log --oneline | grep -c "merge: wave4/"  # count how many merged
# Skip the first N in the merge order
```

### 11.5 — Agent produces a PR > 2000 lines

If an agent's PR is >2000 lines:

1. Check the commit: `git log wave4/<AGENT> --stat`
2. If a single commit is >2000 lines: ask the agent to `git reset
   --soft HEAD~1` and split into smaller commits.
3. If the branch has multiple commits each <2000 lines: it's fine, the
   total is just large. Different agents can produce different volumes.

### 11.6 — Test file added outside Owned Paths

If an agent committed a test file outside its Owned Paths (e.g. a file in
`tests/models/` when the agent doesn't own that directory):

1. Revert that specific file: `git checkout main -- <file>`
2. Ask agent to move the test into its own domain directory.
3. If it doesn't fit any domain, ask the agent to log it as cross-domain
   bug.

### 11.7 — Test fails after merge but not before

This means two agents' work conflict semantically (not git conflict, but
runtime conflict). For example:
- Agent A changed `BaseClass.method()` signature
- Agent B's test calls `BaseClass.method()` with old signature

To resolve:

```bash
git checkout main
git revert <merge commit of B>  # or A, whichever is more recent
pytest tests/unit tests/contract -x -q
# If green: the issue is with B (or A)
# Report: "SEMANTIC CONFLICT: <A> and <B>. B reverted. A may need to update B's test."
```

Do NOT auto-fix. Report and let the user decide.

---

## 12. ABSOLUTE PROHIBITIONS

The following are absolutely forbidden. Violation = automatic rejection and
require human review:

1. **Sharing a worktree between two agents.** Each agent must have its own.
2. **Two agents touching the same file.** Verify at merge time; if
   conflict, report and revert.
3. **Merging without passing tests.** Every merge must be followed by
   `pytest tests/unit tests/contract -x -q`.
4. **Skipping the wait protocol.** Do NOT return control to the user
   before all 20 agents have reported or timed out.
5. **Force-pushing.** Never `git push --force` or `git push --force-with-lease`.
6. **Touching `AGENTS.md`, `.agent/WAVE4_EXECUTE.md`, or `pyproject.toml`
   from an agent's worktree.** These are owned by the orchestrator/human.
7. **Adding `except: pass` without `# allowed-swallow: reason`.**
8. **Replacing a fake implementation with another fake implementation.**
   Use `NotImplementedError`.
9. **Claiming "audited" without evidence.** Every audited file must have
   a test or a documented "verified-safe" note.
10. **Committing more than 2000 lines in one commit.**
11. **Committing files larger than 1 MB.**
12. **Merging a branch with 0 commits.**
13. **Merging a branch that fails tests.**
14. **Skipping the report format in Section 9.**
15. **Restarting from scratch after a server restart.** Always resume from
    `.agent/WAVE4_PROGRESS.json`.

---

## 13. BEGIN

Start now with Section 2 (Pre-Flight). If all pre-flight checks pass,
proceed to Section 3.2 (Worktree creation), then Section 4 (Spawn), then
Section 6 (Wait), then Section 7 (Merge), then Section 8 (Verify), then
Section 9 (Report).

Do NOT stop to ask "should I continue?" — the answer is always YES, continue
to the next step.

Do NOT skip any step.

Do NOT return control to the user until Section 9 (Report) is complete.

BEGIN.
