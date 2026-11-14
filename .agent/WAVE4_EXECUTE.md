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

### 3.7 — Push protocol (per agent)

After the agent's final commit:

```bash
git push -u origin wave4/<AGENT_ID>
```

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

STEP 3: Confirm the worktree is clean.
    git status --porcelain

STEP 4: Read the three reference documents, in this order.
    cat AGENTS.md
    cat .agent/WAVE4_EXECUTE.md
    cat .agent/WAVE4_AGENTS/<AGENT_ID>.md

STEP 5: Confirm you understand your Owned Paths.

═══════════════════════════════════════════════════════════════════════
EXECUTION
═══════════════════════════════════════════════════════════════════════

PHASE A — READ AND INSPECT
PHASE B — SMOKE TEST
PHASE C — RED FLAG SCAN
PHASE D — DEEP AUDIT
PHASE E — FIX AND TEST
PHASE F — COMMIT
PHASE G — FINAL COMMIT AND PUSH
PHASE H — REPORT (COMPLETE or BLOCKED)
```

---

## 5. THE 20 AGENT SPECIFICATIONS

### 5.0 — File header (prepend to every agent file)

```markdown
# Wave 4 Agent <AGENT_ID>: <TITLE>

**Agent ID:** <AGENT_ID>
**Branch:** wave4/<AGENT_ID>
**Task file location:** .agent/WAVE4_AGENTS/<AGENT_ID>.md
**Full reference:** .agent/WAVE4_EXECUTE.md
```

### 5.1 — W4-L1: Layers / Linear Family
### 5.2 — W4-L2: Layers / Sparse Family
### 5.3 — W4-L3: Layers / Standard Family
### 5.4 — W4-L4: Layers / Specialized + Adapters
### 5.5 — W4-O1: Ops / Triton (remaining 30)
### 5.6 — W4-O2: Ops / CUDA + Remaining
### 5.7 — W4-C1: Compression (remaining 37)
### 5.8 — W4-T1: Trainer / Distillation + Optimizers
### 5.9 — W4-T2: Trainer / Methods + Metrics + Losses
### 5.10 — W4-T3: Trainer / Callbacks + Schedulers
### 5.11 — W4-A1: API Facade + Config
### 5.12 — W4-C2: Core Remaining
### 5.13 — W4-E1: Export + Inference + Integrations
### 5.14 — W4-U1: Evaluation + UI
### 5.15 — W4-U2: CLI + Utils + Results Saver
### 5.16 — W4-B1: Benchmarks / Suites
### 5.17 — W4-B2: Benchmarks / Core + KD + Storage
### 5.18 — W4-B3: Benchmarks / Adapters + Persistence
### 5.19 — W4-I1: Benchmarks CLI + Legacy
### 5.20 — W4-R1: Infrastructure + Release

---

## 6. THE WAIT PROTOCOL

Wait loop polls progress until all 20 report COMPLETE or BLOCKED.

---

## 7. THE MERGE PROTOCOL

Phase 1: Infrastructure (W4-R1, W4-I1)
Phase 2: Primitives (W4-L1..W4-L4, W4-O1..W4-O2, W4-C1)
Phase 3: Trainer + Core + Config (W4-T1..W4-T3, W4-A1, W4-C2)
Phase 4: Domain services (W4-E1, W4-U1, W4-U2)
Phase 5: Benchmarks (W4-B1..W4-B3)

---

## 8. FINAL VERIFICATION

Run `scripts/wave4_verify.sh`, verify imports, tests, ruff, mypy, build, and entry points.

---

## 9. THE REPORT FORMAT

Structured report on completed agents, fixed bugs, and verification results.

---

## 10. RESUME PROTOCOL

Read `.agent/WAVE4_PROGRESS.json`, verify branch status, resume incomplete agents.

---

## 11. TROUBLESHOOTING

Resolution paths for common branch, worktree, and test conflict scenarios.

---

## 12. ABSOLUTE PROHIBITIONS

Hard rules governing worktree isolation, disjoint file ownership, and fail-closed quality.

---

## 13. BEGIN

Start now with Section 2 (Pre-Flight).
