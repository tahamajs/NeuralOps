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
