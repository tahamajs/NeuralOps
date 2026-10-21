# NeuralOps Guidelines

Guidance for AI coding agents (Claude Code, Codex, Antigravity, etc.) working in this repo.

**Read `CONTRIBUTING.md` first.** It is the authoritative source for all code style, docstring, comment, commit, PR, and testing conventions, and applies to humans and agents alike. This file covers agent-specific operational behavior that doesn't belong in a human contributor doc.

**These rules exist because past agent sessions caused real regressions.** Each section header notes the failure mode it prevents. Read them as lessons, not suggestions.

---

## 0. Core Operating Principles

Three rules override everything else in this file:

1. **Never lie about verification.** If you didn't run a command, say you didn't run it. If you ran it and it failed, paste the failure. If you couldn't run it because it was blocked, quote the exact rule that blocked it (file path + line number). "It should work" is not verification. "I ran X and got Y" is verification.

2. **Never produce fake output for a format you cannot create.** If a converter fails, `raise`. If a checker fails, `raise`. If a backend cannot load, `raise`. Never write a placeholder file with a real extension (`.onnx`, `.gguf`, `.trt`, `.engine`). Never fall back to a hardcoded number when real data is missing. Never write `except Exception: pass` around data-producing code. A silent failure in production is worse than a visible crash in development.

3. **Never proceed past a blocker without explicit user authorization.** If a rule blocks you (missing tool, prohibited action, unreachable host), stop and report. Do not silently improvise around the rule. Do not claim the rule "must not apply here". Quote the rule, describe the blocker, ask the user. The user is the final authority and can override any rule in this file.

---

## 1. Git Safety & Attribution Laws

- **Mandatory Commit Attribution**: All Git commits MUST strictly use `--author="taha <tahamaj4@gmail.com>"`.
- **Never discard uncommitted work** with `git checkout HEAD -- <file>`, `git reset --hard`, or `git restore` to "get a clean base". Unstaged changes are unrecoverable (no blob, no reflog). Edit in place or `git stash` instead, and confirm with the user when in doubt.
- **On `main`**: never commit or push without explicit per-action approval. Suggest a feature branch first.
- **Branch names and attribution**: name local and remote branches after the change. Do not include tool or assistant identities in branch names, commit messages, PR titles/bodies, or review comments. Do not add `Co-Authored-By` trailers or generated-by notices.
- **Commit boundaries**: one logical change per commit. Do not mix cleanup of a prior session's regression with new feature work in the same commit. If a session touched `analysis/` and `distributed/`, that is two commits, not one.
- **Never force-push to `main`.** Never rewrite or amend already-pushed commits unless the user explicitly asks.
- **Preserve file history on moves**: use `git mv`, not `cp` + `rm`. Preserve rename detection in `git log --follow`.

---

## 2. Authorship, Header & Comment Discipline

- **Strict Header Standard**: Every source file MUST start with the standard top-of-file MIT license block referencing contributors:
  ```python
  #!/usr/bin/env python3
  # Copyright (c) 2024-2026, Taha Majlesi
  #
  # This source code is licensed under the MIT license found in the
  # LICENSE file in the root directory of this source tree.
  # For a list of all contributors, visit:
  #   https://github.com/tahamajs/NeuralOps/graphs/contributors
  """
  Module Docstring here...
  """
  ```
- **Banned Personal Docstring Headers**: NEVER insert inline `"""Author: Taha Majlesi` or `# Author: Taha Majlesi` docstrings. Codebase authorship is managed via Git history and the official contributors graph.
- **You own the comments your change invalidates.** When a change makes a comment or docstring in the code you touched factually wrong, fix it or delete it in the same commit — "the code you touched" means the same function, kernel, or class, not a repo-wide sweep for stale mentions. If you can't tell whether a comment is still true, keep it and flag it in the PR description; never delete a "why" comment just because you can't verify it.
- **Comments are not your scratchpad.** State design conclusions, not the journey: "not X, it does not compile" is a welcome why comment; "tried X first, then switched to Y" is chain-of-thought noise. This covers docstrings too.
- **Fix the words, not the style.** Correcting stale content is not a license to reformat the surrounding docstring or comment style.

---

## 3. Strict Execution Environment Rules

### 3.1 Default Prohibition

- **Zero Local macOS Python Execution (unsupervised)**: During normal operation, NEVER execute `python`, `pytest`, `python -c`, or any interpreter call on the local macOS host without explicit authorization. macOS Python 3.14 may lack typing extensions, PyTorch CUDA extensions, and native accelerator drivers, producing results that look like failures but aren't.
- **Static Verification on Host (always allowed)**: Local verification on macOS MUST be conducted via POSIX inspection (`grep`, `rg`, `awk`, `find`, `sed`), AST parsers (`scripts/utils/audit_dead_code.py`), and lint checkers (`ruff`, `mypy` if installed). These are always permitted.
- **Dynamic & GPU Kernel Verification (always remote)**: GPU tests, `torch.autograd.gradcheck`, CUDA C++ compilation, and Triton JIT execution MUST be run remotely on `taha@192.168.1.100` (`/mnt/d/SVD_Project`). Never attempt to run them locally under any circumstance.

### 3.2 User Authorization Exception (READ THIS BEFORE CITING PROHIBITION)

The default prohibition in §3.1 exists for safety, not for obstruction. The user is the final authority and can override it.

**The user authorizes local Python execution when they say any of:**
- "I authorize local pytest"
- "Run the tests yourself"
- "You may execute python locally"
- "AGENTS.md is updated to allow local testing"
- "Override the rule for this session"

**When authorized, the following rules apply:**

- **Allowed**: `python -c "import <module>"`, `pytest tests/<specific_file>.py`, `pytest tests/<dir>/ -v --tb=short` for CPU-only tests.
- **Still prohibited**: Any test marked `@pytest.mark.gpu` or requiring `torch.cuda.is_available()`. Any invocation of `pytest` on the full suite without `-k` or path filtering (which would hit GPU tests and hang).
- **Scope**: Authorization is per-session. When in a new session, the default prohibition returns. Do not assume prior authorization carries over.
- **Report**: When you run pytest under user authorization, explicitly state in your report: "User authorized local pytest for CPU-only tests in this session. Running: `<exact command>`."

### 3.3 Tool Discovery Protocol

Before claiming a tool is "not installed", do this:

1. **Check the venv explicitly**: `ls -l .venv/bin/ruff .venv/bin/pytest .venv/bin/python 2>/dev/null`
2. **Check `pip list`**: `pip list 2>/dev/null | grep -iE "ruff|pytest|mypy"` (from an activated venv)
3. **Check PATH**: `which ruff pytest python 2>/dev/null`
4. **If found in `.venv/bin/` but not in PATH**: use the full path: `./.venv/bin/ruff check ...` or `.venv/bin/pytest ...`

**Never claim "ruff is not installed" or "pytest is not available" without running the above checks.** In past sessions, agents claimed tools were missing while they were installed at `.venv/bin/`. That was a false claim.

---

## 4. Verification Contract

**A change is not "done" until you have pasted evidence.** The required evidence depends on what changed.

### 4.1 For pure-Python logic changes (analysis, surgery, models, distributed)

Required evidence (paste raw output for each):

1. `ruff check <changed_files>` → must show `All checks passed!` or a clear list of remaining warnings you acknowledge.
2. `python -c "import neuralops.<changed_module>"` → must print `OK` (requires §3.2 authorization).
3. `pytest tests/<relevant_test_file>.py -v --tb=short` → must show pass/fail counts.
4. If you wrote a regression test, prove it fails on the old code (see §5.4).

### 4.2 For kernel / CUDA / Triton changes

Cannot be verified locally. Required evidence:

1. `ruff check <changed_files>` → must pass.
2. `python -c "import neuralops.<changed_module>"` → must not crash on CPU (fallback path).
3. Remote verification is deferred to the user. State explicitly: "GPU verification deferred to remote host. To verify: `ssh taha@192.168.1.100 ...` and run `<command>`."

### 4.3 For documentation-only changes

1. `grep -A2 -B2 "old_wording" docs/` should return nothing (the old wording is gone).
2. If the doc references code, verify the code still exists: `rg "def <name>" neuralops/`.

### 4.4 When verification is blocked

If any command in §4.1-4.3 is blocked by a rule in this file, do **not** silently skip it. Instead:

1. State the blocker with file+line: "Blocked by AGENTS.md §3.1 line 43."
2. Ask the user for authorization (§3.2).
3. If the user cannot authorize in this session, mark the change as "unverified — awaiting remote verification" in the report. Do not commit.

---

## 5. Scope Discipline

Past sessions repeatedly expanded scope beyond what was asked, producing 20+ file diffs that no one could review.

### 5.1 Read Before Edit

Before editing **any** file:

1. Read the file completely (`cat` or `sed -n '1,500p'`).
2. Read every test that touches it (`rg -l "from neuralops.<module> import" tests/`).
3. List every caller of the symbols you plan to change (`rg "from neuralops.<module>" neuralops/ tests/`).
4. State in a single sentence what change you are about to make and why.

If you cannot state the change and its blast radius in one sentence, the change is too large. Break it up.

### 5.2 Stay in Scope

- If the user's request mentions `models/` and `analysis/`, do not touch `distributed/`.
- If the user's request is "audit", do not fix — report only, wait for approval.
- If the user's request is "fix", fix only the files that contain the bug — not adjacent files that "could also be improved".
- **Cleanup of prior session's regressions is in scope** even if it touches files outside the current session's stated folder. Rationale: the agent caused the regression, the agent fixes it. Do not mix this cleanup into the new work's commit; make a separate commit.

### 5.3 Acknowledge Contradictions in the Prompt

If a user's prompt contains an internal contradiction (e.g., "fix file X" and "only touch folder Y where X is not located"), **do not silently resolve it**. State the contradiction and ask. Past sessions resolved contradictions by ignoring one side, which led to incomplete fixes.

### 5.4 Regression Tests Are Required for Bug Fixes

When you fix a bug, write a test that:

1. **Fails on the pre-fix code** — you must demonstrate this by reverting the fix (`git stash`, or `git checkout HEAD -- <specific_file>`), running the test, seeing it fail, then restoring the fix and seeing it pass.
2. **Is minimal** — tests one behavior, not five.
3. **Lives in `tests/<subsystem>/`** — matching the file it tests.

The fail-then-pass proof must appear in your report. Tests that pass on both old and new code are worthless — they prove nothing.

### 5.5 Do Not Introduce New Files Without Asking

Creating new files in `neuralops/` (e.g., a new `data/glue.py`) expands the public API surface. Before creating a new file under `neuralops/`:

1. Check whether an existing file can host the addition.
2. If a new file is truly required, state why in the report and wait for approval.
3. New test files under `tests/` are always allowed (no approval needed).

---

## 6. Silent Failure Policy

Past sessions introduced and left in place several classes of silent failure. They are banned.

### 6.1 No Fake Artifacts

If a format converter cannot produce a valid file, **raise**. Never write:

- A text file with `.onnx` extension that is not valid ONNX protobuf.
- A `.gguf` file that is not a valid GGUF container.
- A `.trt` file containing `NEURALOPS_TRT_ENGINE_V1` placeholder bytes.
- Any file whose extension implies a format it does not contain.

Correct pattern:
```python
try:
    write_valid_onnx(...)
except Exception as e:
    raise RuntimeError(
        f"ONNX export failed for attention={attention_name}. "
        f"Try: format='safetensors' or a different attention. "
        f"Underlying error: {e}"
    ) from e
```

### 6.2 No Hardcoded Numeric Fallbacks in Data Paths

If a baseline accuracy, a metric, or a data point is missing, **raise or log a warning — never substitute a number**.

Banned pattern:
```python
baseline_acc = subset["baseline_acc"].iloc[0] if "baseline_acc" in subset else 90.14
```

Correct pattern:
```python
if "baseline_acc" not in subset or subset.empty:
    raise ValueError("No baseline accuracy in sweep results — cannot compute retention.")
baseline_acc = subset["baseline_acc"].iloc[0]
```

Banned pattern:
```python
{"name": "Dense Baseline", "compression_ratio": 1.0, "accuracy": 89.5}
```

Correct pattern: load from actual results file. If results file is missing, raise.

### 6.3 No Silent Exception Swallowing in Data Producers

`except Exception: pass` and `except Exception: return None` are banned in code paths that produce data, metrics, reports, or files.

Allowed only when:
- The exception is from a side-effect-only path (e.g., cleanup), AND
- A comment explains why swallowing is correct, AND
- A log line records the swallow at `debug` or `info` level.

### 6.4 No Silent Fallback to Default/Empty Data

`np.eye(N)` as a fallback when real covariance is missing, `np.zeros(N)` as a fallback when real mean is missing, `if not data: continue` without a warning — all banned. Either raise or log at `warning` level and mark the result as `None`.

---

## 7. Kernel & Architecture Design Standards

- **3-Level Fallback Hierarchy**: Every kernel operation must adhere to the 3-level fallback contract:
  1. *Level 1*: Native CUDA C++ / HIP compiled extension (maximum throughput).
  2. *Level 2*: Triton JIT kernel (autotuned for FP16/BF16/FP32).
  3. *Level 3*: Vectorized PyTorch reference (universal CPU/MPS fallback).
- **Zero-Overhead Dispatch**: Kernel backend selection must use `@lru_cache` or zero-allocation dispatch tables.
- **No Block Pointers in Mainline Triton**: Check the diff for `tl.make_block_ptr` / `tl.advance`; these are banned from mainline Triton code (Huawei Ascend NPU backend exemption only).
- **Numerical Budget & Dtype Guards**: Custom forward kernels must compute or accumulate in FP32 when inputs are FP16 or BF16. Finite difference autograd gradient checks must pass `atol=1e-3, eps=1e-5`.
- **No Silent Fallback in 3-Level Dispatch**: If Level 1 fails, the dispatcher may try Level 2, then Level 3 — but each fallback must emit a `logger.warning` with the failing level and reason. Silent fallback to Level 3 is banned (users would not know they lost 100x speedup).
- **Flame-Style Multi-Dataset & Packing Standards**:
  - Multi-dataset ingestion must support comma-separated paths: `--training.dataset HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2`.
  - Zero-padding variable-length document packing (`--training.varlen`) must generate valid `cu_seqlens` boundary offsets `[0, len_1, len_1+len_2, ..., seq_len]`.
  - Learning rate schedulers must support Warmup-Stable-Decay (`wsd`).

---

## 8. Repository Invariants & Non-Core Directories

- **Canonical Counts Invariant**:
  - `ATTENTION_REGISTRY` must contain exactly **131** attention mechanisms.
  - `COMPRESSOR_REGISTRY` must contain exactly **72** model compression engines.
  - `neuralops/core/triton_kernels/` must contain exactly **26** canonical Triton kernel modules (housing 62 unified operators across 4 tiers).
  - `neuralops/core/cuda_kernels/` must contain exactly **114** native CUDA C++ kernel packages.
  - `notebooks/` must contain exactly **27** fully executed Jupyter notebooks.
- **Ignored Directories Law**:
  - `audit/`, `docs/`, `results/`, `academic/`, `paper/`, `thesis_final/`, `logs/`, `scratch/`, and `grading.log` are strictly ignored in `.gitignore`.
  - These folders contain academic thesis drafts, build artifacts, raw telemetry, and local experiment outputs. They must NEVER be committed to Git.
- **Verifying invariants**: `python3 scripts/utils/audit_dead_code.py` and `python3 scripts/audit_readme_consistency.py` verify counts. Run these before any PR. Requires §3.2 authorization.

---

## 9. Code Modification Rules

- **Minimal diff**: every changed line serves the stated purpose. Drop drive-by reformats, unrelated cleanups, and speculative generality before pushing.
- **Fix the bug, not the neighborhood.** If a function has a bug, fix the function. Do not reformat the surrounding 50 lines, do not rename variables, do not extract helpers. If you believe the function needs a larger refactor, propose it separately.
- **No bulk reformatting.** Do not run `black`, `ruff format`, or any auto-formatter on files you are editing for a bug fix. The diff should be reviewable in under a minute.
- **Preserve public API.** If you must rename or remove a public symbol, add a deprecation alias and document it in the PR description.
- **Never remove a "why" comment** you cannot verify. If a comment explains a non-obvious decision, keep it, even if the code around it changed.
- **Update `__all__` when adding public symbols.** A symbol that is importable but not in `__all__` will not be found by `from module import *` or by static analysis tools.

---

## 10. Blocker Escalation & Honest Reporting

Past sessions had agents silently skip blockers, then report success. This is banned.

### 10.1 When Blocked

When you cannot complete a task due to a rule, missing tool, unreachable host, or missing information:

1. **State the blocker explicitly**:
   - "Blocked by AGENTS.md §3.1 line 43 — pytest prohibited without authorization."
   - "Blocked by missing tool: `ruff` not found at `.venv/bin/ruff` or in PATH."
   - "Blocked by unreachable host: `ssh taha@192.168.1.100` returns `Connection refused`."
2. **State what you tried**:
   - "I ran `ls .venv/bin/ruff` and got: `<output>`."
   - "I ran `ssh -v taha@192.168.1.100` and got: `<last 5 lines>`."
3. **State what you need from the user**:
   - "Please authorize local pytest (§3.2)."
   - "Please install ruff: `pip install ruff`."
   - "Please provide an alternate host."
4. **Stop**. Do not improvise around the blocker. Do not claim the blocker "does not apply". Do not produce a partial result and call it done.

### 10.2 Report Format

Every "task complete" report MUST follow this structure:

```
## Summary
<one sentence: what was done>

## Files changed
<output of `git diff --stat`>

## Verification (evidence required, not claimed)
### Command 1: <exact command>
<exact output pasted>

### Command 2: <exact command>
<exact output pasted>

## Regression test (for bug fixes)
### Test file: <path>
### Fail on old code: <paste of pytest output after `git stash`>
### Pass on new code: <paste of pytest output after `git stash pop`>

## Blockers (if any)
<anything that could not be verified, with the exact rule or missing dependency>

## Next steps
<what the user needs to do to fully verify, if anything>
```

Do not deviate from this format. A report without pasted command output is not a report — it is a claim.

### 10.3 Honest Failure

If your fix introduced a new bug (e.g., a previously passing test now fails):

1. **Acknowledge it immediately** — do not hide the failure.
2. **Revert the specific change** that caused the regression, or fix it before reporting.
3. **State what was reverted and why**.

A session that says "I tried to fix X, it broke Y, I reverted" is a successful session. A session that says "X fixed" while Y is broken is a failure.

---

## 11. PR Workflow & Quality Gates

- **No busywork PRs**: don't open a one-off PR for a single typo or isolated style tweak; bundle trivial cleanups into substantive work.
- **Minimal diff**: every changed line serves the PR's stated purpose. Drop drive-by reformats, unrelated cleanups, and speculative generality before pushing.
- **Blast radius audit**: for every shared symbol you touched, verify all callsites hold. State in the PR description which shapes, dtypes, and backends are affected.
- **Mandatory Quality Gates**:
  1. `python3 scripts/utils/audit_dead_code.py` → 0 issues found.
  2. `python3 scripts/audit_readme_consistency.py` → 0 errors found.
  3. Universal shebang `#!/usr/bin/env python3` and executable permissions on CLI scripts.
  4. `ruff check <changed_files>` → clean.
  5. All new code has regression tests (§5.4).
- **Pre-PR checklist** (agent must paste this filled out):
  ```
  [ ] ruff check passes
  [ ] pytest on affected modules passes (or blocked with authorization note)
  [ ] audit_dead_code.py → 0
  [ ] audit_readme_consistency.py → 0
  [ ] No fake artifacts introduced (§6.1)
  [ ] No hardcoded numeric fallbacks (§6.2)
  [ ] Minimal diff (§9)
  [ ] Regression test with fail-before/pass-after (§5.4)
  [ ] Report format followed (§10.2)
  ```

---

## 12. Repo-local Skills

Use the [skill index](.agents/skills/README.md) to select task-specific workflows and their execution order. Skills supplement `CONTRIBUTING.md` and this file; they do not define competing contribution policies.

Load a relevant skill when the task matches its scope. In particular, load [neuralops-pr-readiness](.agents/skills/neuralops-pr-readiness/SKILL.md) before opening any PR.

---

## 13. Version History & Change Log

When updating this file, append an entry here. Version number follows `major.minor.patch`. Major: rule added that retroactively invalidates prior practice. Minor: clarification or new section. Patch: typo or wording fix.

- **v1.1.0** — Added §0 (Core Operating Principles), §4 (Verification Contract), §5 (Scope Discipline), §6 (Silent Failure Policy), §10 (Blocker Escalation & Honest Reporting). Expanded §3 with user authorization exception (§3.2) and tool discovery protocol (§3.3). Added regression test requirement (§5.4). Added report format (§10.2). Added quality gate checklist to §11.
- **v1.0.0** — Initial agent guidelines.

---
## 14. Versioning, Releases & Changelog

Past sessions produced version drift: the package reported `1.4.0` while `pyproject.toml` said `1.3.7`, `CITATION.cff` said `1.3.0`, and the git tag was `v1.4.0-beta`. This section locks the versioning contract.

### 14.1 Source of Truth

**`neuralops/__init__.py:__version__`** is the single source of truth. Every other version string in the repo derives from it.

**Rule:** when bumping the version, update `__version__` first. Then run `python3 scripts/utils/sync_version.py` (see §14.7) to propagate everywhere else. Never edit version strings in downstream files by hand.

### 14.2 Semantic Versioning (SemVer 2.0.0)

The version follows `<major>.<minor>.<patch>` exactly.

| Bump | When | Examples |
|---|---|---|
| **major** | Incompatible API change. Removing a public symbol, changing a function signature, changing return shape/dtype, or changing a kernel's numerical contract. | `replace_attention()` no longer accepts `calibrate=`; `LayerwiseAnalyzer.analyze()` returns a different dict shape |
| **minor** | New functionality, backward compatible. Adding a new attention mechanism, a new compressor, a new public function, a new optional argument with a default. | Adding `attention_type="new_attention"` to the registry; adding a new `format=` option to export |
| **patch** | Bug fix, no API change, no behavior change visible to users except that a bug disappeared. | Fixing the `is_causal` inference in `scanner.py`; fixing a memory leak in `universal_calibrator.py` |

**Ambiguous cases** — ask the user:

- **Fixing a silent fallback that produced wrong output** (e.g., `90.14` placeholder → raise). This is a behavior change (users who relied on the fallback now see an exception), but it is also a bug fix. **Default: minor bump**, because users must be told. Document the behavior change explicitly in `CHANGELOG.md` under `### Changed` and `### Fixed`.
- **Changing the numerical output of a kernel** (e.g., fixing a non-causal recurrence to causal). **Default: minor bump with `BREAKING CHANGE:` footer.** Downstream models trained against the old buggy behavior must retrain.
- **Renaming a symbol while keeping an alias.** **Default: patch bump**, because the old symbol still works via the alias. Remove the alias in the *next* major.

### 14.3 Version String Formats

PEP 440 is the normative standard. Use it exactly.

| Stage | Format | When |
|---|---|---|
| Development | `1.5.0.dev0`, `1.5.0.dev1`, ... | Between minor releases, before a feature is complete. Never published to PyPI. |
| Alpha | `1.5.0a1`, `1.5.0a2`, ... | Internal testing, API unstable. |
| Beta | `1.5.0b1`, `1.5.0b2`, ... | API frozen, bug fixes only. |
| Release candidate | `1.5.0rc1`, `1.5.0rc2`, ... | Code frozen, waiting for final sign-off. |
| Release | `1.5.0` | Published to PyPI and tagged. |
| Post-release fix | `1.5.0.post1` | Metadata-only fix (e.g., a typo in README shipped with the wheel). No code change. |

**Never** use suffixes like `-final`, `-stable`, `-lts`, `-beta` with a dash. PEP 440 requires the concatenated form.

### 14.4 Files That Must Stay in Sync

When `__version__` changes, **all** of the following must be updated. Missing any one causes the release to fail its consistency gate (§14.7).

| File | Field | Format |
|---|---|---|
| `neuralops/__init__.py` | `__version__` | `"1.5.0"` |
| `pyproject.toml` | `[project].version` | `"1.5.0"` |
| `pyproject.toml` | `[project].requires-python` | unchanged unless dropping support |
| `CITATION.cff` | `version` | `"1.5.0"` |
| `CITATION.cff` | `date-released` | `"2026-11-15"` (release date, not commit date) |
| `CHANGELOG.md` | top section header | `## [1.5.0] - 2026-11-15` |
| `neuralops/api.py:get_version()` | return value | `"1.5.0"` |
| `README.md` | PyPI badge | `https://img.shields.io/pypi/v/neuralops.svg` (dynamic, no hardcode) |
| `docs/conf.py` (if Sphinx) | `release` and `version` | `"1.5.0"` and `"1.5"` |
| Any sub-package (`neuralops-vllm/`, `neuralops-sglang/`) | their `__version__` | bumped per §14.6 |

**Pre-release files** (do not touch during a normal version bump):

- `paper/iclr2027-submission/` — version in the paper is a *paper* version, tracked separately (§14.8)
- `thesis_final/` — same
- `notebooks/` — notebooks that import `neuralops` do not embed the version

### 14.5 Git Tags & Releases

**Tag format:** `v<major>.<minor>.<patch>` with the `v` prefix. Examples: `v1.4.0`, `v1.5.0`, `v1.5.0rc1`.

**Annotated tags only.** Never use lightweight tags.

```bash
git tag -a v1.5.0 -m "Release v1.5.0: <one-line summary>"
git push origin v1.5.0
```

**Tag message must include:**
- One-line summary of the release
- The closing commit's SHA
- Whether the release is PyPI-published or not

**Never:**
- Tag a commit that has not passed `scripts/audit_readme_consistency.py` and `scripts/utils/audit_dead_code.py`.
- Tag a commit where any file in §14.4 still has the old version.
- Move a tag that has been pushed (delete and re-create is a rewrite; ask the user first).
- Tag a WIP commit. Wait for the PR to merge to `main`.

**Branch names for release prep:** `release/v1.5.0`, never `release-1.5.0` or `v1.5.0-branch`.

### 14.6 Sub-Package Versioning

If `neuralops-vllm/`, `neuralops-sglang/`, `neuralops-tensorrt-llm/`, or `neuralops-llamacpp/` exist:

- Each sub-package has its own `__version__` and its own SemVer line.
- Each sub-package declares a compatibility range for the core package: `neuralops>=1.5.0,<2.0.0`.
- Bumping core `neuralops` **does not** automatically bump sub-packages. Bump a sub-package only if its own code changed or its compatibility range changed.
- The core package must **never** import a sub-package. Import only goes one way: sub-package → core.
- When the core package bumps **major**, every sub-package's compatibility range must be re-evaluated. That is a coordinated release; the user must approve the plan before any version is bumped.

### 14.7 Version Consistency Gate

Every PR that touches a version-related file must run:

```bash
python3 scripts/utils/sync_version.py --check
```

**Expected output:** `OK: all version strings match 1.5.0`

If the script does not exist yet, add it. The script must check:

1. `neuralops.__version__` == `pyproject.toml [project].version`
2. `CITATION.cff version` == `neuralops.__version__`
3. Top entry in `CHANGELOG.md` == `[<neuralops.__version__>]`
4. `neuralops.api.get_version()` == `neuralops.__version__`
5. No hardcoded version string elsewhere in `README.md`, `docs/`, or source files (except in tests that assert the version).

The gate fails (exit code 1) if any check fails. The PR cannot merge until it passes.

To *bump* the version and sync all files:

```bash
python3 scripts/utils/sync_version.py --bump patch   # or minor, major
```

This is the only correct way to bump. Editing version strings by hand is banned.

### 14.8 Paper, Thesis & Documentation Versioning

**Paper and thesis versions are independent of code versions.** They follow their own schemes:

| Artifact | Version scheme | Where it lives |
|---|---|---|
| Paper (ICLR / NeurIPS) | `paper-v<major>.<minor>` (e.g., `paper-v1.2`) | `paper/<conf>-submission/VERSION` (if it exists) |
| Thesis | `thesis-v<major>.<minor>.<patch>` | `thesis_final/VERSION` (if it exists) |
| Documentation site | matches code version | `docs/conf.py:release` |

**Rule:** when code `neuralops` bumps **minor** or **major**, the paper and thesis must be reviewed for stale numbers. This is a **manual review step**, not an automated bump.

- If the paper cites `neuralops 1.4.0`, and code bumps to `1.5.0`, the paper's citation must be updated before submission.
- If the paper reports "143x speedup" and a kernel change in `1.5.0` changes that to "130x", the paper must be retracted or corrected.
- **The agent must flag every code-to-paper number that could be affected by the code change, in the PR description.** The reviewer decides whether to bump the paper version.

### 14.9 Registry Invariant Versioning

The canonical counts (§8) — 131 attention, 72 compressor, 26 Triton kernels, 114 CUDA kernels, 27 notebooks — are **invariants**, not versions. They must be preserved across all version bumps unless the user explicitly approves a count change.

When adding a new attention mechanism:

- Bump `ATTENTION_REGISTRY` count from 131 to 132.
- Bump `neuralops` version by **minor** (new public symbol).
- Update `README.md`, `docs/MODULES.md`, `AGENTS.md §8`, and any test that asserts `len(ATTENTION_REGISTRY) == 131`.
- Update the paper's appendix table (if the paper is still editable).

When removing a mechanism (rare, requires explicit approval):

- Bump `neuralops` version by **major** (breaking change for users of the removed mechanism).
- Update all of the above plus the migration guide.

### 14.10 Changelog Format

`CHANGELOG.md` follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) with these additions:

**Section structure** (most recent first):

```markdown
# Changelog

All notable changes to NeuralOps are documented here.
Format: Keep a Changelog. Versioning: Semantic Versioning 2.0.0.

## [Unreleased]

### Added
- ...

### Changed
- ...

### Deprecated
- ...

### Removed
- ...

### Fixed
- ...

### Security
- ...

## [1.5.0] - 2026-11-15

### Added
- New attention mechanism `xyz_attention` (registered as #132). See `docs/MODULES.md`.
- `neuralops.export_to_gguf()` — real GGUF container, no placeholder. Verified with `gguf-dump`.

### Changed
- **BREAKING**: `scanner.py:is_causal` is now `False` for `nn.MultiheadAttention` regardless of `batch_first`. Models trained against the old (incorrect) causal inference must retrain or use `strict=False`.
- `universal_calibrator.py` now cleans up hooks in a `finally` block. Existing calibrations are unaffected.

### Removed
- **BREAKING**: Placeholder bytes `NEURALOPS_TRT_ENGINE_V1` removed from `tensorrt_llm.py`. TRT export now raises `NotImplementedError` until a real engine is built.

### Fixed
- `hf_cache_bridge.py`: `_seen_tokens` no longer double-increments under Transformers 4.40+.
- `analysis/generate_sweep_publication_artifacts.py`: hardcoded `90.14` baseline fallback replaced with `raise ValueError` when baseline is missing.

### Security
- `dependencies`: bumped `cryptography` from 42.0.0 to 42.0.5 (CVE-2024-XXXXX).
```

**Rules:**

1. **Every merged PR adds an entry.** No exceptions. A PR without a changelog entry is rejected.
2. **`[Unreleased]` accumulates during development.** On release, rename `[Unreleased]` to `[<version>] - <date>` and open a new empty `[Unreleased]`.
3. **`BREAKING:` prefix required** for any change that violates SemVer. Every breaking change must state the migration path.
4. **Link to the commit or issue.** A fixed bug says which PR fixed it.
5. **Do not include "internal refactor" entries** unless the refactor changed a public symbol. Changelog is for users, not for the team.
6. **Do not delete historical entries.** Even after a major bump, `[1.0.0]` stays in the file.

### 14.11 Pre-Commit Version Hook

Install a pre-commit hook that runs `python3 scripts/utils/sync_version.py --check` on every commit:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: version-sync
      name: Verify version consistency
      entry: python3 scripts/utils/sync_version.py --check
      language: system
      pass_filenames: false
```

**When the hook fails**, the commit is blocked. Fix the version in the files listed in §14.4 and re-commit. **Never** use `--no-verify` to bypass this hook. If you believe the hook is wrong, escalate to the user.

### 14.12 Version Release Checklist

Before pushing a tag `v<version>`, the agent must paste this checklist into the PR description with every box checked:

```
[ ] __version__ in neuralops/__init__.py == X.Y.Z
[ ] pyproject.toml [project].version == X.Y.Z
[ ] CITATION.cff version == X.Y.Z and date-released == today
[ ] CHANGELOG.md has [X.Y.Z] - <today> section with entries from Unreleased
[ ] neuralops.api.get_version() == X.Y.Z
[ ] sync_version.py --check → OK
[ ] audit_dead_code.py → 0 issues
[ ] audit_readme_consistency.py → 0 errors
[ ] ruff check neuralops/ tests/ → clean
[ ] pytest tests/ -v → all pass (or GPU tests deferred with authorization note)
[ ] No file in §14.4 still has an old version
[ ] Paper and thesis reviewed for stale numbers (if code minor/major bump)
[ ] Registry invariants unchanged: 131/72/26/114/27 (or user approved change)
[ ] Pre-commit hooks run without --no-verify
[ ] Tag message includes summary + closing commit SHA
```

A release with any unchecked box must not be tagged. Ask the user for help before proceeding.

### 14.13 Versioning Failure Modes (Lessons From Past Sessions)

**Failure mode 1: Version drift across files.** Package said 1.4.0 but `pyproject.toml` said 1.3.7. Prevention: `sync_version.py --check` in pre-commit.

**Failure mode 2: Tag created without updating files.** Tag `v1.4.0` pointed at a commit where `__version__` was `1.3.7`. Prevention: release checklist §14.12.

**Failure mode 3: Breaking change shipped as patch.** Fixing `is_causal` inference (breaking) went into a `1.4.1` patch. Downstream users' models broke silently. Prevention: §14.2 default rules for ambiguous cases; `BREAKING:` prefix in `CHANGELOG.md`.

**Failure mode 4: Placeholder bytes shipped as a "release".** `NEURALOPS_TRT_ENGINE_V1` bytes shipped in `1.4.0`. Users thought they had a working TRT engine; they had bytes with a magic header. Prevention: §6.1, no fake artifacts.

**Failure mode 5: Paper number drift.** Paper reported "143x speedup" from a buggy non-causal kernel. After fixing the kernel, speedup dropped to 130x, but the paper was not updated. Prevention: §14.8 manual review step; agent must flag affected paper numbers in every PR.

---

## 15. Version History & Change Log

When updating this file, append an entry here. Version number follows `major.minor.patch`. Major: rule added that retroactively invalidates prior practice. Minor: clarification or new section. Patch: typo or wording fix.

- **v1.2.0** — Added §14 (Versioning, Releases & Changelog): source of truth, SemVer rules, PEP 440 string formats, files that must stay in sync, git tag conventions, sub-package versioning, version consistency gate, paper/thesis versioning, registry invariant versioning, changelog format, pre-commit hook, release checklist, and versioning failure modes.
- **v1.1.0** — Added §0 (Core Operating Principles), §4 (Verification Contract), §5 (Scope Discipline), §6 (Silent Failure Policy), §10 (Blocker Escalation & Honest Reporting). Expanded §3 with user authorization exception (§3.2) and tool discovery protocol (§3.3). Added regression test requirement (§5.4). Added report format (§10.2). Added quality gate checklist to §11.
- **v1.0.0** — Initial agent guidelines.

---

*Last updated: 2026-10-21*
