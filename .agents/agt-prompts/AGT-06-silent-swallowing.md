## AGT-06 — Stop silent swallowing in data-producing paths; run_eval must fail on checkpoint failure — Phase P0 — Findings F-84
**Depends on:** AGT-00 only
**Goal:** After this prompt, no data-producing path silently swallows an error: `run_eval` fails loudly instead of evaluating the base model when a checkpoint fails to load, metadata-registration failure is logged and re-raised (or recorded as failed), and `load_exported_model` fails on weight/manifest load failure instead of continuing with unloaded weights. A static test forbids `except Exception: pass` in the fixed files.
**Context (Persian, 2-4 lines):** خطای بلعیده‌شده بدتر از خطای آشکار است: نتیجه برای مدل اشتباه گزارش می‌شود و هیچ‌کس نمی‌فهمد. قانون خود پروژه هم همین را می‌گوید. این قدم فقط مسیرهایی را درست می‌کند که داده تولید می‌کنند؛ بقیهٔ موارد در فاز بعد می‌آیند.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are removing silent error-swallowing from the DATA-PRODUCING paths only.
That is a deliberately narrow scope: the P0 subset listed in Task 2. The other
swallow sites in the same files are a later phase's work — you must list them
as out of scope in your report, not fix them here. Your files: `neuralops/api.py`
(P0 sites only), `neuralops/main.py` (P0 sites only),
`neuralops/export/exporter.py` (`load_exported_model` only), one new test file,
one new check script, and `CHANGELOG.md`.

## Background
Exact problem, with file paths and line references (audit, 2026-11-06):
- `neuralops/main.py:378-381` (`run_eval`): on checkpoint-load failure it only
  does `logger.warning("Could not load from …")` and then EVALUATES THE BASE
  MODEL instead — results are reported for the wrong model.
- `neuralops/main.py:500-501`: a compressed-model eval failure only warns and
  the pipeline continues.
- `neuralops/api.py:163-169`: failure to register `_neuralops_metadata` is
  swallowed — surgery history is silently lost, so a later replay in
  `load_exported_model` diverges from what actually ran.
- `neuralops/export/exporter.py`, `load_exported_model`: a safetensors load
  failure only warns and continues with unloaded weights; a manifest-replay
  failure only warns and falls back to a legacy replay.
- The governing rule is the repo's own `AGENTS.md` §0.2/§0.3 (repo root):
  `except Exception: pass` is forbidden in data-producing paths; use logging +
  re-raise, or an explicit `status=failed` return recorded in the ledger/manifest.
  A comment such as `# allowed-swallow: best-effort-telemetry` alone is NOT a
  justification — it must be clear what was lost and where it was recorded.
- Counts for context (do not treat as exact after your edits; verify by grep):
  `neuralops/api.py` has 12 `except Exception` blocks, `neuralops/__init__.py`
  has 6, `neuralops/main.py` has 8 (+ 4 `except Exception as`).
- The baseline from AGT-00 (`BASELINE.md`) records the pre-fix state. Read it
  first; your new tests are the before/after evidence here.

## Tasks
1. Read the exact sites before changing them: `neuralops/main.py` lines
   220-260 (load retry), 370-390 (`run_eval`), 490-510 (compressed-model eval);
   `neuralops/api.py` lines 155-175 (metadata registration); and
   `neuralops/export/exporter.py` function `load_exported_model` in full.
   Quote each site in your report before the fix.
2. Fix ONLY these P0 data-producing sites (in this order):
   a. `neuralops/main.py:378-381` (`run_eval`): on checkpoint-load failure,
      FAIL — raise the error (with the checkpoint path and the cause in the
      message). NEVER evaluate the base model as a substitute. No fallback to
      the base model under any flag in this prompt.
   b. `neuralops/main.py:500-501`: a compressed-model eval failure must return
      an explicit `status="failed"` result (recorded in the ledger/manifest if
      the pipeline has one — check how neighbouring code reports failure and
      follow that exact shape) instead of only warning and continuing.
   c. `neuralops/api.py:163-169`: a failure to register `_neuralops_metadata`
      must log an error naming the model AND re-raise — or, if the surrounding
      function has a result object with a failure status, return that explicit
      failed status with the error recorded. Pick the shape that matches the
      function's existing contract (read the callers first) and document it.
   d. `neuralops/export/exporter.py` (`load_exported_model`): a safetensors
      load failure must FAIL (raise), not warn-and-continue with unloaded
      weights; a manifest-replay failure must FAIL (raise), not warn-and-fall-back
      to a legacy replay. (Only the error handling changes here. The explicit
      `trust_remote_code=False` argument on its `from_pretrained` call belongs
      to AGT-03 — if AGT-03 already added it, leave it alone; if it has not run
      yet, do NOT add it here.)
   The replacement pattern for a swallowed error is one of exactly three shapes:
   (i) logging + re-raise; (ii) an explicit `status="failed"` return recorded in
   the ledger/manifest; (iii) a NARROW except (e.g. `ImportError` for an optional
   dependency) + logging at debug/warning level. Never write a new bare
   `except Exception: pass`.
3. ALSO read, but DO NOT FIX in this prompt — list them verbatim in your report
   under "Deferred swallow sites (later phase)":
   - `neuralops/api.py:459,495,508,540,659,663,669,672` (decode/tokenize failures
     in `TextGenerationPipeline` falling back to `str(seq.tolist())` as "decoded
     text"), `neuralops/api.py:753-755` (`list_attentions(include_extensions=True)`
     dropping FLA listing errors), `neuralops/api.py:1094-1097` (parallel-healing
     failure falling back to single-device healing — it at least logs a warning).
   - `neuralops/main.py:228,246` (tokenizer/model load failure silently retried
     with `local_files_only=True`).
   - `neuralops/__init__.py` (alias-finder and `_patch_torch_compat` swallowing
     everything), `neuralops/_optional.py` (`_require_torch` swallowing around
     `_patch_torch_compat()`), `neuralops/compression/__init__.py` (`__dir__`
     swallowing), and the subcommand-registration `try/except` blocks in
     `neuralops/cli/main.py`.
   For each site, record one line: file, lines, what it swallows, and what the
   correct fix would be. Do not change any of these files for those sites.
4. Create `tests/test_no_silent_swallow.py` (new file) with these tests (names exact):
   - `test_no_bare_except_pass_in_fixed_files`: a static test that reads
     `neuralops/api.py`, `neuralops/main.py` and `neuralops/export/exporter.py`
     and asserts there is no `except Exception: pass` (and no bare `except:`)
     left in the P0 sites fixed in Task 2. Scope the assertion to the fixed
     sites / functions (not the whole files), because the deferred sites from
     Task 3 still exist — the test must pass with the deferred sites present,
     so assert on the specific line ranges or function bodies you fixed, and
     write the assertion so a later prompt can extend it.
   - `test_run_eval_fails_on_checkpoint_failure`: mock checkpoint loading to
     fail and assert `run_eval` raises and does NOT evaluate the base model
     (assert the base-model evaluation function is never called — mock it and
     check the call count is zero).
   - `test_load_exported_model_fails_on_weight_failure`: mock the safetensors
     load to fail and assert `load_exported_model` raises instead of returning
     a model with unloaded weights.
   Each test must FAIL before the fix and PASS after it — state this explicitly
   in your report for each test.
5. Create `scripts/utils/check_no_silent_swallow.py` (new file): a small script
   that greps `neuralops/` for `except Exception: pass` and bare `except:` and
   exits non-zero if any occurrence is found OUTSIDE an explicit allow-list kept
   inside the script itself (the allow-list starts EMPTY; a later phase may add
   entries with a written justification each). Then add ONE step to
   `.github/workflows/quality_gate.yml` that runs this script (read the file
   first; append the step, do not rewrite existing jobs — note that AGT-01 also
   adds a job to this file, so re-read it fresh rather than assuming its shape).
   If the script's allow-list would have to be non-empty on day one because of
   the deferred sites in Task 3, scope the script to the three fixed files from
   Task 2 only, and say so in the script's docstring and in your report.

## Constraints / do-not-do
- No silent fallbacks, no new dependencies, no breaking change without a
  migration note in `CHANGELOG.md`. (Making `run_eval` and `load_exported_model`
  raise where they used to continue IS intentionally breaking for callers who
  relied on the silent behaviour: record it in `CHANGELOG.md` under the current
  version, 2.0.6.)
- Do not edit generated files, do not commit secrets, do not touch any file
  other than the P0 sites in `neuralops/api.py`, `neuralops/main.py`,
  `neuralops/export/exporter.py`, `tests/test_no_silent_swallow.py`,
  `scripts/utils/check_no_silent_swallow.py`,
  `.github/workflows/quality_gate.yml`, and `CHANGELOG.md`.
- Do not fix the deferred sites from Task 3, even if the fix looks trivial.
- Do not change the return type or signature of any function you touch, except
  to add the explicit failed status where Task 2(b)/(c) requires it.
- Do not run a real eval, a real training run, or a real model download in any
  test or verification step; all tests use mocks.

## Tests to add/run
- Create: `tests/test_no_silent_swallow.py` (tests named in Task 4).
- Create: `scripts/utils/check_no_silent_swallow.py` (check script from Task 5).
- Run: `python -m pytest tests/test_no_silent_swallow.py -v`
- Run: `python scripts/utils/check_no_silent_swallow.py; echo "exit=$?"` —
  expected: exit 0.
- Run: `grep -rn "except Exception: pass" neuralops/api.py neuralops/main.py neuralops/export/exporter.py` — expected: zero results in the fixed sites
  (report the full output; any remaining hit must be one of the deferred sites
  from Task 3, named explicitly).
- Run: `python -m ruff check neuralops/api.py neuralops/main.py neuralops/export/exporter.py tests/test_no_silent_swallow.py scripts/utils/check_no_silent_swallow.py`
- Run: `python -c "import neuralops.api, neuralops.export.exporter; print('imports ok')"` — expected: imports succeed.

## Verification (agent MUST run and report)
- `python -m pytest tests/test_no_silent_swallow.py -v` — expected: all tests
  pass; report the pass count.
- `python scripts/utils/check_no_silent_swallow.py; echo "exit=$?"` — expected:
  exit 0.
- The grep output from Tests above, with every remaining hit identified as a
  deferred site from Task 3.
- `git diff --stat` and `git status --short` — confirm only the files listed in
  Constraints changed.
- Checklist to tick: [ ] `run_eval` raises on checkpoint failure and never
  evaluates the base model [ ] compressed-model eval failure returns
  `status="failed"` [ ] metadata-registration failure logs + re-raises (or
  explicit failed status) [ ] `load_exported_model` raises on weight/manifest
  failure [ ] no new `except Exception: pass` added [ ] all new tests pass
  [ ] deferred sites listed, none fixed.

## Definition of done
- `run_eval` with a failing checkpoint raises; the base-model evaluation is
  never invoked (proven by the mocked test).
- Compressed-model eval failure yields an explicit failed status, not a bare warning.
- Metadata-registration failure is logged with the model name and re-raised (or
  an explicit failed status is returned).
- `load_exported_model` raises on safetensors-load and on manifest-replay failure.
- The static test forbids `except Exception: pass` in the fixed sites, and the
  CI check script passes.
- All new tests pass; each would have failed before the fix.
- `CHANGELOG.md` records the intentionally-breaking behaviour change.

## Report back
- Files changed (exact paths) and, for each fixed site, the before/after code
  shape (one line each: "was: swallow; now: raise / status=failed").
- The "Deferred swallow sites (later phase)" list from Task 3, verbatim.
- Test results with pass/fail counts, and for each test why it would have failed
  before the fix.
- The grep output after the fix, with remaining hits identified.
- What you could NOT verify (e.g. an end-to-end eval run with a real checkpoint
  — say so explicitly; the mocked tests are not a substitute).
- Any new finding.

## Stop conditions
- If `neuralops/main.py:378-381` (`run_eval`) does not contain the base-model
  fallback as described, or `load_exported_model` does not exist where described
  — STOP, quote what you actually found, and report it.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If fixing a P0 site requires changing a function's signature or return type in
  a way callers cannot tolerate — STOP and report the exact caller impact;
  do not change the signature silently.
- If a decision is needed (e.g. re-raise vs. `status="failed"` for a site whose
  function has no result object) — STOP and report the decision needed instead
  of choosing silently; the default if you must proceed is logging + re-raise.
```

---
