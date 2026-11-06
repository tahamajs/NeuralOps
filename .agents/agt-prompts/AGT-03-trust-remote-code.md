You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are closing a remote-code-execution-by-default hole. Your scope is the two
files where the `True` default lives, the CLI flag that exposes it, the run
manifest where the value must be recorded, and one new test file. Do not change
model-loading logic, do not change which models load, do not touch serving or
token handling (that is AGT-04).

## Background
Exact problem, with file paths and line references (audit, 2026-11-06):
- `neuralops/benchmarks/config.py:69`: `trust_remote_code: bool = True` in
  `BenchmarkConfig`.
- `neuralops/benchmarks/core/model_loader.py:77`:
  `load_model_and_tokenizer(..., trust_remote_code: bool = True)`, passed to
  SEVEN `from_pretrained` calls (lines 115, 123, 138, 145, 152, 163, 177).
  NOTE: the task brief also names `core/model_loader.py`. First verify which
  path exists: run `ls neuralops/benchmarks/core/model_loader.py
  neuralops/core/model_loader.py` and `grep -rn "def load_model_and_tokenizer"
  neuralops/`. The audit's line references point at
  `neuralops/benchmarks/core/model_loader.py`; if a second file with the same
  function exists, apply the same fix there and report it. If it does not exist,
  say so and continue with the benchmarks path only.
- `neuralops/benchmarks/cli.py:187`: the CLI only offers `--no-trust-remote-code`
  (opt-out). There is no opt-in flag.
- The repo's own rule (`AGENTS.md` §15, repo root) requires `trust_remote_code`
  to be explicit and recorded. The mature-project pattern (HuggingFace
  Transformers) is the same: remote code is never executed unless the caller
  explicitly opts in.
- The baseline from AGT-00 (`BASELINE.md`) does not cover this behaviour, so your
  new tests are the before/after evidence here.

## Tasks
1. Read `neuralops/benchmarks/config.py` (around line 69),
   `neuralops/benchmarks/core/model_loader.py` (lines 77 and 115, 123, 138, 145,
   152, 163, 177), and `neuralops/benchmarks/cli.py` (around line 187). Confirm
   the `True` defaults and the opt-out-only flag with your own eyes and quote
   them in your report.
2. Change the default to `False` in BOTH places:
   `neuralops/benchmarks/config.py:69` (`trust_remote_code: bool = False`) and
   `neuralops/benchmarks/core/model_loader.py:77`
   (`trust_remote_code: bool = False`). No `from_pretrained` call may rely on a
   hidden default: verify each of the seven calls at lines 115, 123, 138, 145,
   152, 163, 177 passes `trust_remote_code=` explicitly from the function
   parameter, and fix any call that does not.
3. Change the CLI from opt-out to opt-in in `neuralops/benchmarks/cli.py`
   (line 187): replace `--no-trust-remote-code` with `--trust-remote-code`
   (default off). Keep `--no-trust-remote-code` ONLY as a deprecated alias that
   sets the value to `False` and emits a `DeprecationWarning` — it must never
   change behaviour. If keeping the alias is technically awkward in the
   argparse setup, remove it instead and record the removal as a breaking change
   in `CHANGELOG.md`; do not silently keep both flags with different meanings.
   After the change, `neuralops run --help` must show `--trust-remote-code`
   (opt-in), not only `--no-trust-remote-code`.
4. When `trust_remote_code=True` IS explicitly requested, log an explicit warning
   at load time in `load_model_and_tokenizer` that includes the model name and
   the sentence "remote code WILL be executed". The warning must be logged
   through the module logger, not printed.
5. Record the value in the run manifest: find where the benchmark run manifest is
   written (run `grep -rn "manifest" neuralops/benchmarks/ --include="*.py" | head -50`
   and read the matching code — likely around the Runner / orchestrator). Add a
   `trust_remote_code: true/false` field (plus the model name) to every run
   manifest for a run that loads a model. If no manifest writer exists at all,
   report that explicitly and instead record the value in the run's result dict
   / log line — do not invent a whole new manifest system in this prompt.
6. Sweep the whole package: `grep -rn "trust_remote_code" neuralops/` and fix
   every OTHER occurrence whose default is `True` to `False` (or make it explicit
   where the call relies on a library default). One specific case: if
   `neuralops/export/exporter.py` (`load_exported_model`) calls `from_pretrained`
   without a `trust_remote_code` argument, add the explicit argument
   `trust_remote_code=False` there — and NOTHING else in that file (its error
   handling belongs to AGT-06; do not change it here). Report the full grep
   output and what you did about each hit.
7. Check `README.md` (repo root) and `docs/` for any place that presents
   `trust_remote_code=True` as the default. Correct those sentences only; if
   there are none, say so. Do not rewrite the docs.
8. Create `tests/benchmarks/test_trust_remote_code.py` (new file) with these
   tests (names exact):
   - `test_default_is_false`: assert `BenchmarkConfig().trust_remote_code is False`
     and that `inspect.signature(load_model_and_tokenizer)` has the default `False`.
   - `test_true_is_logged_and_recorded`: with `trust_remote_code=True`, assert a
     warning is logged containing the model name, and that the run manifest (or
     the result dict from Task 5) contains `trust_remote_code: true`.
   - `test_no_from_pretrained_without_explicit_flag`: a static test that reads
     `neuralops/benchmarks/core/model_loader.py` and asserts every
     `from_pretrained` call in it carries an explicit `trust_remote_code` argument.
   Each test must FAIL before the fix and PASS after it — state this explicitly
   in your report for each test.

## Constraints / do-not-do
- No silent fallbacks, no new dependencies, no breaking change without a
  migration note in `CHANGELOG.md`. (Removing or deprecating
  `--no-trust-remote-code` is a CLI change: document whichever you chose in
  `CHANGELOG.md` under the current version, 2.0.6.)
- Do not edit generated files, do not commit secrets, do not touch serving code,
  token handling, or any file not listed here.
- Never log or print a token or secret in the new warning (this prompt logs only
  the model name and the flag value).
- Do not run a real model download in any test or verification step; mock
  `from_pretrained` in tests.

## Tests to add/run
- Create: `tests/benchmarks/test_trust_remote_code.py` (tests named in Task 8).
- Run: `python -m pytest tests/benchmarks/test_trust_remote_code.py -v`
- Run: `grep -rn "trust_remote_code.*=.*True" neuralops/benchmarks/config.py neuralops/benchmarks/core/model_loader.py` — expected: zero results.
- Run: `grep -rn "trust_remote_code" neuralops/` — report the full list; no
  remaining `= True` default may exist.
- Run: `python -m neuralops run --help` (or `neuralops run --help` if the console
  script works after AGT-01) and confirm `--trust-remote-code` is listed.
- Run: `python -m ruff check neuralops/benchmarks/config.py neuralops/benchmarks/core/model_loader.py neuralops/benchmarks/cli.py tests/benchmarks/test_trust_remote_code.py`
- Run: `python -c "from neuralops.benchmarks.config import BenchmarkConfig; print(BenchmarkConfig().trust_remote_code)"` — expected: `False`.

## Verification (agent MUST run and report)
- `grep -rn "trust_remote_code.*=.*True" neuralops/benchmarks/config.py neuralops/benchmarks/core/model_loader.py` — expected: zero results.
- `python -c "from neuralops.benchmarks.config import BenchmarkConfig; c = BenchmarkConfig(); print(c.trust_remote_code)"` — expected: `False`.
- `python -m pytest tests/benchmarks/test_trust_remote_code.py -v` — expected:
  all tests pass; report the pass count.
- CLI help output showing `--trust-remote-code` (quote the relevant lines).
- `git diff --stat` and `git status --short` — confirm only
  `neuralops/benchmarks/config.py`, `neuralops/benchmarks/core/model_loader.py`,
  `neuralops/benchmarks/cli.py`, the manifest-writing file from Task 5 (if any),
  `neuralops/export/exporter.py` (one-line explicit argument only, if applicable),
  `tests/benchmarks/test_trust_remote_code.py` and `CHANGELOG.md` changed (plus
  at most the README/docs sentences from Task 7).
- Checklist to tick: [ ] both defaults are `False` [ ] all seven
  `from_pretrained` calls pass the flag explicitly [ ] CLI is opt-in
  [ ] `True` logs a warning naming the model [ ] manifest records the value
  [ ] no other `= True` default remains in `neuralops/` [ ] all new tests pass.

## Definition of done
- `BenchmarkConfig().trust_remote_code is False` and
  `load_model_and_tokenizer` defaults to `False`.
- Every `from_pretrained` call in `model_loader.py` passes `trust_remote_code`
  explicitly.
- `neuralops run --help` shows `--trust-remote-code` as an opt-in flag.
- Explicit `True` produces a logged warning with the model name and is recorded
  in the run manifest.
- `grep -rn "trust_remote_code" neuralops/` shows no remaining `True` default.
- All new tests pass; each would have failed before the fix.

## Report back
- Files changed (exact paths), including whether a second `model_loader.py`
  existed and what you did with it.
- Full `grep -rn "trust_remote_code" neuralops/` output after the fix.
- Where the run manifest is written and the exact field added (or the explicit
  statement that no manifest writer exists and what you recorded instead).
- Test results with pass/fail counts, and for each test why it would have failed
  before the fix.
- What you could NOT verify (e.g. a real end-to-end benchmark run with a remote
  model — say so explicitly; the mocked tests are not a substitute).
- Any new finding.

## Stop conditions
- If `neuralops/benchmarks/config.py:69` or
  `neuralops/benchmarks/core/model_loader.py:77` do not contain the `True`
  defaults as described — STOP, quote what you actually found, and report it.
- If any of the seven `from_pretrained` line numbers does not contain a
  `from_pretrained` call — STOP and report the actual call sites; do not assume
  the line numbers shifted harmlessly.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If a decision is needed (e.g. whether to keep the deprecated
  `--no-trust-remote-code` alias) — STOP and report the decision needed instead
  of choosing silently; the default if you must proceed is to keep it as a
  deprecated alias, not to remove it.
