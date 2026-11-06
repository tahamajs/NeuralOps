You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are removing a silent-failure hazard from the training CLI. Your scope is
`neuralops/cli/train_cmd.py` (the fix), one new test file, and the docs notes
listed below. Do not redesign training, do not change hyperparameters, optimiser
choice, data loading, or checkpoint format — only make model-loading failure
fail loudly.

## Background
Exact problem, with file paths and line references (audit, 2026-11-06):
- `neuralops/cli/train_cmd.py:210-263`, function `_load_model()`: when
  `AutoModelForCausalLM` / `AutoModel` loading fails, it only does
  `logger.warning(...)` and then builds a `_FallbackLanguageModel` — a random
  4-layer `nn.TransformerEncoder` with vocab 32,000 and hidden size 512.
- `neuralops/cli/train_cmd.py:279`, `cmd_train()`: continues and trains that
  random model, then saves it under the name of the requested model.
- The correct pattern already exists in the same codebase:
  `neuralops/cli/serve_cmd.py`, `_load_serving_model`, explicitly refuses:
  "the serving command will not start with random placeholder weights."
  Copy that fail-closed behaviour into the train path.
- This directly contradicts the repo's own rule in `AGENTS.md` §0.2 (repo root):
  no placeholder / substitute models.
- The baseline from AGT-00 (`BASELINE.md`) records the pre-fix state. Read it
  first; it does not cover training behaviour, so your new tests are the
  before/after evidence here.

## Tasks
1. Read `neuralops/cli/train_cmd.py` in full (532 lines). Locate `_load_model()`
   (lines 210-263), the `_FallbackLanguageModel` class definition, every
   reference to it, and `cmd_train()` (line 279 onwards). Also read
   `neuralops/cli/serve_cmd.py` (`_load_serving_model`) and quote its refusal
   wording, so the train error message matches its tone.
2. Delete the `_FallbackLanguageModel` class and every reference to it in
   `neuralops/cli/train_cmd.py`. After this step, `grep -rn "_FallbackLanguageModel"
   neuralops/` must return zero results. Do not leave a renamed copy, a commented-out
   copy, or a differently-named random-model builder in the train path.
3. Make `_load_model()` fail closed. Every exception from
   `AutoModelForCausalLM` / `AutoModel` loading — including `OSError` for a
   missing checkpoint, network / HF-hub-offline errors, and `ValueError` for an
   invalid name — must be re-raised as an explicit error (raise a `RuntimeError`
   with `from exc`, preserving the cause) whose message contains ALL of:
   (a) the requested model name, (b) the root cause, (c) the remedy
   (check the name for a typo, check the local checkpoint path, check network /
   HF hub access, `HF_HUB_OFFLINE`), and (d) the sentence "Training will NOT
   continue on a random model." A bare `logger.warning` is NOT sufficient.
4. In `cmd_train()` (line 279), put NO `try/except` around the `_load_model()`
   call that swallows the error. The error must propagate up to `main()` and
   produce a non-zero exit code. Verify by reading the call site after your edit.
5. Grep the rest of the training path for the same pattern and REPORT (do not fix
   here): `grep -rn "FallbackLanguageModel\|random.*model\|placeholder.*model"
   neuralops/cli/ neuralops/main.py` (case-insensitive). In particular, read
   `neuralops/main.py:228,246` (retry with `local_files_only=True` after a
   load failure) and report whether it silently substitutes or retries; do NOT
   change `neuralops/main.py` in this prompt — its `run_eval` checkpoint failure
   belongs to AGT-06, and any other silent retry there is reported for a later
   prompt, not fixed here.
6. Create `tests/test_train_cmd_fallback.py` (new file; the audit notes `tests/`
   has no `cli` subdirectory, so put it at `tests/test_train_cmd_fallback.py`)
   with these tests (names exact):
   - `test_load_model_failure_raises_not_random`: mock
     `AutoModelForCausalLM.from_pretrained` to raise `OSError` (missing
     checkpoint); assert `_load_model()` raises and returns no model.
   - `test_cmd_train_fails_fast_on_missing_checkpoint`: call `cmd_train()` (or
     its argument-parsing entry) with a nonexistent model/checkpoint path, with
     model loading mocked to fail; assert it raises / exits non-zero AND that no
     checkpoint file or directory is created under `tmp_path`.
   - `test_error_message_names_model_and_refuses_random`: with a bogus model
     name such as `"definitely-not-a-real-model-xyz"`, assert the raised error's
     message contains the model name and the phrase "will NOT continue on a
     random model".
   Each test must FAIL before the fix (a random model would have been returned /
   training would have continued) and PASS after it — state this explicitly in
   your report for each test.
7. Check `README.md` (repo root) and any train docs under `docs/` for language
   that implies training on a fallback model is acceptable. If you find such
   language, remove or correct that sentence only; if you find none, say so in
   your report. Do not rewrite the training docs.
8. Do NOT add any new fallback flag in this prompt (no `--allow-random-fallback`,
   no `--synthetic-smoke-model`). If a synthetic smoke-test mode is ever wanted,
   it is a separate, explicitly approved decision — not part of this fix. The
   train path after your change has no random-model construction at all.

## Constraints / do-not-do
- No silent fallbacks of any kind, no new dependencies, no breaking change
  without a migration note. (This change IS intentionally breaking for anyone
  who relied on silent random training: record that in `CHANGELOG.md` as a
  one-line breaking-change note under the current version, 2.0.6.)
- Do not edit generated files, do not commit secrets, do not touch
  `neuralops/main.py`, `neuralops/cli/serve_cmd.py`, or any other file.
- Do not change training behaviour for models that DO load successfully.
- Do not run a real training job or download a real model in any test or
  verification step; all tests use mocks / bogus names only.

## Tests to add/run
- Create: `tests/test_train_cmd_fallback.py` (tests named in Task 6).
- Run: `python -m pytest tests/test_train_cmd_fallback.py -v`
- Run: `grep -rn "_FallbackLanguageModel" neuralops/` — expected: zero results.
- Run: `python -m ruff check neuralops/cli/train_cmd.py tests/test_train_cmd_fallback.py`
- Run: `python -c "from neuralops.cli.train_cmd import _load_model; print(_load_model)"`
  to confirm the module still imports cleanly.
- Run (negative check, mocked — do NOT download a model):
  `python -m pytest tests/test_train_cmd_fallback.py::test_load_model_failure_raises_not_random -v`

## Verification (agent MUST run and report)
- `grep -rn "_FallbackLanguageModel" neuralops/` — expected: zero results.
- `python -m pytest tests/test_train_cmd_fallback.py -v` — expected: all 3 tests
  pass; report the pass count.
- `grep -n "will NOT continue on a random model" neuralops/cli/train_cmd.py` —
  expected: present in the `_load_model` error path.
- `git diff --stat` and `git status --short` — confirm only
  `neuralops/cli/train_cmd.py`, `tests/test_train_cmd_fallback.py`,
  `CHANGELOG.md` changed (plus at most the one README/docs sentence from Task 7).
- Read the final `_load_model()` and the `cmd_train()` call site once more and
  confirm no `try/except` swallows the load error.
- Checklist to tick: [ ] fallback class deleted [ ] load failure raises with
  model name + cause + remedy [ ] no checkpoint created on failure
  [ ] `serve_cmd.py` refusal pattern matched [ ] all new tests pass
  [ ] no other file changed.

## Definition of done
- `grep -rn "_FallbackLanguageModel" neuralops/` returns zero results.
- `_load_model()` raises a `RuntimeError` (with cause chained) naming the model,
  the cause, the remedy, and stating training will NOT continue on a random model.
- `cmd_train()` with a missing checkpoint exits non-zero and creates no checkpoint.
- All three new tests pass; each would have failed before the fix.
- `CHANGELOG.md` carries the one-line breaking-change note.

## Report back
- Files changed (exact paths).
- The `serve_cmd.py` refusal wording you matched.
- Test results with pass/fail counts, and for each test why it would have failed
  before the fix.
- The `neuralops/main.py:228,246` read-only review result from Task 5 (what you
  found, verbatim quotes if it silently retries — you did not fix it here).
- What you could NOT verify (e.g. an end-to-end `neuralops train` run with a real
  model — say so explicitly; the mocked tests are not a substitute).
- Any new finding.

## Stop conditions
- If `neuralops/cli/train_cmd.py` does not contain `_FallbackLanguageModel` or
  `_load_model` where described — STOP, quote what you actually found, and report it.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If model loading turns out to happen in a different function or file than
  `_load_model()` (e.g. the real load is elsewhere and `_load_model` is dead code)
  — STOP and report the actual load path; do not fix the wrong function.
- If a decision is needed (e.g. whether a synthetic smoke-test mode should exist)
  — STOP and report the decision needed; do not invent the flag yourself.
