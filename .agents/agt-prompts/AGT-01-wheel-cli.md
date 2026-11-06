You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are fixing the broken pip-installed CLI. The audit (2026-11-06) proved the
bug from the files, without running code: all four console scripts point at a
module that the wheel build excludes. Your job is the packaging fix plus a CI
job that catches any regression, plus a test that proves the wheel contains the
module. Do not redesign the CLI, do not rename commands, do not change CLI
behaviour — only make the installed CLI exist and work.

## Background
Exact problem, with file paths and evidence:
- `pyproject.toml`, `[project.scripts]`: `neuralops`, `neuralops-cli`, `nops`
  and `nops-cli` are all `neuralops.benchmarks.cli:main`.
- `pyproject.toml`, `[tool.setuptools.packages.find] exclude`: contains
  `"neuralops.benchmarks*"`. The module exists in the checkout
  (`neuralops/benchmarks/cli.py`, 1,766 lines) but is NOT shipped in the wheel
  or sdist. Only `neuralops-eval` (`neuralops.evaluation`) survives, because
  `neuralops.evaluation` is not excluded.
- No CI catches this: `.github/workflows/ci.yml` and `.github/workflows/test.yml`
  run from the checkout with `PYTHONPATH: .`, and
  `.github/workflows/quality_gate.yml` only runs `python -m build` + `twine check`
  without installing the wheel and without invoking any script.
- Effect: the advertised install path (`pip install neuralops`) gives
  `ModuleNotFoundError: neuralops.benchmarks` on `neuralops --help`.
- The baseline from AGT-00 (`BASELINE.md` at the repo root) records whether
  `neuralops --help` currently works from an editable install. Read it first and
  quote its CLI smoke results in your report, so the before/after is explicit.

## Tasks
1. Read `BASELINE.md` (repo root) and record the current CLI smoke results.
   Then read `pyproject.toml` in full. Confirm with your own eyes that
   `[project.scripts]` lists the four scripts above pointing at
   `neuralops.benchmarks.cli:main`, and that the `exclude` list contains
   `"neuralops.benchmarks*"`. Quote both sections in your report.
2. Decide between the two fix options and implement exactly ONE (do not do both):
   - Option A (preferred, smallest change): delete the string
     `"neuralops.benchmarks*"` from the `exclude` list in `pyproject.toml`,
     leaving `[project.scripts]` untouched.
   - Option B (only if Option A is impossible for a reason you can state):
     move the CLI entry point to `neuralops/cli/` (which is already packaged),
     point `[project.scripts]` at the new entry point, and turn
     `neuralops/benchmarks/cli.py` into a thin shim that re-exports it.
   Record which option you chose and why in `CHANGELOG.md` (see Task 5).
3. Build the wheel: `python -m build` from the repo root. Verify the fix with
   `unzip -l dist/*.whl | grep -E "benchmarks/cli|benchmarks/__init__"`.
   The wheel MUST contain `neuralops/benchmarks/cli.py`. If it does not, stop
   (see Stop conditions) — do not paper over it.
4. Create a clean venv OUTSIDE the repo (`python3 -m venv /tmp/neuralops-wheel-venv`),
   install the built wheel there (`python -m pip install dist/*.whl`), and run
   `neuralops --help`, `neuralops info`, and `python -m neuralops --help` inside
   it. All three must exit 0. Record the exit codes and the first lines of output.
   Note: `neuralops/__main__.py` also imports `neuralops.benchmarks.cli:main`,
   so the third command is a real check, not a duplicate.
5. Add a CI job to `.github/workflows/quality_gate.yml` that does exactly this
   on every run: build the wheel (`python -m build`), create a clean venv,
   `pip install dist/*.whl` in it, then run `neuralops --help` and `neuralops info`
   (and `python -m neuralops --help`). Read the existing file first and append a
   new job; do not rewrite the existing jobs, and do not change what the existing
   jobs do. The new job must FAIL the workflow if any of those three commands
   exits non-zero.
6. Create `tests/test_packaging_wheel_cli.py` with these tests (names exact):
   - `test_console_scripts_resolve_after_wheel_install`: using
     `importlib.metadata.entry_points`, assert that each of `neuralops`,
     `neuralops-cli`, `nops`, `nops-cli` resolves to a callable that can be
     imported (import the target object; do not invoke the CLI).
   - `test_benchmarks_not_excluded_from_wheel`: build the wheel into `tmp_path`
     (use `python -m build --wheel --outdir <tmp_path>` via subprocess) and assert
     that `neuralops/benchmarks/cli.py` is present inside the wheel file
     (open it with `zipfile` and check the member list).
   Both tests must FAIL on the pre-fix code (verify this by running them against
   the fix reverted in a scratch copy if needed — at minimum, reason explicitly
   in your report about why each test would have failed before, e.g. the member
   list would not contain the file) and PASS after the fix.
7. Update `CHANGELOG.md` (repo root): add an entry under the current version
   (2.0.6) recording the fix, which option (A or B) was chosen, and that a
   wheel-install CI job was added. Do not invent a new version number.

## Constraints / do-not-do
- No silent fallbacks, no new dependencies, no breaking change without a
  migration note in `CHANGELOG.md`. This prompt makes no breaking change under
  Option A; under Option B the entry-point move IS a change — document it.
- Do not edit generated files (`dist/*`, `*.egg-info/*`), do not commit secrets,
  do not touch any CLI implementation code, any other file in `neuralops/`, or
  any other workflow file.
- Do not change the console script names or their target module under Option A.
- Do not delete the `exclude` list itself — remove only the one string.
- Do not run the full test suite; run only the tests named in this prompt plus
  the verification commands.

## Tests to add/run
- Create: `tests/test_packaging_wheel_cli.py` (tests named in Task 6).
- Run: `python -m pytest tests/test_packaging_wheel_cli.py -v`
- Run: `python -m build` then `unzip -l dist/*.whl | grep benchmarks/cli`
- Run (in the clean venv): `neuralops --help`, `neuralops info`,
  `python -m neuralops --help`
- Run: `python -m ruff check tests/test_packaging_wheel_cli.py pyproject.toml`
  (ruff on the changed Python file; if ruff flags the toml file as out of scope,
  record that and move on).

## Verification (agent MUST run and report)
- `grep -n "benchmarks" pyproject.toml` — show the scripts section and the
  exclude list after the fix; the exclude list must NOT contain
  `"neuralops.benchmarks*"`.
- `python -m build && unzip -l dist/*.whl | grep benchmarks/cli` — expected:
  a line showing `neuralops/benchmarks/cli.py` inside the wheel.
- In `/tmp/neuralops-wheel-venv`: `neuralops --help` (exit 0),
  `neuralops info` (exit 0), `python -m neuralops --help` (exit 0). Quote the
  exit code of each.
- `python -m pytest tests/test_packaging_wheel_cli.py -v` — expected: all tests
  pass; report the pass count.
- `git status --short` and `git diff --stat` — confirm only `pyproject.toml`,
  `.github/workflows/quality_gate.yml`, `tests/test_packaging_wheel_cli.py`
  and `CHANGELOG.md` changed (plus build artefacts under `dist/`, which are
  not to be committed).
- Checklist to tick: [ ] wheel contains `neuralops/benchmarks/cli.py`
  [ ] clean-venv `neuralops --help` exits 0 [ ] clean-venv `neuralops info`
  exits 0 [ ] clean-venv `python -m neuralops --help` exits 0 [ ] new tests pass
  [ ] CI job added and its YAML is valid (`python -c "import yaml; yaml.safe_load(open('.github/workflows/quality_gate.yml'))"`).

## Definition of done
- `"neuralops.benchmarks*"` is gone from the `exclude` list (Option A) OR the
  entry point was moved with a shim (Option B) — exactly one of the two.
- The built wheel contains `neuralops/benchmarks/cli.py`.
- In a clean venv, `pip install dist/*.whl` followed by `neuralops --help`,
  `neuralops info` and `python -m neuralops --help` all exit 0.
- `tests/test_packaging_wheel_cli.py` exists and passes.
- `.github/workflows/quality_gate.yml` has a new job that installs the built
  wheel in a clean venv and runs `neuralops --help` and `neuralops info`.
- `CHANGELOG.md` records the fix and the option chosen.

## Report back
- Files changed (exact paths), option chosen (A or B) and why.
- Wheel check output, clean-venv results with exit codes, test pass/fail counts.
- The `BASELINE.md` CLI results you started from, for before/after comparison.
- What you could NOT verify (e.g. you cannot run the new CI job on the Mac —
  say so explicitly; the YAML validity check is not a substitute for a CI run).
- Any new finding (e.g. another script or entry point that is also broken).

## Stop conditions
- If `pyproject.toml` does not contain the four scripts or the exclude string
  exactly as described — STOP, quote what you actually found, and report it.
  Do not guess at a different fix.
- If the wheel still does not contain `neuralops/benchmarks/cli.py` after your
  edit — STOP and report the `unzip -l` output; do not try a second, different
  packaging mechanism.
- If any test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If a decision is needed (e.g. Option A turns out impossible) — STOP and report
  the decision needed instead of choosing silently.
