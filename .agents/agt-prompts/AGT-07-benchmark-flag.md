## AGT-07 — Make --benchmark honoured exactly and fix the benchmark catalog — Phase P0 — Findings F-72
**Depends on:** AGT-00 only
**Goal:** After this prompt, `neuralops run --benchmark <name>` runs exactly that benchmark and nothing else, an invalid name fails with a clear error listing the valid names instead of silently running everything, the benchmark catalog points only at files that exist, and there is exactly one canonical benchmark specification (the honest audit one) with the other two marked superseded.
**Context (Persian, 2-4 lines):** امروز دستور `--benchmark` کاملاً نادیده گرفته می‌شود و کاربر بی‌صدا همهٔ بنچمارک‌ها را اجرا می‌کند؛ یعنی هزینهٔ زمانی و محاسباتی و نتایجی که نخواسته است. کاتالوگ هم به فایل‌هایی اشاره می‌کند که وجود ندارند و سه سند هم‌روز با هم تناقض دارند. این قدم انتخاب کاربر را دوباره واقعی می‌کند.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
package lives in /Users/tahamajs/dev/Project/neuralops). Only change the files
listed. Do not touch anything else.

## Role + scope
You are fixing the benchmark selector and the benchmark catalog. Your scope is
`neuralops/benchmarks/cli.py` (the `handle_run` fix only), the catalog file
`benchmarks/benchmark_catalog.json` (repo root), the three specification
documents at the repo root, and two new test files. Do not redesign the benchmark
runner, adapters, or registry — only the selection logic, the catalog entries,
and which spec is canonical.

## Background
Exact problem, with file paths and line references (audit, 2026-11-06):
- `neuralops/benchmarks/cli.py`, around line 113:
  `subparsers.add_parser("run", aliases=["benchmark"])` defines a `--benchmark`
  argument — but `handle_run()` (lines 690-764, read it in full) builds
  `requested_modules` only from `args.modules` and `mods` only from
  `args.module` / `args.modules`. `args.benchmark` is NEVER read there, so
  `neuralops run --benchmark attention:cosformer` silently falls back to
  `mods="all"` and runs everything.
- `benchmarks/benchmark_catalog.json` (963 bytes, repo root): of ~21 scripts it
  lists, at least 8 names exist NOWHERE under `neuralops/benchmarks/` (checked
  against the full 87-entry listing): `benchmark_all_108_cuda_isolated.py`,
  `benchmark_dispatch_integrity.py`, `benchmark_engine_matrix.py`,
  `benchmark_resource_scaling.py`, `benchmark_serving_matrix.py` (only
  `serving_matrix.py` exists), `benchmark_precision_parity.py` (only
  `precision_parity.py`), `benchmark_reproducibility.py` (only
  `reproducibility.py`), `benchmark_regression_gate.py` (only
  `regression_gate.py`). The catalog's own policy line says "Root files are
  stable entry points; implementations live in neuralops.benchmarks" — the
  catalog contradicts it by pointing at non-existent root-style names.
- Three same-day (2026-11-06) specification documents at the repo root
  contradict each other:
  - `UNIFIED_BENCHMARK_SPECIFICATION_REPORT.md` (7,957 bytes) claims
    "IMPLEMENTED & PHYSICALLY VERIFIED", 8/8 tests passing, registry = 131+72,
    and that the `--benchmark` CLI is wired up.
  - `BENCHMARK_UNIFIED_AUDIT_AND_ROADMAP.md` (11,309 bytes) lists 6 OPEN GAPS —
    gap 1 (critical): `--benchmark` is parsed but ignored in `handle_run()`;
    gap 2 (high): the `benchmark` alias collides; gap 3 (high): legacy adapters
    do not inherit from `BaseBenchmarkAdapter`; gap 4 (medium): only 5 CUDA + 1
    Triton registered (of 114 / 26); gap 5 (high): no subprocess isolation;
    gap 6 (medium): no `compare` CLI — and claims "70 passed, 14 skipped,
    0 failed".
  - `BENCHMARK_UNIFIED_CLI_COMPLETE_SPEC.md` (10,718 bytes, Persian) claims
    `BENCHMARK_REGISTRY` is EMPTY (0 entries) and `OPS_REGISTRY` has 194 —
    contradicted by the code: `neuralops/benchmarks/registry.py` (around line
    270) auto-populates `BENCHMARK_REGISTRY` at import time, and the audit
    counts 209 specs (131+72+5+1).
  - The audit's verdict, which this prompt follows: the audit-and-roadmap
    document is the honest one and becomes canonical.
- `neuralops/benchmarks/results/schema.py` (732 lines) really does contain
  `BenchmarkStatus` + `normalize()` — that part of the spec report is correct;
  do not "fix" it.
- The baseline from AGT-00 (`BASELINE.md`) records the pre-fix state. Read it
  first; your new tests are the before/after evidence here.

## Tasks
1. Read `neuralops/benchmarks/cli.py` lines 100-200 (the `run` subparser and its
   `--benchmark` / `--module` / `--modules` arguments) and lines 690-764
   (`handle_run()`) in full. Quote the exact lines where `requested_modules`
   and `mods` are built, and confirm `args.benchmark` is never read. Also read
   `neuralops/benchmarks/registry.py` around line 270 to confirm how
   `BENCHMARK_REGISTRY` is populated, and find the function that resolves a
   benchmark name/selector to a runnable entry (if one exists, use it; if none
   exists, write a small `resolve_benchmark_selector` helper in
   `neuralops/benchmarks/cli.py` next to `handle_run`).
2. Fix `handle_run()` in `neuralops/benchmarks/cli.py` (lines 690-764) so the
   contract is exactly this:
   - If `args.benchmark` is given (non-empty): resolve it to the selected
     benchmark(s) ONLY. `--benchmark attention:cosformer` means exactly that
     benchmark. An empty or absent value means `all` — but explicitly, and
     logged (log the resolved selection at info level before running).
   - If `args.benchmark` names something invalid: raise a clear error
     (`ValueError` or the CLI's existing error type — check what `handle_run`
     already raises and match it) whose message contains the invalid value AND
     the list of valid benchmark names. NEVER fall back to `all` on an invalid
     value.
   - If BOTH `--benchmark` and `--module`/`--modules` are given: pick ONE rule
     and document it in the `--help` text and in your report — either (a) it is
     an error ("pass only one of --benchmark and --module/--modules"), or
     (b) the union runs. The recommended rule is (a), an error; if you choose
     (b), the help text must say so. Do not leave the combination undefined.
   - Do not change the behaviour of any other subcommand, and do not change the
     runner/executor itself — only what `handle_run` selects.
3. Fix the catalog `benchmarks/benchmark_catalog.json`: every entry must point
   at a file that REALLY exists under `neuralops/benchmarks/`. For each of the 8
   missing names, either (a) rename the entry to the real file
   (`serving_matrix.py`, `precision_parity.py`, `reproducibility.py`,
   `regression_gate.py`, and the real equivalents of the other four — verify
   each one on disk with `ls` before writing it; do not guess a filename), or
   (b) delete the entry if no such script exists or is planned. Also fix the
   catalog's policy line so it matches reality. After the fix,
   `grep -rn "benchmark_all_108_cuda_isolated\|benchmark_dispatch_integrity"
   benchmarks/benchmark_catalog.json` must return zero results (unless you
   created the corresponding files, which is NOT required by this prompt — do
   not create benchmark scripts in this prompt).
4. Choose the canonical spec: `BENCHMARK_UNIFIED_AUDIT_AND_ROADMAP.md` is
   canonical. Mark the other two as superseded by adding an explicit header at
   the very top of each (`UNIFIED_BENCHMARK_SPECIFICATION_REPORT.md` and
   `BENCHMARK_UNIFIED_CLI_COMPLETE_SPEC.md`):
   "Superseded by BENCHMARK_UNIFIED_AUDIT_AND_ROADMAP.md — do not use."
   (plus, in the spec report, a one-line correction that the `--benchmark` CLI
   was NOT wired up and that the registry is NOT empty.) Do NOT delete the two
   files in this prompt, and do NOT rewrite them.
5. Do NOT attempt the other open gaps in this prompt — list them in your report
   under "Deferred gaps (later phases, not fixed here)": the `benchmark` alias
   collision, legacy adapters not inheriting from `BaseBenchmarkAdapter`
   (`neuralops/benchmarks/adapters/` — `base.py`, `cuda_adapter.py`,
   `triton_adapter.py`, `compressor.py`, `model.py`), incomplete CUDA/Triton
   registration, subprocess isolation, and the missing `compare` CLI. Do not
   change `neuralops/benchmarks/registry.py` or any adapter file here.
6. Create two test files (names exact):
   - `tests/benchmarks/test_benchmark_flag.py` (new file under the existing
     `tests/benchmarks/` directory):
     `test_handle_run_respects_benchmark_flag` — with
     `args.benchmark="attention:cosformer"`, assert `handle_run` runs ONLY that
     benchmark, not `all` (mock the runner/executor so no real benchmark runs);
     `test_handle_run_invalid_benchmark_fails` — with an invalid value, assert
     it raises and does NOT fall back to `all`.
   - `tests/benchmarks/test_catalog_integrity.py` (new file):
     `test_catalog_points_to_existing_files` — for every entry in
     `benchmarks/benchmark_catalog.json`, assert the file exists under
     `neuralops/benchmarks/`; `test_registry_not_empty` — assert
     `BENCHMARK_REGISTRY` is non-empty after import (refuting the "0 entries"
     claim).
   Also run the existing tests `tests/benchmarks/test_registry_and_schema_integrity.py`
   and `tests/benchmarks/test_cli_devices.py` IF they exist (the spec report says
   they hold 8 tests) and confirm they still pass after your change; if they do
   not exist, say so — do not create them. If they fail, update them only if the
   failure is caused by the corrected behaviour, and explain why in your report;
   never delete them.
   Each new test must FAIL before the fix and PASS after it — state this
   explicitly in your report for each test.

## Constraints / do-not-do
- No silent fallbacks (that is the bug being fixed), no new dependencies, no
  breaking change without a migration note in `CHANGELOG.md`. (Making an invalid
  `--benchmark` value raise instead of running everything IS intentionally
  breaking for anyone who relied on the silent fallback: record it in
  `CHANGELOG.md` under the current version, 2.0.6.)
- Do not edit generated files, do not commit secrets, do not touch any file
  other than `neuralops/benchmarks/cli.py`, `benchmarks/benchmark_catalog.json`,
  the two superseded spec documents (header only),
  `tests/benchmarks/test_benchmark_flag.py`,
  `tests/benchmarks/test_catalog_integrity.py`, and `CHANGELOG.md`.
- Do not run a full benchmark suite in any test or verification step; all
  `handle_run` tests use a mocked runner. Do not create benchmark scripts.
- Do not delete any specification document.

## Tests to add/run
- Create: `tests/benchmarks/test_benchmark_flag.py`,
  `tests/benchmarks/test_catalog_integrity.py` (tests named in Task 6).
- Run: `python -m pytest tests/benchmarks/test_benchmark_flag.py tests/benchmarks/test_catalog_integrity.py -v`
- Run (if they exist): `python -m pytest tests/benchmarks/test_registry_and_schema_integrity.py tests/benchmarks/test_cli_devices.py -v`
- Run: `grep -rn "benchmark_all_108_cuda_isolated\|benchmark_dispatch_integrity" benchmarks/benchmark_catalog.json` — expected: zero results.
- Run: `python -m ruff check neuralops/benchmarks/cli.py tests/benchmarks/test_benchmark_flag.py tests/benchmarks/test_catalog_integrity.py`
- Run (selection check, mocked — do NOT run a real benchmark): check first whether
  the `run` subparser has a `--dry-run` flag (`python -m neuralops run --help`
  or `neuralops run --help` if the console script works after AGT-01). If it
  does, run `neuralops run --benchmark attention:cosformer --dry-run` and confirm
  the output names only `cosformer`, not `all`. If it does not, verify the same
  selection through the mocked `handle_run` test and say explicitly that no
  `--dry-run` flag exists.

## Verification (agent MUST run and report)
- `python -m pytest tests/benchmarks/test_benchmark_flag.py tests/benchmarks/test_catalog_integrity.py -v` — expected: all tests pass; report the pass count.
- The catalog grep from Tests above — expected: zero results.
- The `--dry-run` selection check (or the explicit statement that no `--dry-run`
  flag exists and the mocked test is the evidence).
- `head -5 UNIFIED_BENCHMARK_SPECIFICATION_REPORT.md` and
  `head -5 BENCHMARK_UNIFIED_CLI_COMPLETE_SPEC.md` — expected: both start with
  the superseded header.
- `git diff --stat` and `git status --short` — confirm only the files listed in
  Constraints changed.
- Checklist to tick: [ ] `--benchmark attention:cosformer` selects only
  `cosformer` [ ] invalid value raises with the valid list, never falls back to
  `all` [ ] `--benchmark` + `--module(s)` rule defined and documented
  [ ] every catalog entry points at an existing file [ ] canonical spec chosen,
  other two marked superseded [ ] all new tests pass [ ] deferred gaps listed,
  none attempted.

## Definition of done
- `handle_run` reads `args.benchmark`; `neuralops run --benchmark
  attention:cosformer` runs exactly that benchmark.
- An invalid `--benchmark` value raises with the valid names listed; it never
  silently runs `all`.
- The combination rule for `--benchmark` with `--module`/`--modules` is defined
  and visible in `--help`.
- Every entry in `benchmarks/benchmark_catalog.json` points at a file that exists
  under `neuralops/benchmarks/`.
- `BENCHMARK_UNIFIED_AUDIT_AND_ROADMAP.md` is canonical; the other two spec
  documents carry the superseded header.
- All new tests pass (and the two existing registry/CLI-device test files, if
  present, still pass); each new test would have failed before the fix.

## Report back
- Files changed (exact paths).
- The `--benchmark` + `--module(s)` combination rule you chose, and the exact
  `--help` text that documents it.
- For each of the 8 missing catalog names: renamed to what (with the `ls` proof)
  or deleted, and why.
- Test results with pass/fail counts, and for each new test why it would have
  failed before the fix.
- The "Deferred gaps (later phases, not fixed here)" list from Task 5.
- What you could NOT verify (e.g. a real end-to-end benchmark run — say so
  explicitly; the mocked tests are not a substitute).
- Any new finding.

## Stop conditions
- If `neuralops/benchmarks/cli.py:690-764` does not contain `handle_run` building
  `mods` from `args.module`/`args.modules` without reading `args.benchmark` —
  STOP, quote what you actually found, and report it.
- If a catalog entry's real filename cannot be verified on disk — STOP for that
  entry and report it; do not guess a filename and do not delete the entry
  silently.
- If a test that passed before (per `BASELINE.md`) now fails — STOP and report
  which test and its failure output.
- If a decision is needed (e.g. the `--benchmark` + `--module(s)` combination
  rule, or whether a missing catalog script should be created rather than
  deleted) — STOP and report the decision needed instead of choosing silently;
  the defaults if you must proceed are: combination = error, and missing
  scripts = delete the entry (do not create scripts).
```

---

### فاز ۱ (P1) — درستی + API واحد + رجیستری صادقانه

۲۷ یافته. پیش‌نیاز: P0 سبز شده باشد (به‌خصوص AGT-01 برای CLI و AGT-05 برای dispatch). در این فاز فریم‌ورک «راست می‌گوید»: یک نسخه، یک CLI، یک API، یک سلسله‌مراتب خطا، و شمارش‌های رجیستری که با README می‌خوانند.

