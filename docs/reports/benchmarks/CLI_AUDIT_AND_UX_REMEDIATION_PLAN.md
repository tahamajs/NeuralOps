# NeuralOps CLI: Complete Audit, Cleanup, and User-Experience Remediation Plan

## Executive summary

The NeuralOps CLI is powerful but currently behaves like several products
merged into one parser. The top-level command exposes benchmark execution,
model surgery, compression, generation, serving, diagnostics, export, TUI,
roofline analysis, result management, and training commands. That breadth is
useful, but it also creates a high risk of confusing defaults, inconsistent
configuration, partial output, and commands that appear successful while using
another execution path.

This document defines how to make the CLI predictable, discoverable, safe for
GPU work, and pleasant for both interactive users and CI. It is an audit and an
implementation plan; the CLI should not be declared complete until the gates at
the end have evidence.

## 1. Current CLI surface

The primary implementation is `neuralops/benchmarks/cli.py`, entered through:

```bash
python -m neuralops
python -m neuralops.benchmarks
neuralops
neuralops-cli
nops
```

The parser currently exposes a very broad command set, including:

```text
run / benchmark
resume
plan
list
verify
doctor
info
search
compress
generate
niah
benchmark-fused
transplant / surgery
compose
roofline
recommend
calibrate
inspect
ui
validate
scaffold
results
ledger
train
eval / evaluate
serve
monitor
distill
export
test
recipe
evidence
ops
compat
```

The first cleanup task is not to remove commands. It is to make every command
belong to a documented command family and give it the same conventions for
help, configuration, errors, output, dry runs, and exit codes.

## 2. Main UX problems

### 2.1 The command surface is too flat

`run`, `compress`, `generate`, `export`, `serve`, `test`, and `results` are
different products but are exposed next to each other without a clear family
structure. A new user cannot easily infer which commands are safe read-only
operations and which commands allocate GPUs, download models, write files, or
modify checkpoints.

Recommended grouping in documentation and help:

```text
neuralops benchmark ...    # benchmark execution and planning
neuralops module ...       # list, inspect, validate, scaffold
neuralops model ...        # compress, transplant, calibrate, export
neuralops run ...          # generation, serving, training workloads
neuralops system ...       # doctor, info, verify, compat
neuralops results ...      # summarize, compare, ledger, evidence
```

Backward-compatible aliases can remain, but the help output should promote the
grouped form.

### 2.2 `run` has too many unrelated flags

The benchmark parser combines module selection, model and dataset grids,
training, compression, calibration, distributed execution, tracking, Hugging
Face credentials, output policy, comparison backends, and recipe parsing.

This creates several failure modes:

- invalid combinations are accepted and fail late;
- a flag is parsed but ignored by a downstream runner;
- two flags express the same concept with different names;
- `--parallel` and `--parallel-workers` can disagree;
- `--dtype`, `--mixed-precision`, and config `precision` can disagree;
- `--output` and `--output-dir` can produce different artifact locations;
- a command intended as a benchmark can unexpectedly fine-tune a model.

The parser should be split into common options and command-family options.
Each command should reject unsupported combinations before importing a model or
allocating a GPU.

### 2.3 Configuration behavior is not obvious

The CLI declares `--config` as a YAML configuration path in its help text, while
the canonical configuration now supports JSON and YAML depending on optional
dependencies. Users need a single, explicit policy:

- JSON works with the base installation;
- YAML requires PyYAML and must give an actionable error if absent;
- CLI flags override config values only when explicitly supplied;
- the effective merged configuration is printed or saved before execution;
- unknown configuration keys fail with a useful key list;
- environment-variable expansion is explicit and safe.

The current argparse defaults make it difficult to distinguish “not supplied”
from “supplied with the default”. Use `None` as the parser default for fields
that need config precedence, then apply defaults during merge.

### 2.4 Device names are inconsistent

Some commands use `cuda`, some use `cuda:0`, some accept `mps`, and some pass
the string directly into `torch.device`. The CLI must normalize:

```text
cpu       -> cpu
cuda      -> cuda:0 or the explicitly selected CUDA index
cuda:1    -> cuda:1
mps       -> mps
auto      -> selected device recorded in metadata
```

`auto` must never silently choose CPU when the user explicitly requested CUDA.
For `--device cuda`, unavailable CUDA is a hard error. For `--device auto`, the
fallback policy may choose CPU, but it must be printed and recorded.

### 2.5 Output policy is inconsistent

Some commands use `--output`, some `--output-json`, some `--output-dir`, and
some write fixed files under `results/`. This makes parallel execution unsafe.

All commands should support:

```text
--output-dir PATH
--run-id ID
--overwrite / --no-overwrite
--format jsonl,csv,markdown
--quiet
--json
```

The default should be a unique run directory, not a shared “latest” file.
Every output must contain `config.json`, `environment.json`, and a manifest.

### 2.6 Error messages are not consistently actionable

The CLI has rich error formatting and `diagnose_error`, but not every command
routes exceptions through the same handler. A good error must say:

1. what failed;
2. which command and module were active;
3. whether an artifact is partial;
4. how to reproduce with `--verbose`;
5. the safest next command.

Tracebacks should be hidden by default for user errors and shown with
`--debug` or `--traceback`.

## 3. Command-by-command remediation

### 3.1 `neuralops benchmark run`

Make this the canonical benchmark command. Keep `neuralops run` as a backward
compatible alias only if its behavior is identical.

Recommended interface:

```bash
neuralops benchmark run \
  --config benchmarks/configs/quick_cuda.json \
  --category attention \
  --device cuda:0 \
  --output-dir results/attention_quick
```

Required behavior:

- validate config before importing optional model dependencies;
- show an execution plan with module count, shapes, dtype, device, and output;
- require `--yes` for a large or destructive run in non-interactive mode;
- fail closed on explicit CUDA requests;
- isolate every module result;
- record failures rather than relabelling them as passes;
- return non-zero if any required module fails;
- provide `--continue-on-error` only when failures remain visible and the final
  exit code still reflects them.

### 3.2 `neuralops benchmark plan`

`plan` should be side-effect free. It must not load datasets, compile kernels,
write result rows, or allocate large tensors.

It should print:

- effective config;
- selected modules and their source locations;
- expected number of cases;
- estimated memory and duration when known;
- unavailable backends;
- exact command that will execute the plan.

With `--json`, it should emit a machine-readable plan and nothing else on
stdout.

### 3.3 `neuralops benchmark resume`

Resume must validate that the original run was created by the same code and
config. It must refuse to resume when:

- the config hash changed without `--force`;
- the output directory belongs to another run;
- a result row is malformed;
- the run is already complete;
- two workers claim the same module.

Use a lock file and atomic result writes. A resume summary must identify skipped,
reused, rerun, and failed cases.

### 3.4 `neuralops module list`

This command should be the main discovery tool. It needs consistent filters:

```bash
neuralops module list --kind attention --backend cuda --supports causal
neuralops module list --kind compressor --json
```

The output should include canonical key, aliases, implementation path, family,
backend support, causal support, dtype support, and known status. Aliases must
not inflate canonical counts.

### 3.5 `neuralops module inspect`

Inspect must be read-only and must not compile a kernel by default. Add:

```text
--source
--schema
--json
--check-import
--check-backends
```

Any live check must be clearly labelled and must return a non-zero status when
the requested backend cannot load.

### 3.6 `neuralops module validate`

Validation should expose named gates rather than a single opaque “6-point”
operation:

```text
--gate forward
--gate backward
--gate finite
--gate causal
--gate parity
--gate dispatch
--all-gates
```

The report must distinguish `PASS`, `FAIL`, `SKIP_UNSUPPORTED`, and
`ERROR_INFRASTRUCTURE`. A skipped backend must never be rendered as passed.

### 3.7 `neuralops system verify` and `doctor`

These commands should be clearly different:

- `verify`: fast, read-only hardware and dependency checks;
- `doctor`: deep diagnostics, registry scans, import checks, and optional
  compilation probes.

Both should support `--json`, and `verify --device cuda` must fail if CUDA is
not available. They should print the exact Python executable and Torch/CUDA
versions.

### 3.8 `neuralops compress`

Compression is a mutating operation. It needs:

- explicit `--output-dir` required in non-interactive mode;
- refusal to overwrite unless `--overwrite` is supplied;
- compressor capability validation before model loading;
- a manifest containing original and compressed parameter counts;
- a reconstruction/parity gate;
- exact error reporting when the named compressor fails;
- no substitute compressor or generic SVD under the requested name.

### 3.9 `neuralops transplant` and `calibrate`

These commands should share a checkpoint manifest contract. The manifest must
include source model, target attention, rank, calibration data identity, seed,
software versions, and output files. Partial checkpoints must be written to a
temporary directory and atomically renamed only after validation.

### 3.10 `neuralops generate` and `serve`

Generation and serving need a clear distinction between a functional smoke
test and a performance benchmark. Add explicit modes:

```text
--mode smoke
--mode benchmark
```

Benchmark mode must warm up, synchronize CUDA, measure TTFT/ITL separately,
record token counts, and fail on non-finite output. It must not report a
throughput number when the engine was unavailable.

### 3.11 `neuralops results`

Results should become a command group with predictable subcommands:

```text
neuralops results inspect PATH
neuralops results summarize PATH
neuralops results compare BASELINE CANDIDATE --metric p95_ms
neuralops results validate PATH
neuralops results export PATH --format csv
neuralops results ledger PATH
```

Validation must reject missing metrics, malformed statuses, duplicate run IDs,
and records claiming GPU execution without GPU metadata.

### 3.12 `neuralops export`

Export must be fail-closed for every format. Never create an artifact with a
valid-looking extension when conversion failed. The command should validate the
resulting file before reporting success.

## 4. Parser design rules

### 4.1 Common options

Create one reusable parent parser for:

```text
--config
--device
--seed
--output-dir
--run-id
--json
--quiet
--verbose
--debug
--dry-run
--overwrite
```

Do not redeclare these independently in every subparser.

### 4.2 Strong validation

Validate at parse time where possible:

- positive batch size, rank, sequence length, warmup, and iterations;
- `0 <= tolerance`;
- valid comma-separated integer lists;
- no duplicate modules after alias expansion;
- `--parallel-workers >= 1`;
- incompatible device/backend combinations rejected early;
- output path is writable before expensive work begins.

### 4.3 Configuration precedence

Document and implement this exact order:

```text
built-in defaults
    < config file
    < environment variables
    < explicit CLI flags
```

The effective config must be printed in verbose mode and always saved in the
run directory. Explicit CLI flags need a sentinel default so they can override
config values without accidentally overriding them with argparse defaults.

### 4.4 Output streams

Use stdout for results and concise progress, stderr for diagnostics and logs.
With `--json`, stdout must contain valid JSON only. This is essential for shell
pipelines and CI.

## 5. Exit-code contract

Define constants and document them:

```text
0  success
1  benchmark or validation failure
2  CLI usage/configuration error
3  requested backend unavailable
4  numerical parity failure
5  partial run with recorded failures
6  output/schema error
70 unexpected internal error
```

Do not use `0` for a run in which the requested GPU backend was unavailable.

## 6. Error and logging contract

Every command should include a stable error code, for example:

```json
{
  "error_code": "CUDA_UNAVAILABLE",
  "command": "benchmark run",
  "message": "CUDA was requested but is unavailable",
  "hint": "Run `neuralops system verify --device cuda`",
  "run_id": "..."
}
```

Logging levels:

- `ERROR`: command cannot produce the requested result;
- `WARNING`: optional artifact unavailable or an explicit fallback occurred;
- `INFO`: selected device, config, progress, and output paths;
- `DEBUG`: backend attempts, compiler commands, and tracebacks.

Warnings must state the fallback level and reason. No `except: pass` is allowed
in result-producing code.

## 7. GPU and backend UX

Before a GPU benchmark starts, print and record:

```text
Python executable
Torch version
Torch CUDA version
CUDA availability
GPU name and index
driver/runtime information when available
dtype and tensor shapes
```

For the three-level dispatch hierarchy, each attempt must be visible:

```text
CUDA extension failed: <error>; trying Triton
Triton failed: <error>; trying PyTorch reference
```

The final result must identify the backend that actually executed. A PyTorch
fallback must not be labelled CUDA or Triton.

## 8. Completion and progress UX

Interactive mode should show progress bars or concise progress lines. CI mode
should use plain text with stable prefixes:

```text
NEURALOPS_RUN_START
NEURALOPS_MODULE_START name=...
NEURALOPS_MODULE_RESULT name=... status=success
NEURALOPS_RUN_SUMMARY passed=... failed=... skipped=...
```

Add `--no-progress` for log collectors and automatically disable rich output
when stdout is not a TTY.

## 9. Completion suggestions

Shell completions should derive from the parser or a single command catalog,
not maintain a manually duplicated command list. They should complete:

- command names;
- categories and backends;
- config files;
- registered modules;
- supported output formats;
- device names visible on the host.

The current completion generator should not advertise commands that are not
actually registered in the parser.

## 10. Recommended implementation order

### Phase 1: parser foundation

1. Create common parent parsers.
2. Normalize device, dtype, list, and path parsing.
3. Add explicit exit-code constants.
4. Add config precedence and effective-config serialization.
5. Add `--json`, `--quiet`, `--verbose`, `--debug`, and `--dry-run` behavior.

### Phase 2: benchmark command

1. Make `neuralops benchmark run` canonical.
2. Move all standard benchmark dispatch into one suite runner.
3. Keep root scripts as thin wrappers.
4. Validate device/backend before execution.
5. Add unique run directories and run locks.

### Phase 3: results and errors

1. Add `results validate`, `compare`, and `summarize`.
2. Validate typed schema before ingestion.
3. Separate warnings from failures.
4. Make JSON stdout pure JSON.

### Phase 4: module and system UX

1. Group list/inspect/validate/scaffold.
2. Improve `verify` and `doctor` output.
3. Add machine-readable diagnostics.

### Phase 5: migration and cleanup

1. Build a command migration table for `scripts/benchmarks`.
2. Convert wrappers to call package implementations.
3. Mark non-migrated scripts legacy.
4. Remove only scripts with no callers, no catalog entry, and no historical
   preservation requirement.

## 11. Required tests

### Parser tests

- every advertised command parses;
- aliases resolve to the same handler;
- invalid combinations fail before execution;
- config values and explicit flags merge correctly;
- comma-separated values reject malformed tokens;
- help output contains no unregistered command;
- completion commands match parser commands.

### Backend tests

- explicit CUDA unavailable raises and returns exit code 3;
- `auto` behavior is explicit and recorded;
- backend fallback emits warnings and records the actual backend;
- unavailable Triton/CUDA never becomes `PASS`.

### Output tests

- unique run directory is created;
- effective config and environment metadata are written;
- JSON mode produces parseable JSON only;
- duplicate run IDs are refused;
- malformed result files are rejected;
- missing metrics raise instead of defaulting.

### UX tests

- `--help` succeeds for every command;
- non-TTY output contains no terminal escape sequences;
- `--quiet` suppresses progress but preserves errors;
- `--verbose` includes backend and timing diagnostics;
- `--debug` includes the original traceback.

## 12. Acceptance commands

Host-side static checks:

```bash
ruff check neuralops/benchmarks benchmarks tests/benchmarks
bash -n benchmarks/run_suite.sh scripts/*.pbs
git diff --check
```

CPU-safe/authorized environment checks:

```bash
python -m neuralops --help
python -m neuralops.benchmarks --help
pytest tests/benchmarks/test_benchmark_contracts.py -v --tb=short
```

GPU checks on the actual GPU host:

```bash
python -m neuralops system verify --device cuda --json
python -m neuralops benchmark plan --config benchmarks/configs/quick_cuda.json --json
python -m neuralops benchmark run --config benchmarks/configs/quick_cuda.json
pytest tests/gpu_exhaustive -v --tb=short
```

Every GPU report must include the final pytest summary and exit code. A partial
log is not evidence of completion.

## 13. Definition of done

The CLI cleanup is complete only when:

- all advertised commands are registered and documented;
- common options have one implementation;
- config precedence is deterministic and tested;
- explicit GPU requests fail closed;
- JSON output is machine-readable and uncontaminated;
- every run has isolated, typed artifacts;
- exit codes distinguish usage, backend, numerical, and benchmark failures;
- completions match the parser;
- legacy scripts have a documented migration status;
- all parser, backend, output, and UX tests pass;
- the exhaustive GPU suite has a final summary;
- no command reports success for an artifact it did not actually produce.

Until these conditions are met, the CLI should be described as under
remediation rather than production-complete.

## 14. Concrete implementation blueprint

The following sections turn the policy into code-level work. The examples are
intended as contracts: names may be adjusted to match existing modules, but the
behavior must remain the same.

### 14.1 Proposed package layout

Add the following focused modules under `neuralops/benchmarks/cli/` rather than
continuing to grow one 1,000-line parser file:

```text
neuralops/benchmarks/cli/
  __init__.py
  app.py                 # main() and top-level parser
  common.py              # shared options and parser helpers
  config_merge.py        # defaults/config/env/CLI precedence
  devices.py             # device normalization and preflight
  errors.py              # typed errors and exit codes
  output.py              # stdout/stderr and JSON discipline
  locks.py               # run-directory and run-id locking
  commands/
    benchmark.py
    module.py
    system.py
    results.py
    model.py
```

If moving the existing file immediately is too risky, implement these helpers
in existing files first, add tests, and migrate the parser incrementally.

### 14.2 Typed exit codes

Create one source of truth instead of returning arbitrary integers from command
handlers:

```python
# neuralops/benchmarks/cli/errors.py
from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    USAGE = 2
    BACKEND_UNAVAILABLE = 3
    NUMERICAL_FAILURE = 4
    PARTIAL = 5
    OUTPUT_FAILURE = 6
    INTERNAL = 70


class CliError(Exception):
    """Expected user-facing error with a stable machine-readable code."""

    def __init__(self, message: str, code: ExitCode, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint


class BackendUnavailableError(CliError):
    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message, ExitCode.BACKEND_UNAVAILABLE, hint)


class NumericalGateError(CliError):
    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message, ExitCode.NUMERICAL_FAILURE, hint)
```

The application boundary should catch only `CliError` and known exceptions. It
must not turn every exception into a success return:

```python
def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(dispatch(args))
    except CliError as exc:
        emit_error(exc)
        return int(exc.code)
    except KeyboardInterrupt:
        emit_error(CliError("Interrupted by user", ExitCode.FAILURE))
        return int(ExitCode.FAILURE)
    except Exception as exc:
        emit_internal_error(exc)
        return int(ExitCode.INTERNAL)
```

### 14.3 Configuration merge implementation

The main bug to avoid is allowing argparse defaults to overwrite values loaded
from a config file. Use `None` for parser defaults and merge explicitly:

```python
# neuralops/benchmarks/cli/config_merge.py
from dataclasses import replace
import os
from typing import Any

from neuralops.benchmarks.config import BenchmarkConfig


def parse_csv(value: str, cast: type[Any]) -> list[Any]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise ValueError("comma-separated value must contain at least one item")
    try:
        return [cast(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated value: {value!r}") from exc


def cli_overrides(namespace: Any) -> dict[str, Any]:
    raw = vars(namespace)
    aliases = {"dtype": "precision", "batch_size": "batch_sizes", "seq_len": "seq_lens"}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or key in {"config", "subcommand"}:
            continue
        target = aliases.get(key, key)
        result[target] = value
    if isinstance(result.get("batch_sizes"), str):
        result["batch_sizes"] = parse_csv(result["batch_sizes"], int)
    if isinstance(result.get("seq_lens"), str):
        result["seq_lens"] = parse_csv(result["seq_lens"], int)
    return result


def load_effective_config(args: Any) -> BenchmarkConfig:
    config = BenchmarkConfig.from_file(args.config) if args.config else BenchmarkConfig()
    values = config.to_dict()
    values.update(cli_overrides(args))
    if os.environ.get("NEURALOPS_DEVICE") and "device" not in cli_overrides(args):
        values["device"] = os.environ["NEURALOPS_DEVICE"]
    return BenchmarkConfig.from_dict(values)
```

Before accepting unknown config keys, produce a useful error:

```python
valid = set(BenchmarkConfig.__dataclass_fields__)
unknown = sorted(set(data) - valid - {"dtype"})
if unknown:
    raise CliError(
        f"Unknown benchmark config key(s): {', '.join(unknown)}",
        ExitCode.USAGE,
        "Run `neuralops benchmark config-schema` to see valid keys.",
    )
```

### 14.4 Device normalization and preflight

Do not scatter `torch.cuda.is_available()` checks throughout commands. Use one
preflight function:

```python
# neuralops/benchmarks/cli/devices.py
from dataclasses import dataclass
import torch

from .errors import BackendUnavailableError


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    kind: str
    index: int | None
    name: str | None
    cuda_version: str | None


def resolve_device(requested: str, *, allow_auto: bool = True) -> DeviceInfo:
    requested = requested.lower().strip()
    if requested == "auto":
        if not allow_auto:
            raise ValueError("auto device selection is disabled for this command")
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA was requested but is unavailable; refusing CPU fallback",
                "Run `neuralops system verify --device cuda`.",
            )
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise BackendUnavailableError(f"CUDA device index {index} does not exist")
        return DeviceInfo(
            requested=requested,
            resolved=f"cuda:{index}",
            kind="cuda",
            index=index,
            name=torch.cuda.get_device_name(index),
            cuda_version=torch.version.cuda,
        )

    if requested == "mps" and not torch.backends.mps.is_available():
        raise BackendUnavailableError("MPS was requested but is unavailable")
    if requested not in {"cpu", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return DeviceInfo(requested=requested, resolved=requested, kind=requested,
                      index=None, name=None, cuda_version=None)
```

Every benchmark should store `DeviceInfo` in its environment metadata and use
the resolved device object for allocations.

### 14.5 Shared parser construction

Build common options once:

```python
def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default=None,
                        help="cpu, mps, cuda, cuda:N, or auto")
    parser.add_argument("--seed", type=positive_int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def add_benchmark_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", choices=KNOWN_CATEGORIES)
    parser.add_argument("--modules")
    parser.add_argument("--batch-sizes")
    parser.add_argument("--seq-lens")
    parser.add_argument("--dtype", dest="precision")
    parser.add_argument("--warmup", type=non_negative_int)
    parser.add_argument("--iters", type=positive_int)
    parser.add_argument("--parallel-workers", type=positive_int)
```

Use `parser.set_defaults(handler=...)` so dispatch does not depend on a large
chain of string comparisons.

### 14.6 Command dispatch table

Prefer a table over a fragile if/elif chain:

```python
HANDLERS = {
    ("benchmark", "run"): run_benchmark_command,
    ("benchmark", "plan"): plan_benchmark_command,
    ("module", "list"): list_modules_command,
    ("module", "inspect"): inspect_module_command,
    ("system", "verify"): verify_system_command,
    ("results", "compare"): compare_results_command,
}


def dispatch(args: argparse.Namespace) -> int:
    key = (args.group, args.command)
    try:
        handler = HANDLERS[key]
    except KeyError as exc:
        raise CliError(f"No handler registered for {key}", ExitCode.INTERNAL) from exc
    return handler(args)
```

The parser test should compare all help-advertised command names with the keys
in the dispatch table.

### 14.7 Run directory and locking

Parallel execution requires atomic ownership:

```python
# neuralops/benchmarks/cli/locks.py
from pathlib import Path
import json
import os
import time


def create_run_directory(root: Path, run_id: str, overwrite: bool = False) -> Path:
    path = root / run_id
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"run directory already exists: {path}; use --run-id another-id "
            "or --overwrite only if replacement is intentional"
        )
    path.mkdir(parents=True, exist_ok=overwrite)
    lock = path / ".active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"run is already active: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": time.time()}, stream)
    return path


def release_run_directory(path: Path) -> None:
    (path / ".active.lock").unlink(missing_ok=True)
```

Use a `try/finally` around the command so locks are released even after a
failure. Never write two processes to the same `results/latest` file.

### 14.8 Atomic result writes

Write to a temporary file in the same directory and rename it:

```python
def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
```

If a process dies halfway through a write, result validation must ignore the
temporary file and report the run as incomplete rather than treating partial
JSON as a valid pass.

### 14.9 Result status model

Use a constrained status vocabulary:

```python
ResultStatus = Literal[
    "success",
    "FAIL",
    "FAIL_NUMERICS",
    "FAIL_BACKWARD",
    "FAIL_DISPATCH",
    "SKIPPED_UNSUPPORTED",
    "SKIPPED_OPTIONAL_DEPENDENCY",
    "INCOMPLETE",
]
```

Do not mix `PASS`, `pass`, `success`, and `ok` across exporters. Normalize once
at result creation and reject unknown statuses on ingestion.

### 14.10 Numerical parity helper

Centralize the numerical gate:

```python
def check_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerance: float,
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise NumericalGateError(
            f"shape mismatch: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )
    if not torch.isfinite(candidate).all():
        raise NumericalGateError("candidate contains NaN or Inf")
    max_diff = float((candidate.float() - reference.float()).abs().max().item())
    passed = max_diff <= tolerance
    return {"max_diff": max_diff, "tolerance": tolerance, "passed": passed}
```

Call it for CUDA and Triton outputs before timing results are marked usable.
Timing a numerically invalid kernel is allowed only as a diagnostic, never as a
successful benchmark.

### 14.11 Backward validation helper

The helper must inspect all parameters and the input gradient:

```python
def validate_backward(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    output = module(*inputs)
    loss = output.float().square().mean()
    loss.backward()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"non-finite gradient for {name}")
    for index, tensor in enumerate(inputs):
        if tensor.requires_grad and tensor.grad is None:
            raise AssertionError(f"missing input gradient at index {index}")
```

### 14.12 Standard command output

Interactive output may be rich, but machine output must be strict:

```python
def emit(payload: object, *, json_mode: bool, quiet: bool = False) -> None:
    if json_mode:
        print(json.dumps(payload, sort_keys=True))
        return
    if not quiet:
        console.print(payload)
```

Never print a warning, progress line, or traceback to stdout in JSON mode.
Warnings belong on stderr.

### 14.13 Shell completion generation

Do not keep a manually typed command list. Generate completion names from the
parser:

```python
def parser_command_names(parser: argparse.ArgumentParser) -> list[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    return []
```

The implementation can use a small public helper around argparse internals,
but the test must fail if a completion command is not parseable.

## 15. Exact migration table to create

Create `benchmarks/CLI_MIGRATION_MATRIX.csv` with these columns:

```text
legacy_path,canonical_handler,category,flags_mapped,output_schema,
status,blocking_issue,test_file,owner
```

Classify every file in `scripts/benchmarks/` as:

- `MIGRATE`: active and suitable for canonicalization;
- `WRAPPER`: keep only as a compatibility entry point;
- `HISTORICAL`: retain but exclude from standard suite;
- `REMOVE_AFTER_REVIEW`: no callers, no unique behavior, no documentation use.

Do not delete a historical script until `rg` proves there are no callers and
the migration matrix records its replacement.

## 16. Detailed test matrix

Add or extend tests under `tests/benchmarks/`:

```text
test_cli_parser.py
  test_every_command_has_handler
  test_aliases_share_handler
  test_help_has_no_stale_commands
  test_json_output_is_single_json_document

test_cli_config_merge.py
  test_defaults_then_config_then_cli_precedence
  test_unknown_key_is_rejected
  test_dtype_alias_is_normalized
  test_csv_lists_are_validated

test_cli_devices.py
  test_explicit_cuda_unavailable_is_error
  test_auto_device_is_recorded
  test_invalid_cuda_index_is_error

test_cli_outputs.py
  test_run_directory_is_unique
  test_active_lock_prevents_duplicate_run
  test_atomic_result_write
  test_missing_metric_is_not_defaulted

test_cli_numerics.py
  test_parity_passes_within_tolerance
  test_parity_returns_fail_numerics_above_tolerance
  test_nan_candidate_fails

test_cli_completion.py
  test_completion_commands_are_parseable
```

GPU-only tests must live in the GPU suite and must report `SKIPPED_UNSUPPORTED`
only when the requested hardware capability is genuinely absent.

## 17. Recommended user workflows after remediation

### Inspect the environment

```bash
neuralops system verify --device cuda --json
neuralops system doctor --json --output-dir results/doctor
```

### Preview a benchmark

```bash
neuralops benchmark plan \
  --config benchmarks/configs/quick_cuda.json \
  --device cuda:0 \
  --json
```

### Run a quick correctness benchmark

```bash
neuralops benchmark run \
  --config benchmarks/configs/quick_cuda.json \
  --category numerics \
  --output-dir results/quick_numerics \
  --verbose
```

### Run an exhaustive suite

```bash
neuralops benchmark run \
  --config benchmarks/configs/exhaustive_cuda.json \
  --category all \
  --output-dir results/exhaustive_$(date +%Y%m%d_%H%M%S) \
  --yes
```

### Compare two runs

```bash
neuralops results validate results/baseline
neuralops results validate results/candidate
neuralops results compare results/baseline results/candidate \
  --metric p95_ms \
  --max-regression 5
```

## 18. Final implementation checklist

```text
[ ] Common parser options exist once
[ ] Config merge precedence is tested
[ ] Unknown config keys fail early
[ ] Device resolution is centralized
[ ] Explicit CUDA requests fail closed
[ ] Exit codes are typed and documented
[ ] JSON stdout contains JSON only
[ ] Run directories are unique and locked
[ ] Result writes are atomic
[ ] Status vocabulary is normalized
[ ] Numerical parity helper is shared
[ ] Backward helper rejects None/NaN gradients
[ ] Completion names come from the parser
[ ] Every command has a handler
[ ] Legacy scripts have migration status
[ ] CLI migration matrix is complete
[ ] Parser tests pass
[ ] Backend tests pass
[ ] Output/schema tests pass
[ ] GPU benchmark has final summary and exit code
[ ] ruff, bash -n, diff check, and repository audits pass
```
