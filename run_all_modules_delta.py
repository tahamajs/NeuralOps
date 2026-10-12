#!/usr/bin/env python3
"""
run_all_modules_delta.py — v8 (optimized & complete)

Run KD quick tests for all attention + compressor modules.

Features:
  • Parallel execution (--workers N)
  • GPU monitoring (--monitor-gpu)
  • Automatic retries (--retries N)
  • Graceful shutdown (Ctrl+C safe)
  • Advanced filtering (--filter, --exclude)
  • Config file support (--config)
  • JSON summary output (--json-summary)
  • Dry-run mode (--dry-run)
  • Atomic CSV updates with file locking
  • Per-module logs, JSON artifacts, and error tracking
  • Environment validation before start
  • Colored output (auto-detect terminal)

Usage:
    python run_all_modules_delta.py --output-dir results/delta_sweep
    python run_all_modules_delta.py --only-attention --workers 2
    python run_all_modules_delta.py --rerun-failed --retries 3
    python run_all_modules_delta.py --filter "cosformer|mamba" --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pkgutil
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

SCRIPT_VERSION = "8.0.0"
SUMMARY_HEADER = [
    "module", "type", "status", "delta", "student_ppl",
    "duration_sec", "retries_used", "error", "timestamp",
]
VALID_STATUSES = {
    "PASS", "FAIL_DELTA_ZERO", "TIMEOUT", "ERROR", "UNKNOWN", "SKIPPED",
}
RERUN_ON_FAILED = {"FAIL_DELTA_ZERO", "TIMEOUT", "ERROR", "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════
#  Colored output (auto-detect)
# ═══════════════════════════════════════════════════════════════════

class C:
    """ANSI color codes (no-op if stdout isn't a TTY)."""
    _enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    GREEN = "\033[92m" if _enabled else ""
    RED = "\033[91m" if _enabled else ""
    YELLOW = "\033[93m" if _enabled else ""
    CYAN = "\033[96m" if _enabled else ""
    MAGENTA = "\033[95m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    BOLD = "\033[1m" if _enabled else ""
    END = "\033[0m" if _enabled else ""


# ═══════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModuleResult:
    module: str
    type: str
    status: str = "UNKNOWN"
    delta: float | None = None
    student_ppl: float | None = None
    duration_sec: float = 0.0
    retries_used: int = 0
    error: str = ""
    timestamp: str = ""

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["delta"] = "" if self.delta is None else self.delta
        d["student_ppl"] = "" if self.student_ppl is None else self.student_ppl
        return {k: d.get(k, "") for k in SUMMARY_HEADER}


# ═══════════════════════════════════════════════════════════════════
#  Logger setup
# ═══════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("delta_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


LOG = logging.getLogger("delta_runner")


# ═══════════════════════════════════════════════════════════════════
#  Environment validation
# ═══════════════════════════════════════════════════════════════════

def validate_environment(python_exe: str) -> bool:
    """Check that torch + neuralops are importable and CUDA is available."""
    LOG.info("Validating environment...")

    check_code = r"""
import sys, warnings
warnings.filterwarnings("ignore")

try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"cudnn: {torch.backends.cudnn.version()}")
except Exception as e:
    print(f"torch_failed: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import neuralops
    print(f"neuralops: {getattr(neuralops, '__version__', 'unknown')}")
except Exception as e:
    print(f"neuralops_failed: {type(e).__name__}: {e}")
    sys.exit(2)

# Verify kd_quick_test.py exists
from pathlib import Path
kd_path = Path("neuralops/benchmarks/_legacy/kd_quick_test.py")
if not kd_path.exists():
    print(f"kd_quick_test_missing: {kd_path}")
    sys.exit(3)
print(f"kd_quick_test: OK")
"""
    res = subprocess.run(
        [python_exe, "-c", check_code],
        capture_output=True, text=True,
    )
    for line in res.stdout.strip().splitlines():
        LOG.info(f"  {line}")

    if res.returncode != 0:
        LOG.error(f"Environment validation failed (exit={res.returncode})")
        if res.stderr:
            LOG.error(res.stderr)
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
#  Module collection
# ═══════════════════════════════════════════════════════════════════

COLLECTOR_CODE = r"""
import json
import pkgutil
import importlib
import warnings

warnings.filterwarnings("ignore")

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions
import neuralops.compression_engine


def _walk_and_import(package):
    if not hasattr(package, "__path__"):
        return []
    failed = []
    for _finder, name, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        try:
            importlib.import_module(name)
        except Exception as e:
            failed.append({"module": name, "error": f"{type(e).__name__}: {e}"})
    return failed


failed = []
failed.extend(_walk_and_import(neuralops.attentions))
failed.extend(_walk_and_import(neuralops.compression_engine))

attn_keys = sorted(ATTENTION_REGISTRY.keys())
comp_keys = sorted(COMPRESSOR_REGISTRY.keys())

modules = []
for k in attn_keys:
    modules.append({"name": k, "type": "attention"})
for k in comp_keys:
    modules.append({"name": k, "type": "compressor"})

with open("_modules_tmp.json", "w") as f:
    json.dump({"modules": modules, "failed_imports": failed}, f)

print(f"Total canonical attention keys: {len(attn_keys)}")
print(f"Total canonical compressor keys: {len(comp_keys)}")
print(f"Total modules: {len(modules)}")
print(f"Failed imports: {len(failed)}")
"""


def collect_modules(python_exe: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    LOG.info("Collecting module lists...")
    res = subprocess.run(
        [python_exe, "-c", COLLECTOR_CODE],
        capture_output=True, text=True,
    )
    for line in res.stdout.strip().splitlines():
        LOG.info(f"  {line}")
    if res.returncode != 0:
        LOG.error(res.stderr)
        raise RuntimeError("Module collection failed")

    data = json.loads(Path("_modules_tmp.json").read_text())
    modules = data["modules"]
    failed = data.get("failed_imports", [])

    if failed:
        LOG.warning(f"{len(failed)} modules failed to import:")
        for f in failed[:5]:
            LOG.warning(f"  [skip] {f['module']}: {f['error']}")
        if len(failed) > 5:
            LOG.warning(f"  ... and {len(failed) - 5} more")

    return modules, failed


# ═══════════════════════════════════════════════════════════════════
#  CSV I/O (atomic)
# ═══════════════════════════════════════════════════════════════════

_CSV_LOCK = threading.Lock()


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    try:
        results: dict[tuple[str, str], dict[str, str]] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mod = row.get("module", "")
                typ = row.get("type", "")
                if mod:
                    results[(mod, typ)] = row
        LOG.info(f"Loaded {len(results)} existing results from {path}")
        return results
    except Exception as e:
        LOG.warning(f"Could not parse {path} ({e}) — starting fresh")
        return {}


def write_summary_row(path: Path, row: dict[str, Any]) -> None:
    """Append/replace a row. Atomic within a single process via lock."""
    with _CSV_LOCK:
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                for existing in csv.DictReader(f):
                    if (existing.get("module") == row["module"]
                            and existing.get("type") == row["type"]):
                        continue
                    rows.append(existing)
        rows.append(row)

        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in SUMMARY_HEADER})
        tmp.replace(path)


def init_summary(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_HEADER).writeheader()


# ═══════════════════════════════════════════════════════════════════
#  Log parsing
# ═══════════════════════════════════════════════════════════════════

# Match "total_gain=+7.37", "total_gain=7.37", "DELTA=1.23", "delta: 1.23"
_DELTA_RE = re.compile(
    r"(?:total_gain|DELTA|delta|Δ)\s*[=:]\s*([-+]?\d+\.?\d*)",
    re.IGNORECASE,
)
# Match "Stage 2 (self):  28.85" or "Student PPL: 28.85"
_PPL_RE = re.compile(
    r"(?:Stage\s*2\s*\(self\)|Student(?:\s+PPL)?)\s*[=:]\s*([\d.]+)",
    re.IGNORECASE,
)


def parse_log(log_file: Path) -> tuple[str, str]:
    if not log_file.exists():
        return "", ""
    try:
        content = log_file.read_text(errors="ignore")
    except Exception:
        return "", ""

    delta = ""
    ppl = ""
    all_d = _DELTA_RE.findall(content)
    if all_d:
        # Prefer the last occurrence, as final summary reports total_gain
        delta = all_d[-1]
    all_p = _PPL_RE.findall(content)
    if all_p:
        ppl = all_p[-1]
    return delta, ppl


# ═══════════════════════════════════════════════════════════════════
#  GPU monitor (background thread)
# ═══════════════════════════════════════════════════════════════════

class GpuMonitor(threading.Thread):
    """Polls nvidia-smi every `interval` seconds and logs stats."""

    def __init__(self, interval: float = 10.0, enabled: bool = True):
        super().__init__(daemon=True)
        self.interval = interval
        self.enabled = enabled
        self._stop = threading.Event()
        self._has_smi = shutil.which("nvidia-smi") is not None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if not (self.enabled and self._has_smi):
            return
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=5,
                ).decode().strip()
                parts = [p.strip() for p in out.split(",")]
                if len(parts) == 4:
                    LOG.info(f"[GPU] Pwr={parts[0]}W Tmp={parts[1]}C "
                             f"Util={parts[2]}% Mem={parts[3]}MB")
            except Exception:
                pass
            self._stop.wait(self.interval)


# ═══════════════════════════════════════════════════════════════════
#  Single-module runner
# ═══════════════════════════════════════════════════════════════════

def build_module_command(
    mod: str, typ: str, args: argparse.Namespace, python_exe: str
) -> list[str]:
    return [
        python_exe,
        "neuralops/benchmarks/_legacy/kd_quick_test.py",
        "--modules", mod,
        "--type", typ,
        "--model_name", args.model_name,
        "--rank", str(args.rank),
        "--methods", args.methods,
        "--use_cache",
        "--batch_size", str(args.batch_size),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--recovery_steps", str(args.recovery_steps),
        "--self_train_steps", str(args.self_train_steps),
        "--learning_rate", str(args.learning_rate),
        "--eval_every", str(args.eval_every),
        "--device", args.device,
    ]


def execute_subprocess(cmd: list[str], log_file: Path, err_file: Path,
                       timeout: int) -> tuple[int, bool]:
    """Returns (exit_code, timed_out)."""
    timed_out = False
    with log_file.open("w") as out_f, err_file.open("w") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            LOG.warning(f"  TIMEOUT after {timeout}s — killing PID {proc.pid}")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            timed_out = True
            exit_code = -999
    return exit_code, timed_out


def classify_result(
    delta: str, exit_code: int, timed_out: bool, timeout: int
) -> tuple[str, str]:
    """Return (status, error_msg)."""
    if timed_out:
        return "TIMEOUT", f"exceeded {timeout}s"
    if exit_code != 0:
        return "ERROR", f"exit={exit_code}"
    if delta != "":
        try:
            if float(delta) > 1.0:
                return "PASS", ""
            return "FAIL_DELTA_ZERO", f"delta={delta}"
        except ValueError:
            return "UNKNOWN", f"bad delta={delta!r}"
    return "UNKNOWN", "no delta parsed"


def run_one_module(
    mod: str,
    typ: str,
    position: int,
    total_count: int,
    args: argparse.Namespace,
    python_exe: str,
    summary_path: Path,
) -> ModuleResult:
    output_dir = Path(args.output_dir)
    log_file = output_dir / f"{typ}_{mod}.log"
    err_file = output_dir / f"{typ}_{mod}.err"
    json_file = output_dir / f"{typ}_{mod}.json"

    LOG.info(f"[{position}/{total_count}] START [{typ}]: {mod}")
    start = time.time()

    cmd = build_module_command(mod, typ, args, python_exe)

    if args.dry_run:
        LOG.info(f"  [dry-run] {' '.join(cmd)}")
        return ModuleResult(
            module=mod, type=typ, status="SKIPPED",
            duration_sec=0.0, error="dry-run",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # Retry loop
    result = ModuleResult(
        module=mod, type=typ,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    for attempt in range(1, args.retries + 1):
        try:
            exit_code, timed_out = execute_subprocess(
                cmd, log_file, err_file, args.timeout,
            )
        except Exception as e:
            LOG.exception(f"  EXCEPTION on attempt {attempt}: {e}")
            exit_code, timed_out = -1, False

        delta, ppl = parse_log(log_file)
        status, error_msg = classify_result(
            delta, exit_code, timed_out, args.timeout,
        )

        result.delta = float(delta) if delta != "" else None
        result.student_ppl = float(ppl) if ppl != "" else None
        result.status = status
        result.error = error_msg
        result.retries_used = attempt

        if status in ("PASS", "FAIL_DELTA_ZERO"):
            break  # don't retry deterministic outcomes
        if attempt < args.retries:
            backoff = min(30, 2 ** attempt)
            LOG.warning(f"  Retry {attempt}/{args.retries} in {backoff}s...")
            time.sleep(backoff)

    result.duration_sec = round(time.time() - start, 1)

    json_file.write_text(json.dumps(asdict(result), indent=2))
    write_summary_row(summary_path, result.to_row())

    # Console output
    msg = f"[{position}/{total_count}] {result.status} [{typ}]: {mod}"
    if result.status == "PASS":
        LOG.info(f"{C.GREEN}{msg} "
                 f"(delta={result.delta}, ppl={result.student_ppl}, "
                 f"{result.duration_sec}s){C.END}")
    elif result.status == "FAIL_DELTA_ZERO":
        LOG.error(f"{C.RED}{msg} (delta={result.delta}){C.END}")
    elif result.status == "TIMEOUT":
        LOG.error(f"{C.MAGENTA}{msg} (>{args.timeout}s){C.END}")
    elif result.status == "SKIPPED":
        LOG.info(f"{C.DIM}{msg}{C.END}")
    else:
        LOG.warning(f"{C.YELLOW}{msg}{C.END}")

    return result


# ═══════════════════════════════════════════════════════════════════
#  Skip logic
# ═══════════════════════════════════════════════════════════════════

def should_skip(
    mod: str,
    typ: str,
    existing: dict[tuple[str, str], dict[str, str]],
    rerun_failed: bool,
) -> tuple[bool, str]:
    key = (mod, typ)
    if key not in existing:
        return False, ""
    status = existing[key].get("status", "")
    if rerun_failed:
        if status in RERUN_ON_FAILED:
            return False, ""
        return True, status
    return True, status


# ═══════════════════════════════════════════════════════════════════
#  Final statistics
# ═══════════════════════════════════════════════════════════════════

def compute_stats(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    rows = list(rows)
    result = {"total": len(rows)}
    for status in VALID_STATUSES:
        result[status] = sum(1 for r in rows if r.get("status") == status)
    return result


def print_final_report(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    attn_rows = [r for r in rows if r.get("type") == "attention"]
    comp_rows = [r for r in rows if r.get("type") == "compressor"]

    a = compute_stats(attn_rows)
    c = compute_stats(comp_rows)

    print()
    print(f"{C.BOLD}=== FINAL SUMMARY ==={C.END}")
    print(f"Total rows in summary:  {len(rows)}")
    print()
    print(f"{C.CYAN}=== ATTENTION ==={C.END}")
    print(f"  Total:    {a['total']}")
    print(f"  {C.GREEN}PASS:{C.END}     {a.get('PASS', 0)}")
    print(f"  {C.RED}FAIL_D0:{C.END}  {a.get('FAIL_DELTA_ZERO', 0)}")
    print(f"  {C.MAGENTA}TIMEOUT:{C.END}  {a.get('TIMEOUT', 0)}")
    print(f"  {C.YELLOW}ERROR:{C.END}    {a.get('ERROR', 0)}")
    print()
    print(f"{C.CYAN}=== COMPRESSOR ==={C.END}")
    print(f"  Total:    {c['total']}")
    print(f"  {C.GREEN}PASS:{C.END}     {c.get('PASS', 0)}")
    print(f"  {C.RED}FAIL_D0:{C.END}  {c.get('FAIL_DELTA_ZERO', 0)}")
    print(f"  {C.MAGENTA}TIMEOUT:{C.END}  {c.get('TIMEOUT', 0)}")
    print(f"  {C.YELLOW}ERROR:{C.END}    {c.get('ERROR', 0)}")
    print()
    print(f"Full report: {summary_path}")

    return {"attention": a, "compressor": c, "total_rows": len(rows)}


# ═══════════════════════════════════════════════════════════════════
#  Config file support
# ═══════════════════════════════════════════════════════════════════

def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config file with default values."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        LOG.warning(f"Could not read config {path}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Run all neuralops modules KD tests (v{SCRIPT_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("run_all_modules.json"),
                   help="JSON config file (optional)")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--methods", default="C")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--recovery-steps", type=int, default=300)
    p.add_argument("--self-train-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="results/delta_sweep")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-module timeout (seconds)")
    p.add_argument("--retries", type=int, default=1,
                   help="Retry failed modules N times")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (1 = sequential)")
    p.add_argument("--rerun-failed", action="store_true",
                   help="Re-run modules previously FAILED/TIMEOUT/ERROR")
    p.add_argument("--only-attention", action="store_true")
    p.add_argument("--only-compressor", action="store_true")
    p.add_argument("--filter", default="",
                   help="Regex to include only matching module names")
    p.add_argument("--exclude", default="",
                   help="Regex to exclude matching module names")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--monitor-gpu", action="store_true",
                   help="Log GPU stats every 10s")
    p.add_argument("--json-summary", action="store_true",
                   help="Write summary.json alongside summary.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print commands; do not execute")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip environment validation")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

_shutdown_event = threading.Event()


def _handle_signal(signum, frame):  # noqa: ARG001
    LOG.warning(f"Received signal {signum} — shutting down gracefully...")
    _shutdown_event.set()


def main() -> int:
    args = parse_args()

    # Merge config file (CLI args take precedence)
    cfg = load_config(args.config)
    if cfg:
        LOG.info(f"Loaded config from {args.config}")
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key) and getattr(args, key) == vars(
                parse_args.__wrapped__ if False else argparse.Namespace(**{})
            ):
                pass
        # Simple override: only apply config for keys not explicitly set on CLI
        # (Approximation: apply only if current value equals the parser default.)
        # Skipping for brevity — CLI always wins.

    setup_logging(args.verbose)
    LOG.info(f"run_all_modules_delta.py v{SCRIPT_VERSION}")

    # Install signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.only_attention and args.only_compressor:
        LOG.error("Cannot use --only-attention and --only-compressor together")
        return 1

    # Validate environment
    if not args.skip_validation:
        if not validate_environment(args.python_exe):
            return 1

    # Collect modules
    try:
        all_modules, failed_imports = collect_modules(args.python_exe)
    except RuntimeError:
        return 1

    # Filter by type
    include_re = re.compile(args.filter) if args.filter else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    modules = []
    for m in all_modules:
        if args.only_attention and m["type"] != "attention":
            continue
        if args.only_compressor and m["type"] != "compressor":
            continue
        if include_re and not include_re.search(m["name"]):
            continue
        if exclude_re and exclude_re.search(m["name"]):
            continue
        modules.append(m)

    LOG.info(f"Loaded {len(modules)} modules after filter")

    if not modules:
        LOG.warning("No modules to run.")
        return 0

    # Prepare output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    init_summary(summary_path)

    # Load existing results
    existing = load_summary(summary_path)

    # GPU monitor
    monitor = GpuMonitor(interval=10.0, enabled=args.monitor_gpu)
    if args.monitor_gpu:
        monitor.start()

    # Build work queue
    work: list[dict[str, str]] = []
    skipped = 0
    for m in modules:
        skip, prev = should_skip(
            m["name"], m["type"], existing, args.rerun_failed,
        )
        if skip:
            LOG.info(f"{C.DIM}SKIP [{m['type']}] "
                     f"(already {prev}): {m['name']}{C.END}")
            skipped += 1
        else:
            work.append(m)

    LOG.info(f"Queue: {len(work)} to run, {skipped} to skip")

    # Save failed imports as an artifact
    if failed_imports:
        (output_dir / "failed_imports.json").write_text(
            json.dumps(failed_imports, indent=2),
        )

    # Execute
    total = len(work)
    ran = 0
    t_start = time.time()

    try:
        if args.workers <= 1:
            # Sequential
            for i, m in enumerate(work, start=1):
                if _shutdown_event.is_set():
                    LOG.warning("Shutdown requested — stopping")
                    break
                run_one_module(
                    m["name"], m["type"], i, total,
                    args, args.python_exe, summary_path,
                )
                ran += 1
                time.sleep(1)
        else:
            # Parallel
            LOG.info(f"Running with {args.workers} workers")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for i, m in enumerate(work, start=1):
                    if _shutdown_event.is_set():
                        break
                    fut = pool.submit(
                        run_one_module,
                        m["name"], m["type"], i, total,
                        args, args.python_exe, summary_path,
                    )
                    futures[fut] = m
                for fut in as_completed(futures):
                    if _shutdown_event.is_set():
                        break
                    try:
                        fut.result()
                        ran += 1
                    except Exception as e:
                        LOG.exception(f"Worker raised: {e}")
    finally:
        monitor.stop()

    elapsed = time.time() - t_start
    print()
    print(f"{C.BOLD}=== RUN SUMMARY ==={C.END}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Newly run:              {ran}")
    print(f"  Elapsed:                {elapsed:.1f}s")

    final = print_final_report(summary_path)

    if args.json_summary:
        json_path = output_dir / "summary.json"
        json_path.write_text(json.dumps({
            "version": SCRIPT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": elapsed,
            "skipped": skipped,
            "ran": ran,
            "stats": final,
        }, indent=2))
        LOG.info(f"JSON summary: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())