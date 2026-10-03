#!/usr/bin/env python3
"""
Master Orchestrator for Comprehensive Model Evaluation.

Sequentially executes and aggregates:
  1. Perplexity (PPL) on WikiText-2 (via run_gpt2_sweep.py)
  2. LM Evaluation Harness reasoning/QA tasks (via run_lm_eval_harness.py)
  3. RULER Needle-in-a-Haystack (NIAH) context retrieval (via run_ruler_benchmark.py)

Outputs:
  - <output_dir>/aggregate_benchmarks.csv
  - <output_dir>/aggregate_benchmarks.json

CSV Schema:
  module_name, ppl, hellaswag_acc, arc_easy_acc, arc_challenge_acc,
  winogrande_acc, gsm8k_em, niah_512, niah_1024, niah_2048, niah_4096,
  status, error, timestamp
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT), str(ROOT / "neuralops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("BenchmarkMaster")

AGGREGATE_FIELDS: list[str] = [
    "module_name",
    "ppl",
    "hellaswag_acc",
    "arc_easy_acc",
    "arc_challenge_acc",
    "winogrande_acc",
    "gsm8k_em",
    "niah_512",
    "niah_1024",
    "niah_2048",
    "niah_4096",
    "status",
    "error",
    "timestamp",
]


def discover_modules(category: str = "all") -> list[str]:
    """Discover all compatible attention and compressor modules."""
    try:
        from neuralops.benchmarks.run_gpt2_sweep import get_all_decoder_modules

        return get_all_decoder_modules(category)
    except Exception:
        pass

    try:
        from neuralops.core.registry import (
            ATTENTION_REGISTRY,
            COMPRESSOR_REGISTRY,
            is_helper_module,
        )

        items: list[str] = ["full"]
        for name in list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys()):
            if name != "full" and not is_helper_module(name) and name not in items:
                items.append(name)
        return items
    except Exception:
        return ["full"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Orchestrator: Unified evaluation across PPL, LM-Eval, and RULER NIAH."
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["all"],
        help="List or comma-separated module names (or 'all' for all registered modules)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/gpt2_wikitext",
        help="Pretrained model path (default: models/gpt2_wikitext)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Rank for low-rank approximation (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for benchmarks (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/07_misc_tests/master_benchmarks",
        help="Directory to save aggregate results (default: results/07_misc_tests/master_benchmarks)",
    )
    parser.add_argument(
        "--ppl_dir",
        type=str,
        default="results/06_adhoc_experiments/perplexity_seqlen",
        help="Directory where PPL sweep results reside (default: results/06_adhoc_experiments/perplexity_seqlen)",
    )
    parser.add_argument(
        "--lmeval_dir",
        type=str,
        default="results/03_zeroshot_evals",
        help="Directory where LM Eval Harness results reside (default: results/03_zeroshot_evals)",
    )
    parser.add_argument(
        "--ruler_dir",
        type=str,
        default="results/06_adhoc_experiments/ruler_benchmark",
        help="Directory where RULER NIAH results reside (default: results/06_adhoc_experiments/ruler_benchmark)",
    )
    parser.add_argument(
        "--skip_ppl",
        action="store_true",
        help="Skip PPL benchmark stage",
    )
    parser.add_argument(
        "--skip_lmeval",
        action="store_true",
        help="Skip LM Eval Harness benchmark stage",
    )
    parser.add_argument(
        "--skip_ruler",
        action="store_true",
        help="Skip RULER NIAH benchmark stage",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run lightweight smoke test across all runners",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running existing benchmark results",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Do not skip existing results",
    )
    return parser.parse_args(argv)


def run_command(cmd: list[str], desc: str) -> tuple[int, str, str]:
    """Run a subprocess command safely, capturing returncode and stderr."""
    LOG.info("Executing: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        LOG.warning(
            "%s failed (code %d):\n%s", desc, proc.returncode, proc.stderr[-500:]
        )
    return proc.returncode, proc.stdout, proc.stderr


def upsert_aggregate_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Atomic upsert of row into aggregate CSV."""
    rows: dict[str, dict[str, Any]] = {}
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    name = r.get("module_name", "").strip()
                    if name:
                        rows[name] = r
        except Exception as e:
            LOG.warning("Could not read existing aggregate CSV %s: %s", csv_path, e)

    rows[row["module_name"]] = row

    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        for r in rows.values():
            writer.writerow({k: r.get(k, "") for k in AGGREGATE_FIELDS})
    tmp_path.replace(csv_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve module list
    parsed: list[str] = []
    for item in args.modules:
        parsed.extend([m.strip() for m in str(item).split(",") if m.strip()])
    if not parsed or parsed == ["all"]:
        module_list = discover_modules("all")
    else:
        module_list = parsed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "aggregate_benchmarks.csv"
    json_file = out_dir / "aggregate_benchmarks.json"

    ppl_dir = Path(args.ppl_dir)
    lmeval_dir = Path(args.lmeval_dir)
    ruler_dir = Path(args.ruler_dir)

    LOG.info("=" * 78)
    LOG.info("MASTER BENCHMARK ORCHESTRATOR")
    LOG.info("  Modules:       %d (%s...)", len(module_list), module_list[:5])
    LOG.info("  Model Path:    %s", args.model_path)
    LOG.info("  Rank:          %d", args.rank)
    LOG.info("  Device:        %s", args.device)
    LOG.info("  Smoke Test:    %s", args.smoke_test)
    LOG.info("  Output Dir:    %s", out_dir)
    LOG.info("=" * 78)

    all_rows: list[dict[str, Any]] = []

    for mod in module_list:
        LOG.info("\n>>> Processing Module: %s <<<", mod)
        errors: list[str] = []

        ppl_val: Any = ""
        hellaswag_acc: Any = ""
        arc_easy_acc: Any = ""
        arc_challenge_acc: Any = ""
        winogrande_acc: Any = ""
        gsm8k_em: Any = ""
        niah_512: Any = ""
        niah_1024: Any = ""
        niah_2048: Any = ""
        niah_4096: Any = ""

        # -------------------------------------------------------------------
        # 1. Perplexity Benchmark
        # -------------------------------------------------------------------
        if not args.skip_ppl:
            ppl_res_file = ppl_dir / mod / "result.json"
            need_ppl = args.force or args.no_skip or not ppl_res_file.exists()

            if need_ppl:
                ppl_cmd = [
                    sys.executable,
                    str(CODES / "benchmarks" / "run_gpt2_sweep.py"),
                    "--modules",
                    mod,
                    "--model_path",
                    args.model_path,
                    "--rank",
                    str(args.rank),
                    "--device",
                    args.device,
                    "--output_dir",
                    str(ppl_dir),
                ]
                if args.smoke_test:
                    ppl_cmd.extend(["--max_eval_tokens", "2000"])
                code, _, stderr = run_command(ppl_cmd, f"PPL runner for {mod}")
                if code != 0:
                    errors.append(f"PPL failed (exit {code})")

            if ppl_res_file.exists():
                try:
                    res_data = json.loads(ppl_res_file.read_text(encoding="utf-8"))
                    ppl_val = res_data.get("perplexity", "")
                except Exception as e:
                    errors.append(f"Failed to read PPL result: {e}")

        # -------------------------------------------------------------------
        # 2. LM Evaluation Harness Benchmark
        # -------------------------------------------------------------------
        if not args.skip_lmeval:
            lmeval_res_file = lmeval_dir / f"{mod}_lmeval.json"
            need_lmeval = args.force or args.no_skip or not lmeval_res_file.exists()

            if need_lmeval:
                lmeval_cmd = [
                    sys.executable,
                    str(CODES / "evaluation" / "run_lm_eval_harness.py"),
                    "--module",
                    mod,
                    "--model_path",
                    args.model_path,
                    "--rank",
                    str(args.rank),
                    "--device",
                    args.device,
                    "--output_dir",
                    str(lmeval_dir),
                ]
                if args.smoke_test:
                    lmeval_cmd.append("--smoke_test")
                code, _, stderr = run_command(lmeval_cmd, f"LM-Eval runner for {mod}")
                if code != 0:
                    errors.append(f"LM-Eval failed (exit {code})")

            if lmeval_res_file.exists():
                try:
                    res_data = json.loads(lmeval_res_file.read_text(encoding="utf-8"))
                    tasks = res_data.get("tasks", {})
                    hs = tasks.get("hellaswag", {})
                    ae = tasks.get("arc_easy", {})
                    ac = tasks.get("arc_challenge", {})
                    wg = tasks.get("winogrande", {})
                    gs = tasks.get("gsm8k", {})

                    hellaswag_acc = hs.get("acc", hs.get("acc_norm", ""))
                    arc_easy_acc = ae.get("acc", ae.get("acc_norm", ""))
                    arc_challenge_acc = ac.get("acc_norm", ac.get("acc", ""))
                    winogrande_acc = wg.get("acc", "")
                    gsm8k_em = gs.get("exact_match", gs.get("acc", ""))
                except Exception as e:
                    errors.append(f"Failed to read LM-Eval result: {e}")

        # -------------------------------------------------------------------
        # 3. RULER NIAH Benchmark
        # -------------------------------------------------------------------
        if not args.skip_ruler:
            ruler_res_file = ruler_dir / f"{mod}_niah.json"
            need_ruler = args.force or args.no_skip or not ruler_res_file.exists()

            if need_ruler:
                ruler_cmd = [
                    sys.executable,
                    str(CODES / "benchmarks" / "run_ruler_benchmark.py"),
                    "--module",
                    mod,
                    "--model_path",
                    args.model_path,
                    "--rank",
                    str(args.rank),
                    "--device",
                    args.device,
                    "--output_dir",
                    str(ruler_dir),
                ]
                if args.smoke_test:
                    ruler_cmd.append("--smoke_test")
                code, _, stderr = run_command(ruler_cmd, f"RULER runner for {mod}")
                if code != 0:
                    errors.append(f"RULER failed (exit {code})")

            if ruler_res_file.exists():
                try:
                    res_data = json.loads(ruler_res_file.read_text(encoding="utf-8"))
                    results = res_data.get("results", {})
                    niah_512 = results.get("512", {}).get("accuracy", "")
                    niah_1024 = results.get("1024", {}).get("accuracy", "")
                    niah_2048 = results.get("2048", {}).get("accuracy", "")
                    niah_4096 = results.get("4096", {}).get("accuracy", "")
                except Exception as e:
                    errors.append(f"Failed to read RULER result: {e}")

        # -------------------------------------------------------------------
        # 4. Status Determination & Aggregation
        # -------------------------------------------------------------------
        if not errors:
            status = "PASS"
        elif any(v != "" for v in [ppl_val, hellaswag_acc, niah_512]):
            status = "PARTIAL_SUCCESS"
        else:
            status = "FAIL"

        row: dict[str, Any] = {
            "module_name": mod,
            "ppl": ppl_val,
            "hellaswag_acc": hellaswag_acc,
            "arc_easy_acc": arc_easy_acc,
            "arc_challenge_acc": arc_challenge_acc,
            "winogrande_acc": winogrande_acc,
            "gsm8k_em": gsm8k_em,
            "niah_512": niah_512,
            "niah_1024": niah_1024,
            "niah_2048": niah_2048,
            "niah_4096": niah_4096,
            "status": status,
            "error": "; ".join(errors),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        upsert_aggregate_row(csv_file, row)
        all_rows.append(row)
        LOG.info("Row recorded for %s: status=%s", mod, status)

    # Write aggregate JSON
    json_file.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    LOG.info("Master benchmark aggregation complete: %s and %s", csv_file, json_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
