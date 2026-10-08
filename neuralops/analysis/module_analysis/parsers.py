"""Parsers and resolvers for module analysis benchmark records."""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from neuralops.analysis.module_analysis.constants import (
    ALIASES,
    PAPERS_DIR,
    PROJECT_ROOT,
)


def load_all_csv_records() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load and parse all empirical CSV benchmark records from results directory.

    Returns:
        Mapping of lowercased module name -> dataset name -> list of record dicts.
    """
    all_csvs = glob.glob(str(PROJECT_ROOT / "results/**/*.csv"), recursive=True)
    all_csvs += glob.glob(str(PROJECT_ROOT / "results/*.csv"))
    all_csvs = sorted(list(set(all_csvs)))

    # Map: lower_name -> dict(dataset -> list of rows)
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for p in all_csvs:
        lp = p.lower()
        rel_p = str(Path(p).relative_to(PROJECT_ROOT))
        ds = "other"
        if "sst2" in lp or "sst-2" in lp:
            ds = "sst2"
        elif "mnli" in lp:
            ds = "mnli"
        elif "agnews" in lp or "ag_news" in lp:
            ds = "agnews"
        elif "wikitext" in lp or "gpt2" in lp:
            ds = "wikitext2"

        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                fns = [c.strip().lower().replace(" ", "_") if c else "" for c in reader.fieldnames]
                if "module_name" not in fns and "module" not in fns:
                    continue
                for r in reader:
                    norm_r = {
                        k.strip().lower().replace(" ", "_"): v.strip() for k, v in r.items() if k
                    }
                    m = norm_r.get("module_name", "") or norm_r.get("module", "")
                    if not m or m in ("full", "baseline"):
                        continue
                    norm_r["_source_csv"] = rel_p
                    norm_r["_detected_ds"] = ds
                    data[m.lower()][ds].append(norm_r)
        except Exception:
            pass

    return data


def resolve_code_path(mod_name: str) -> Path:
    """Resolve file path on filesystem for a given module name.

    Args:
        mod_name: The module identifier name.

    Returns:
        Path pointing to the module implementation or Path('N/A').
    """
    special = {
        "AdaptiveSVD": "neuralops/attentions/adaptive_svd_attention.py",
        "OH": "neuralops/attentions/optimized_hybrid_attention.py",
        "SqueezingHeadsAttentionDistillationLayer": "neuralops/attentions/squeezing_heads_distiller.py",
        "AdaptiveSpanAttention": "neuralops/attentions/adaptive_span_attention.py",
        "AdaptiveSparseAttention": "neuralops/attentions/adaptive_sparse_attention.py",
        "HALO": "neuralops/attentions/halo_regla.py",
        "fusion": "neuralops/attentions/fusion_attn.py",
    }
    if mod_name in special:
        return PROJECT_ROOT / special[mod_name]

    candidates = [
        PROJECT_ROOT / f"neuralops/compression_engine/{mod_name}.py",
        PROJECT_ROOT / f"neuralops/attentions/{mod_name}.py",
        PROJECT_ROOT / f"neuralops/compression_engine/{mod_name.lower()}.py",
        PROJECT_ROOT / f"neuralops/attentions/{mod_name.lower()}.py",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Check inside paper folder for python file
    p_files = glob.glob(str(PAPERS_DIR / mod_name / "*.py"))
    p_files = [f for f in p_files if not Path(f).name.startswith(("build", "apply_vars", "test"))]
    if p_files:
        fname = Path(p_files[0]).name
        c1 = PROJECT_ROOT / f"neuralops/compression_engine/{fname}"
        c2 = PROJECT_ROOT / f"neuralops/attentions/{fname}"
        if c1.exists():
            return c1
        if c2.exists():
            return c2
        return Path(p_files[0])

    return Path("N/A")


__all__ = ["load_all_csv_records", "resolve_code_path"]
