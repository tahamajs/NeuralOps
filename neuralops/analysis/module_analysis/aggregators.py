"""Aggregators and categorizers for module analysis metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from neuralops.analysis.module_analysis.constants import ALIASES


def get_all_module_rows(
    mod_name: str, csv_data: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """Gather all CSV rows associated with a module name and its aliases.

    Args:
        mod_name: Module identifier name.
        csv_data: Parsed CSV records dict.

    Returns:
        List of distinct benchmark rows matching the module.
    """
    lookup_keys = [mod_name.lower()]
    if mod_name in ALIASES:
        lookup_keys.extend([a.lower() for a in ALIASES[mod_name]])

    # Strip common suffixes
    for suffix in ["_compressor", "_attention", "_pruner", "_quantizer"]:
        if mod_name.lower().endswith(suffix):
            lookup_keys.append(mod_name.lower()[: -len(suffix)])

    all_rows: List[Dict[str, Any]] = []
    seen: Set[Any] = set()
    for k in lookup_keys:
        for ds, rows in csv_data.get(k, {}).items():
            for r in rows:
                key = (
                    r.get("_source_csv"),
                    r.get("timestamp"),
                    r.get("accuracy_pct"),
                    r.get("compression_rate"),
                )
                if key not in seen:
                    seen.add(key)
                    all_rows.append(r)
    return all_rows


def get_best_dataset_result(rows: List[Dict[str, Any]], target_ds: str) -> str:
    """Extract best metric value string for a target dataset from benchmark rows.

    Args:
        rows: Module benchmark rows.
        target_ds: Target dataset name ('sst2', 'mnli', 'agnews', 'wikitext2').

    Returns:
        Formatted metric string or 'N/A'.
    """
    ds_rows = [r for r in rows if r.get("_detected_ds") == target_ds]
    if not ds_rows:
        return "N/A"

    valid_vals = []
    for r in ds_rows:
        if target_ds == "wikitext2":
            ppl_str = r.get("perplexity") or r.get("delta_ppl")
            if ppl_str and ppl_str not in ("N/A", "nan", "None", ""):
                try:
                    val = float(ppl_str)
                    if val > 0:
                        valid_vals.append(val)
                except Exception:
                    pass
        else:
            acc_str = r.get("accuracy_pct")
            if acc_str and acc_str not in ("N/A", "nan", "None", ""):
                try:
                    val = float(acc_str)
                    valid_vals.append(val)
                except Exception:
                    pass

    if not valid_vals:
        return "N/A"

    if target_ds == "wikitext2":
        return f"{min(valid_vals):.2f}"
    else:
        return f"{max(valid_vals):.2f}%"


def determine_sub_category(mod_name: str, mod_type: str, cat: str) -> str:
    """Determine architectural subcategory based on module characteristics."""
    lname = mod_name.lower()
    if mod_type == "Utility" or mod_name == "pruning_utils":
        return "Utility"

    if mod_type == "Compressor":
        if cat == "VALID":
            return "Compression-Working"
        else:
            return "Compression-Failed"

    if any(
        k in lname for k in ["mamba", "jamba", "samba", "griffin", "hgrn2", "hymba", "simple_ssm"]
    ):
        return "SSM"
    if any(
        k in lname
        for k in [
            "cosformer",
            "linformer",
            "performer",
            "nystrom",
            "linear_attention",
            "galerkin",
            "decay_linear",
        ]
    ):
        return "Linear-Attention"
    if any(k in lname for k in ["mla", "nsa", "halo", "moba", "sparse"]):
        return "Sparse-Attention"
    if any(k in lname for k in ["gqa", "mqa", "wgqa"]):
        return "GQA-MQA"
    if any(k in lname for k in ["svd", "compressed_mha", "soa"]):
        return "SVD-Attention"

    return "Unknown"


def determine_fix_priority(cat: str, mod_type: str, sub_cat: str) -> str:
    """Determine issue fix priority (P0 to P3, or NONE)."""
    if cat == "BROKEN":
        return "P0"
    if mod_type == "Compressor" and cat in ("FALLBACK", "QUESTIONABLE"):
        return "P1"
    if mod_type == "Attention Replacement" and cat in ("FALLBACK", "QUESTIONABLE"):
        return "P2"
    if cat == "NO_DATA":
        return "P3"
    if cat == "VALID":
        return "NONE"
    return "P2"


__all__ = [
    "get_all_module_rows",
    "get_best_dataset_result",
    "determine_sub_category",
    "determine_fix_priority",
]
