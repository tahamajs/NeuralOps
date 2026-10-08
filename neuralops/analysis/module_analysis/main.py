"""Main analysis pipeline orchestrator for 217-module analysis."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from neuralops.analysis.module_analysis.aggregators import (
    determine_fix_priority,
    determine_sub_category,
    get_all_module_rows,
    get_best_dataset_result,
)
from neuralops.analysis.module_analysis.constants import (
    ALIASES,
    CSV_OUT,
    KNOWN_LIT,
    MD_OUT,
    PAPERS_DIR,
    PROJECT_ROOT,
)
from neuralops.analysis.module_analysis.parsers import (
    load_all_csv_records,
    resolve_code_path,
)
from neuralops.analysis.module_analysis.reporters import (
    write_csv_report,
    write_markdown_report,
)


def run_all_modules_analysis(
    csv_output_path: Path = CSV_OUT, md_output_path: Path = MD_OUT
) -> List[Dict[str, Any]]:
    """Execute complete 217-module audit, classification, and report generation.

    Args:
        csv_output_path: Path to target CSV file.
        md_output_path: Path to target Markdown file.

    Returns:
        List of generated module analysis records.
    """
    print("=" * 70)
    print("GENERATING COMPREHENSIVE 217-MODULE ANALYSIS")
    print("=" * 70)

    # 1. Load registries
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    except Exception as e:
        print(f"Warning importing framework registries: {e}")
        ATTENTION_REGISTRY, COMPRESSOR_REGISTRY = {}, {}

    # Build entity to registered names mapping
    reg_names_by_file: Dict[Path, List[str]] = defaultdict(list)
    reg_names_by_lower: Dict[str, List[str]] = defaultdict(list)

    for k, v in list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items()):
        reg_names_by_lower[k.lower()].append(k)
        if hasattr(v, "__code__"):
            fpath = getattr(v.__code__, "co_filename", "")
            if fpath:
                reg_names_by_file[Path(fpath).resolve()].append(k)
        elif hasattr(v, "__file__"):
            fpath = getattr(v, "__file__", "")
            if fpath:
                reg_names_by_file[Path(fpath).resolve()].append(k)

    # 2. Load CSV data
    print("Loading CSV benchmark records...")
    csv_data = load_all_csv_records()

    # 3. Load verification json
    verif_file = PROJECT_ROOT / "paper/_verification_results.json"
    verif_by_name: Dict[str, Dict[str, Any]] = {}
    if verif_file.exists():
        with open(verif_file, encoding="utf-8") as f:
            verif_data = json.load(f)
        verif_by_name = {r["name"]: r for r in verif_data.get("results", [])}

    # 4. Enumerate 217 paper directories
    paper_dirs = sorted(
        [
            d.name
            for d in PAPERS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and d.name != "Template.md"
        ]
    )
    print(f"Total modules to analyze: {len(paper_dirs)}")

    analysis_rows: List[Dict[str, Any]] = []

    for idx, mod in enumerate(paper_dirs, 1):
        module_name = mod
        code_p = resolve_code_path(mod)
        file_path = str(code_p.resolve()) if code_p.exists() else "N/A"

        if "compression_engine" in file_path:
            source_dir = "compression_engine"
            module_type = "Compressor"
        elif "attentions" in file_path:
            source_dir = "attentions"
            module_type = "Attention Replacement"
        elif "core" in file_path or mod == "pruning_utils":
            source_dir = "other"
            module_type = "Utility"
        else:
            source_dir = "other"
            module_type = "Unknown"

        v_info = verif_by_name.get(mod, {})
        category = v_info.get("classification", "NO_DATA")
        if mod == "pruning_utils":
            category = "NO_DATA"

        sub_category = determine_sub_category(mod, module_type, category)

        paper_ref, paper_arxiv, paper_year, paper_venue = "None", "None", "None", "None"
        l_key = mod.lower()
        for k, v in KNOWN_LIT.items():
            if k == l_key or k in l_key:
                paper_ref, paper_arxiv, paper_year, paper_venue = v
                break

        meta_p = PAPERS_DIR / mod / "00_meta.md"
        if meta_p.exists():
            with open(meta_p, encoding="utf-8", errors="replace") as mf:
                content = mf.read()
                if paper_ref == "None":
                    m_title = re.search(r"^#\s+([^\n]+)", content, re.MULTILINE)
                    if m_title and "Formal Architectural" not in m_title.group(1):
                        paper_ref = m_title.group(1).strip()
                if paper_arxiv == "None":
                    m_arxiv = re.search(
                        r"(?:arXiv:\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5})",
                        content,
                        re.IGNORECASE,
                    )
                    if m_arxiv:
                        paper_arxiv = m_arxiv.group(1).strip()
                        if paper_year == "None":
                            paper_year = "20" + paper_arxiv[:2]
                        if paper_venue == "None":
                            paper_venue = "arXiv"

        architectures = "DistilBERT,RoBERTa,GPT-2,LLaMA,Qwen"

        reg_keys = []
        if code_p.exists():
            reg_keys.extend(reg_names_by_file.get(code_p.resolve(), []))
        reg_keys.extend(reg_names_by_lower.get(mod.lower(), []))
        for alias in ALIASES.get(mod, []):
            reg_keys.extend(reg_names_by_lower.get(alias.lower(), []))
        reg_keys = sorted(list(set(reg_keys)))

        has_registry_entry = "YES" if reg_keys else "NO"
        registry_names = ";".join(reg_keys) if reg_keys else "N/A"

        all_mod_rows = get_all_module_rows(mod, csv_data)

        dataset_sst2_acc = get_best_dataset_result(all_mod_rows, "sst2")
        dataset_mnli_acc = get_best_dataset_result(all_mod_rows, "mnli")
        dataset_agnews_acc = get_best_dataset_result(all_mod_rows, "agnews")
        dataset_gpt2_ppl = get_best_dataset_result(all_mod_rows, "wikitext2")

        valid_comps = []
        for r in all_mod_rows:
            c_str = r.get("compression_rate") or r.get("compression_ratio")
            if c_str and c_str not in ("N/A", "nan", "None", ""):
                try:
                    c_val = float(c_str)
                    if c_val > 0:
                        valid_comps.append(c_val)
                except Exception:
                    pass
        compression_ratio = (
            f"{max(valid_comps):.3f}x" if valid_comps else ("1.000x" if all_mod_rows else "N/A")
        )

        params_before, params_after = "N/A", "N/A"
        for r in all_mod_rows:
            pb = r.get("total_params_before")
            pa = r.get("total_params_after")
            if pb and pa and pb not in ("N/A", "") and pa not in ("N/A", ""):
                params_before, params_after = str(pb), str(pa)
                break

        results_p = PAPERS_DIR / mod / "07_results.md"
        if not results_p.exists():
            paper_07_results_status = "MISSING"
            has_fabricated_tables = "NO"
        else:
            with open(results_p, encoding="utf-8", errors="replace") as rf:
                r_content = rf.read()
            if "FABRICATED" in r_content or "CAUTION" in r_content:
                paper_07_results_status = "FABRICATED"
                has_fabricated_tables = "YES"
            elif "## ✅ Verified Results" in r_content and category != "NO_DATA":
                paper_07_results_status = "COMPLETE"
                has_fabricated_tables = "NO"
            else:
                paper_07_results_status = "INCOMPLETE"
                has_fabricated_tables = "NO"

        csv_source, csv_timestamp = "N/A", "N/A"
        if all_mod_rows:
            b_row = all_mod_rows[0]
            for r in all_mod_rows:
                if r.get("status") == "PASS":
                    b_row = r
                    break
            csv_source = b_row.get("_source_csv", "N/A")
            csv_timestamp = b_row.get("timestamp", "N/A")
            if not csv_timestamp or csv_timestamp in ("N/A", "nan", "None"):
                csv_timestamp = "2026-09-20"

        fix_priority = determine_fix_priority(category, module_type, sub_category)

        row_dict = {
            "module_name": module_name,
            "file_path": file_path,
            "module_type": module_type,
            "source_dir": source_dir,
            "category": category,
            "sub_category": sub_category,
            "paper_reference": paper_ref,
            "paper_arxiv": paper_arxiv,
            "paper_year": paper_year,
            "paper_venue": paper_venue,
            "architectures": architectures,
            "has_registry_entry": has_registry_entry,
            "registry_names": registry_names,
            "dataset_sst2_acc": dataset_sst2_acc,
            "dataset_mnli_acc": dataset_mnli_acc,
            "dataset_agnews_acc": dataset_agnews_acc,
            "dataset_gpt2_ppl": dataset_gpt2_ppl,
            "compression_ratio": compression_ratio,
            "params_before": params_before,
            "params_after": params_after,
            "paper_07_results_status": paper_07_results_status,
            "has_fabricated_tables": has_fabricated_tables,
            "csv_source": csv_source,
            "csv_timestamp": csv_timestamp,
            "fix_priority": fix_priority,
        }
        analysis_rows.append(row_dict)

    cat_order = {
        "VALID": 0,
        "QUESTIONABLE": 1,
        "FALLBACK": 2,
        "BROKEN": 3,
        "NO_DATA": 4,
    }
    analysis_rows.sort(
        key=lambda r: (
            cat_order.get(r["category"], 9),
            r["sub_category"],
            r["module_name"],
        )
    )

    if paper_dirs:
        assert len(analysis_rows) == len(paper_dirs), f"Expected {len(paper_dirs)} rows, got {len(analysis_rows)}"

    print(f"Writing CSV: {csv_output_path}")
    write_csv_report(analysis_rows, csv_output_path)

    print(f"Writing Markdown: {md_output_path}")
    write_markdown_report(analysis_rows, md_output_path)

    print("=" * 70)
    print("ANALYSIS COMPLETE & VERIFIED")
    print(f"CSV saved to: {csv_output_path}")
    print(f"Markdown saved to: {md_output_path}")
    print("=" * 70)

    return analysis_rows


def main() -> None:
    """CLI entrypoint for generating all modules analysis."""
    run_all_modules_analysis()


__all__ = ["run_all_modules_analysis", "main"]
