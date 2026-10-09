"""Automated rolling summaries generator (by_module, by_model, by_dataset, leaderboards, failures, coverage, timeline)."""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)


class SummaryGenerator:
    """Regenerates rolling publication-ready Markdown summaries from benchmark logs and indexes."""

    def __init__(self, output_dir: Path | str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.summaries_dir = self.output_dir / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_file = self.output_dir / "results.jsonl"

    def _load_all_records(self) -> List[Dict[str, Any]]:
        records = []
        if self.jsonl_file.exists():
            try:
                with open(self.jsonl_file, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception as e:
                LOG.warning("Could not read results.jsonl for summary generation: %s", e)
        return records

    def generate_all(self) -> Dict[str, Path]:
        """Regenerate all 7 analytical Markdown summary documents."""
        records = self._load_all_records()
        return {
            "by_module": self.generate_by_module(records),
            "by_model": self.generate_by_model(records),
            "by_dataset": self.generate_by_dataset(records),
            "leaderboards": self.generate_leaderboards(records),
            "failures": self.generate_failures(records),
            "coverage": self.generate_coverage(records),
            "timeline": self.generate_timeline(records),
        }

    def generate_by_module(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "by_module.md"
        grouped = defaultdict(list)
        for r in records:
            if r.get("status") == "success" and r.get("accuracy") is not None:
                grouped[r.get("module", "unknown")].append(r["accuracy"])

        rows = []
        for mod, accs in grouped.items():
            rows.append({
                "module": mod,
                "runs": len(accs),
                "best_accuracy": max(accs),
                "worst_accuracy": min(accs),
                "avg_accuracy": sum(accs) / len(accs),
            })
        rows.sort(key=lambda x: x["best_accuracy"], reverse=True)

        lines = [
            "# Benchmark Summary: By Module\n",
            "| Module | Runs | Best Accuracy | Worst Accuracy | Avg Accuracy |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(f"| `{r['module']}` | {r['runs']} | {r['best_accuracy']:.4f} | {r['worst_accuracy']:.4f} | {r['avg_accuracy']:.4f} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_by_model(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "by_model.md"
        grouped = defaultdict(list)
        for r in records:
            if r.get("status") == "success" and r.get("accuracy") is not None:
                grouped[r.get("model", "unknown")].append((r.get("module"), r["accuracy"]))

        lines = [
            "# Benchmark Summary: By Model\n",
            "| Model | Total Runs | Unique Modules | Best Module | Best Accuracy |",
            "|---|---|---|---|---|",
        ]
        for mdl, items in grouped.items():
            mods = {m for m, _ in items}
            best_mod, best_acc = max(items, key=lambda x: x[1])
            lines.append(f"| `{mdl}` | {len(items)} | {len(mods)} | `{best_mod}` | {best_acc:.4f} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_by_dataset(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "by_dataset.md"
        grouped = defaultdict(list)
        for r in records:
            if r.get("status") == "success" and r.get("accuracy") is not None:
                grouped[r.get("dataset", "unknown")].append((r.get("module"), r["accuracy"]))

        lines = [
            "# Benchmark Summary: By Dataset\n",
            "| Dataset | Total Runs | Unique Modules | Best Module | Best Accuracy |",
            "|---|---|---|---|---|",
        ]
        for ds, items in grouped.items():
            mods = {m for m, _ in items}
            best_mod, best_acc = max(items, key=lambda x: x[1])
            lines.append(f"| `{ds}` | {len(items)} | {len(mods)} | `{best_mod}` | {best_acc:.4f} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_leaderboards(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "leaderboards.md"
        pairs = defaultdict(list)
        for r in records:
            if r.get("status") == "success" and r.get("accuracy") is not None:
                pairs[(r.get("model"), r.get("dataset"))].append(r)

        lines = ["# Benchmark Leaderboards\n"]
        for (mdl, ds), runs in pairs.items():
            lines.append(f"## Target: `{mdl}` on `{ds}`\n")
            lines.append("| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |")
            lines.append("|---|---|---|---|---|---|")
            sorted_runs = sorted(runs, key=lambda x: x.get("accuracy") or 0.0, reverse=True)[:20]
            for idx, r in enumerate(sorted_runs, 1):
                acc = r.get("accuracy") or 0.0
                dur = r.get("duration_s") or 0.0
                vram = r.get("peak_vram_mb") or 0.0
                lines.append(f"| {idx} | `{r.get('module')}` | {r.get('rank')} | {acc:.4f} | {dur}s | {vram:.1f}MB |")
            lines.append("")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_failures(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "failures.md"
        failures = [r for r in records if r.get("status") != "success"]
        lines = [
            "# Benchmark Failures & Error Log\n",
            f"Total failures observed: **{len(failures)}**\n",
            "| Run ID | Module | Model | Dataset | Status |",
            "|---|---|---|---|---|",
        ]
        for f in failures:
            lines.append(f"| `{f.get('run_id')[:8]}` | `{f.get('module')}` | `{f.get('model')}` | `{f.get('dataset')}` | **{f.get('status')}** |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_coverage(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "coverage.md"
        successful_modules = {r.get("module") for r in records if r.get("status") == "success"}
        all_registered = set(ATTENTION_REGISTRY.keys()) | set(COMPRESSOR_REGISTRY.keys())
        tested = len(successful_modules & all_registered)
        total = len(all_registered)
        pct = (tested / total * 100.0) if total > 0 else 0.0

        missing = sorted(list(all_registered - successful_modules))

        lines = [
            "# Benchmark Module Coverage Report\n",
            f"- **Registry Total**: {total} modules ({len(ATTENTION_REGISTRY)} Attention + {len(COMPRESSOR_REGISTRY)} Compressors)",
            f"- **Tested Modules**: {tested} ({pct:.1f}% coverage)",
            f"- **Remaining Unbenchmarked**: {len(missing)}\n",
            "### Missing Modules",
        ]
        for m in missing[:50]:
            lines.append(f"- `{m}`")
        if len(missing) > 50:
            lines.append(f"- ... and {len(missing) - 50} more")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def generate_timeline(self, records: List[Dict[str, Any]]) -> Path:
        p = self.summaries_dir / "timeline.md"
        sorted_records = sorted(records, key=lambda x: str(x.get("timestamp")), reverse=True)
        lines = [
            "# Benchmark Execution Timeline\n",
            "| Timestamp | Module | Model | Dataset | Status | Accuracy |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted_records:
            acc = r.get("accuracy")
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
            lines.append(f"| {r.get('timestamp')} | `{r.get('module')}` | `{r.get('model')}` | `{r.get('dataset')}` | {r.get('status')} | {acc_str} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p


__all__ = ["SummaryGenerator"]
