"""High-level query API for querying, filtering, and comparing benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)


class ResultsIndex:
    """Query engine for exploring, comparing, and analyzing benchmark runs."""

    def __init__(self, output_dir: Path | str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.jsonl_file = self.output_dir / "results.jsonl"
        self._cached_results: Optional[List[BenchmarkResult]] = None

    def _load_results(self, force_reload: bool = False) -> List[BenchmarkResult]:
        """Lazy loader for all individual JSON benchmark results."""
        if self._cached_results is not None and not force_reload:
            return self._cached_results

        results: List[BenchmarkResult] = []
        if self.output_dir.exists():
            for path in self.output_dir.rglob("*.json"):
                if path.name in ("run_index.json", "run_state.json", "package.json", "index.json"):
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if "module" in data and ("status" in data or "run_metadata" in data):
                        results.append(BenchmarkResult.from_dict(data))
                except Exception:
                    continue

        self._cached_results = results
        return self._cached_results

    def find(
        self,
        module: Optional[str] = None,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        rank: Optional[int] = None,
        training_type: Optional[str] = None,
        status: Optional[str] = None,
        accuracy_min: Optional[float] = None,
        perplexity_max: Optional[float] = None,
    ) -> List[BenchmarkResult]:
        """Query benchmark results with multi-criteria filtering."""
        records = self._load_results()
        filtered: List[BenchmarkResult] = []
        for r in records:
            mod_name = r.module.get("name") if isinstance(r.module, dict) else r.module
            if module and mod_name != module:
                continue
            model_name = r.model.get("name") if isinstance(r.model, dict) else r.model
            if model and model_name != model:
                continue
            ds_name = r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset
            if dataset and ds_name != dataset:
                continue
            if rank is not None and r.config.get("rank") != rank:
                continue
            if training_type is not None:
                r_tt = getattr(r, "training_type", None) or r.config.get("training_type")
                if r_tt != training_type:
                    continue
            if status and r.status != status:
                continue
            if accuracy_min is not None:
                acc = float((r.metrics or {}).get("accuracy", 0.0) or 0.0)
                if acc < accuracy_min:
                    continue
            if perplexity_max is not None:
                ppl = float((r.metrics or {}).get("perplexity", float("inf")) or float("inf"))
                if ppl > perplexity_max:
                    continue
            filtered.append(r)
        return filtered

    def best(
        self,
        metric: str = "accuracy",
        group_by: Optional[Union[str, Sequence[str]]] = None,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> Union[Optional[BenchmarkResult], Dict[Any, BenchmarkResult]]:
        """Find top-performing benchmark result(s).

        If `group_by` is provided (e.g. 'module' or ['model', 'dataset']), returns
        a dict mapping group keys to their best BenchmarkResult.
        If `group_by` is None, returns the single top BenchmarkResult across candidates.
        """
        candidates = [r for r in self._load_results() if r.status == "success"]
        if model:
            candidates = [r for r in candidates if (r.model.get("name") if isinstance(r.model, dict) else r.model) == model]
        if dataset:
            candidates = [r for r in candidates if (r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset) == dataset]

        if not candidates:
            return {} if group_by is not None else None

        if group_by is None:
            return self._best_from_list(candidates, metric)

        groups: Dict[Any, List[BenchmarkResult]] = {}
        for r in candidates:
            if isinstance(group_by, (list, tuple)):
                key = tuple(self._get_attr_value(r, g) for g in group_by)
            else:
                key = self._get_attr_value(r, group_by)
            groups.setdefault(key, []).append(r)

        best_dict: Dict[Any, BenchmarkResult] = {}
        for k, v in groups.items():
            b = self._best_from_list(v, metric)
            if b is not None:
                best_dict[k] = b
        return best_dict

    def _get_attr_value(self, r: BenchmarkResult, attr: str) -> str:
        if attr == "module":
            return str(r.module.get("name") if isinstance(r.module, dict) else r.module)
        if attr == "model":
            return str(r.model.get("name") if isinstance(r.model, dict) else r.model)
        if attr == "dataset":
            return str(r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset)
        if attr == "training_type":
            return str(getattr(r, "training_type", None) or r.config.get("training_type", "none"))
        if attr == "rank":
            return str(r.config.get("rank", "default"))
        return str(r.config.get(attr, "unknown"))

    def _best_from_list(self, results: List[BenchmarkResult], metric: str) -> Optional[BenchmarkResult]:
        if not results:
            return None
        is_lower_better = metric.lower() in ("perplexity", "loss", "latency_ms", "duration_s", "peak_vram_mb")
        def score_fn(res: BenchmarkResult) -> float:
            val = (res.metrics or {}).get(metric)
            if val is None and res.resources:
                val = res.resources.get(metric)
            if val is None:
                return float("inf") if is_lower_better else float("-inf")
            try:
                return float(val)
            except (ValueError, TypeError):
                return float("inf") if is_lower_better else float("-inf")

        if is_lower_better:
            return min(results, key=score_fn)
        return max(results, key=score_fn)

    def compare(
        self,
        modules: List[str],
        metric: str = "accuracy",
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """Compare specific modules on identical model and dataset configuration."""
        matched: List[BenchmarkResult] = []
        for mod in modules:
            runs = self.find(module=mod, model=model, dataset=dataset, status="success")
            if runs:
                best_run = self._best_from_list(runs, metric)
                if best_run:
                    matched.append(best_run)
            else:
                # Include any run if success not found
                any_runs = self.find(module=mod, model=model, dataset=dataset)
                if any_runs:
                    matched.append(any_runs[-1])
        return matched

    def coverage(self) -> pd.DataFrame:
        """Return coverage DataFrame across all modules, models, datasets."""
        all_modules = sorted(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys())))
        runs = self._load_results()
        data = []
        for mod in all_modules:
            mod_runs = [r for r in runs if (r.module.get("name") if isinstance(r.module, dict) else r.module) == mod]
            successful = [r for r in mod_runs if r.status == "success"]
            models_covered = sorted(set(r.model.get("name") if isinstance(r.model, dict) else r.model for r in mod_runs))
            datasets_covered = sorted(set(r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset for r in mod_runs))
            mod_type = "attention" if mod in ATTENTION_REGISTRY else "compressor"
            data.append({
                "module": mod,
                "type": mod_type,
                "total_runs": len(mod_runs),
                "successful_runs": len(successful),
                "models": ", ".join(models_covered),
                "datasets": ", ".join(datasets_covered),
                "is_covered": len(successful) > 0,
            })
        return pd.DataFrame(data)

    def missing_combinations(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Identify module x model x dataset combinations that have no successful run."""
        models = models or ["distilbert-base-uncased", "gpt2"]
        datasets = datasets or ["sst2", "wikitext2"]
        all_modules = sorted(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys())))
        successful_runs = [r for r in self._load_results() if r.status == "success"]

        existing = set()
        for r in successful_runs:
            m = r.module.get("name") if isinstance(r.module, dict) else r.module
            md = r.model.get("name") if isinstance(r.model, dict) else r.model
            ds = r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset
            existing.add((m, md, ds))

        missing: List[Dict[str, str]] = []
        for mod in all_modules:
            for md in models:
                for ds in datasets:
                    if (mod, md, ds) not in existing:
                        missing.append({"module": mod, "model": md, "dataset": ds})
        return missing

    def failures(self) -> List[BenchmarkResult]:
        """Return all failed runs."""
        return [r for r in self._load_results() if r.status != "success"]

    def to_dataframe(self) -> pd.DataFrame:
        """Export indexed benchmark runs to Pandas DataFrame."""
        rows = []
        for r in self._load_results():
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "module": r.module.get("name") if isinstance(r.module, dict) else r.module,
                "module_type": r.module.get("type") if isinstance(r.module, dict) else "",
                "family": r.module.get("family") if isinstance(r.module, dict) else "",
                "model": r.model.get("name") if isinstance(r.model, dict) else r.model,
                "dataset": r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset,
                "training_type": getattr(r, "training_type", None) or r.config.get("training_type", "none"),
                "rank": r.config.get("rank"),
                "status": r.status,
                "duration_s": (r.resources or {}).get("duration_s") or (r.run_metadata or {}).get("wall_clock_duration_s"),
                "peak_vram_mb": (r.resources or {}).get("peak_vram_mb"),
            }
            if r.metrics:
                for k, v in r.metrics.items():
                    row[f"metric_{k}"] = v
            if getattr(r, "baseline_metrics", None):
                for k, v in r.baseline_metrics.items():
                    row[f"baseline_{k}"] = v
            if getattr(r, "delta_metrics", None):
                for k, v in r.delta_metrics.items():
                    row[f"delta_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Export summary table in GitHub-flavored Markdown."""
        df = self.to_dataframe()
        if df.empty:
            return "No benchmark results found."
        cols = [c for c in ["module", "model", "dataset", "training_type", "rank", "metric_accuracy", "metric_perplexity", "status"] if c in df.columns]
        return df[cols].to_markdown(index=False)

    def to_latex(self) -> str:
        """Export publication-ready LaTeX table."""
        df = self.to_dataframe()
        if df.empty:
            return "% No benchmark results found."
        cols = [c for c in ["module", "model", "dataset", "training_type", "rank", "metric_accuracy", "metric_perplexity", "status"] if c in df.columns]
        return df[cols].to_latex(index=False, escape=True)

    def build_global_index(self) -> Dict[str, Any]:
        """Build and persist results/index.json summary index."""
        runs = self._load_results(force_reload=True)
        all_modules = sorted(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys())))
        modules_covered = sorted(set(r.module.get("name") if isinstance(r.module, dict) else r.module for r in runs if r.status == "success"))
        models_covered = sorted(set(r.model.get("name") if isinstance(r.model, dict) else r.model for r in runs))
        datasets_covered = sorted(set(r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset for r in runs))

        index_data: Dict[str, Any] = {
            "total_runs": len(runs),
            "total_modules_available": len(all_modules),
            "modules_covered": len(modules_covered),
            "coverage_pct": round((len(modules_covered) / max(1, len(all_modules))) * 100.0, 2),
            "models_covered": models_covered,
            "datasets_covered": datasets_covered,
            "successful_runs": len([r for r in runs if r.status == "success"]),
            "failed_runs": len([r for r in runs if r.status != "success"]),
            "modules_list": modules_covered,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.output_dir / "index.json"
        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        return index_data


__all__ = ["ResultsIndex"]
