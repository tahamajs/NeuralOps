"""
Experiment Comparison Engine – side-by-side diff, parallel coordinates,
metric delta tables, and run diff for training runs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deepdiff import DeepDiff

from .models import TrainingRun


@dataclass
class ComparisonResult:
    """Result of comparing multiple training runs."""

    run_ids: List[str]
    metrics_table: List[Dict[str, Any]]  # one row per run
    config_diff: Dict[str, Any]  # DeepDiff between configs
    hyperparameter_table: List[Dict[str, Any]]  # hyperparams across runs
    metric_deltas: List[Dict[str, Any]]  # delta from baseline
    timeline_data: Dict[str, List[float]]  # for loss curve overlay
    parallel_coords: Dict[str, Any]  # data for parallel coords plot
    best_run_id: Optional[str] = None
    summary: str = ""


class ExperimentComparator:
    """
    Compare multiple TrainingRuns with rich diff output.
    Inspired by ZenML's comparison tool, MLflow compare, and RagaAI diff.
    """

    def __init__(self, run_ids: List[str], baseline_id: Optional[str] = None):
        self.run_ids = run_ids
        self.baseline_id = baseline_id or (run_ids[0] if run_ids else None)
        self.runs: List[TrainingRun] = []

    def load_runs(self):
        self.runs = list(
            TrainingRun.objects.filter(id__in=self.run_ids).order_by("created_at")
        )

    def compare(self) -> ComparisonResult:
        self.load_runs()
        if len(self.runs) < 2:
            raise ValueError("Need at least 2 runs to compare")

        baseline = next(
            (r for r in self.runs if str(r.id) == self.baseline_id), self.runs[0]
        )
        result = ComparisonResult(
            run_ids=[str(r.id) for r in self.runs],
            metrics_table=self._build_metrics_table(),
            config_diff=self._build_config_diff(),
            hyperparameter_table=self._build_hyperparameter_table(),
            metric_deltas=self._build_metric_deltas(baseline),
            timeline_data=self._build_timeline_overlay(),
            parallel_coords=self._build_parallel_coords(),
            best_run_id=self._find_best_run(),
            summary="",
        )
        result.summary = self._generate_summary(result)
        return result

    def _build_metrics_table(self) -> List[Dict[str, Any]]:
        rows = []
        for run in self.runs:
            metrics = run.final_metrics or {}
            config = run.config or {}
            rows.append(
                {
                    "run_id": str(run.id),
                    "name": run.name or str(run.id)[:8],
                    "status": run.status,
                    "created_at": run.created_at.isoformat(),
                    "epochs": config.get("epochs", "N/A"),
                    "batch_size": config.get("batch_size", "N/A"),
                    "learning_rate": config.get("learning_rate", "N/A"),
                    "model_name": config.get("model_name", "N/A"),
                    "eval_accuracy": metrics.get("eval_accuracy"),
                    "eval_loss": metrics.get("eval_loss"),
                    "eval_f1": metrics.get("eval_f1"),
                    "duration_seconds": run.actual_duration,
                }
            )
        return rows

    def _build_config_diff(self) -> Dict[str, Any]:
        configs = {str(r.id): r.config for r in self.runs}
        first_id = str(self.runs[0].id)
        diffs = {}
        for rid, cfg in configs.items():
            if rid != first_id:
                diff = DeepDiff(
                    configs[first_id], cfg, ignore_order=True, verbose_level=2
                )
                diffs[rid] = diff
        return {
            "baseline_id": first_id,
            "diffs": diffs,
            "all_config_keys": list(
                set().union(*[set(c.keys()) for c in configs.values()])
            ),
        }

    def _build_hyperparameter_table(self) -> List[Dict[str, Any]]:
        hp_keys = [
            "learning_rate",
            "batch_size",
            "epochs",
            "max_seq_length",
            "compression_rank",
            "lora_r",
            "lora_alpha",
            "weight_decay",
            "warmup_steps",
            "grad_accum",
            "max_grad_norm",
        ]
        rows = []
        for run in self.runs:
            config = run.config or {}
            row = {"run_id": str(run.id), "name": run.name or str(run.id)[:8]}
            for k in hp_keys:
                row[k] = config.get(k)
            rows.append(row)
        return rows

    def _build_metric_deltas(self, baseline: TrainingRun) -> List[Dict[str, Any]]:
        base_metrics = baseline.final_metrics or {}
        deltas = []
        metric_keys = [
            "eval_accuracy",
            "eval_loss",
            "eval_f1",
            "eval_precision",
            "eval_recall",
        ]
        for run in self.runs:
            if str(run.id) == str(baseline.id):
                continue
            run_metrics = run.final_metrics or {}
            delta_row = {"run_id": str(run.id), "name": run.name or str(run.id)[:8]}
            for k in metric_keys:
                base_v = base_metrics.get(k)
                run_v = run_metrics.get(k)
                if base_v is not None and run_v is not None:
                    delta_row[k] = round(run_v - base_v, 6)
                    delta_row[f"{k}_pct"] = (
                        round(((run_v - base_v) / abs(base_v)) * 100, 2)
                        if base_v != 0
                        else None
                    )
                else:
                    delta_row[k] = None
            deltas.append(delta_row)
        return deltas

    def _build_timeline_overlay(self) -> Dict[str, List[float]]:
        """Extract training history for overlaid loss curves."""
        timeline = {"steps": [], "series": {}}
        for run in self.runs:
            history = run.training_history or {}
            steps = history.get("steps", [])
            losses = history.get("losses", [])
            if steps and losses:
                if len(steps) > len(timeline["steps"]):
                    timeline["steps"] = steps
                timeline["series"][str(run.id)] = losses
        return timeline

    def _build_parallel_coords(self) -> Dict[str, Any]:
        """Build data for a parallel coordinates plot (hyperparams + metrics)."""
        dimensions = [
            "learning_rate",
            "batch_size",
            "epochs",
            "max_seq_length",
            "compression_rank",
            "eval_accuracy",
            "eval_loss",
            "eval_f1",
        ]
        data = []
        for run in self.runs:
            config = run.config or {}
            metrics = run.final_metrics or {}
            point = {"run_id": str(run.id), "name": run.name or str(run.id)[:8]}
            for d in dimensions:
                point[d] = config.get(d) or metrics.get(d)
            data.append(point)
        return {"dimensions": dimensions, "data": data}

    def _find_best_run(self) -> Optional[str]:
        best = None
        best_acc = -1
        for run in self.runs:
            acc = (run.final_metrics or {}).get("eval_accuracy", -1)
            if acc is not None and acc > best_acc:
                best_acc = acc
                best = str(run.id)
        return best

    def _generate_summary(self, result: ComparisonResult) -> str:
        lines = [f"Comparison of {len(self.runs)} runs."]
        if result.best_run_id:
            best_run = next(
                (r for r in self.runs if str(r.id) == result.best_run_id), None
            )
            if best_run:
                lines.append(f"Best run: {best_run.name or result.best_run_id[:8]}")
        # Count improvements
        improved = sum(
            1
            for d in result.metric_deltas
            if d.get("eval_accuracy") is not None and d["eval_accuracy"] > 0
        )
        lines.append(f"{improved} run(s) improved over baseline in accuracy.")
        return " ".join(lines)


def compare_runs_api(run_ids: List[str], baseline_id: Optional[str] = None) -> Dict:
    """API-friendly wrapper."""
    comparator = ExperimentComparator(run_ids, baseline_id)
    result = comparator.compare()
    return {
        "run_ids": result.run_ids,
        "metrics_table": result.metrics_table,
        "config_diff": result.config_diff,
        "hyperparameter_table": result.hyperparameter_table,
        "metric_deltas": result.metric_deltas,
        "timeline_data": result.timeline_data,
        "parallel_coords": result.parallel_coords,
        "best_run_id": result.best_run_id,
        "summary": result.summary,
    }
