"""
ComparisonEngine – compares multiple runs, methods, or models.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .leaderboard import generate_leaderboard
from .plotting import plot_critical_difference_diagram, plot_pairwise_comparison_heatmap
from .statistical_tests import (
    anova_test,
    dunn_test,
    mann_whitney_test,
    nemenyi_test,
    pairwise_t_tests,
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    metric: str
    groups: Dict[str, np.ndarray]
    group_stats: pd.DataFrame
    anova: Optional[Dict[str, float]] = None
    pairwise_t_tests: Optional[pd.DataFrame] = None
    mann_whitney: Optional[pd.DataFrame] = None
    nemenyi: Optional[pd.DataFrame] = None
    dunn: Optional[pd.DataFrame] = None
    leaderboard: Optional[pd.DataFrame] = None
    cd_diagram_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    is_significant: bool = False
    best_group: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class ComparisonEngine:
    def __init__(self) -> None:
        self._groups: Dict[str, np.ndarray] = {}

    def load_from_dirs(
        self,
        dirs: List[Union[str, Path]],
        group_names: Optional[List[str]] = None,
        metric: str = "eval_accuracy",
    ) -> None:
        """Load from dirs from disk or config."""
        for i, d in enumerate(dirs):
            name = group_names[i] if group_names else Path(d).name
            metrics_path = Path(d) / "artifacts" / "final_metrics.json"
            if not metrics_path.exists():
                logger.warning("No final_metrics.json in %s", d)
                continue
            with open(metrics_path) as f:
                data = json.load(f)
            score = data.get(metric, data.get("accuracy", data.get("loss")))
            if score is None:
                continue
            self._groups.setdefault(name, []).append(float(score))
        for k in self._groups:
            self._groups[k] = np.array(self._groups[k])

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        group_col: str = "model",
        metric_col: str = "eval_accuracy",
    ) -> None:
        """Load from dataframe from disk or config."""
        for group, sub in df.groupby(group_col):
            self._groups[group] = sub[metric_col].dropna().to_numpy()

    def load_from_dict(self, data: Dict[str, List[float]]) -> None:
        """Load from dict from disk or config."""
        for k, v in data.items():
            self._groups[k] = np.array(v)

    def add_group(self, name: str, scores: Union[List[float], np.ndarray]) -> None:
        """Add group."""
        self._groups[name] = np.array(scores)

    def compare(
        self,
        metric: str = "eval_accuracy",
        significance_level: float = 0.05,
        generate_plots: bool = True,
        output_dir: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare."""
        if len(self._groups) < 2:
            raise ValueError("Need at least 2 groups")

        stats_rows = []
        for name, scores in self._groups.items():
            n = len(scores)
            mean = np.mean(scores)
            std = np.std(scores, ddof=1) if n > 1 else 0.0
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            stats_rows.append(
                {
                    "group": name,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci95": ci,
                }
            )
        group_stats = pd.DataFrame(stats_rows).sort_values("mean", ascending=False)

        anova_result = anova_test(self._groups)
        is_significant = (
            anova_result is not None
            and anova_result.get("p_value", 1) < significance_level
        )

        ttest_df = pairwise_t_tests(self._groups, alpha=significance_level)
        mw_df = mann_whitney_test(self._groups, alpha=significance_level)
        dunn_df = dunn_test(self._groups, alpha=significance_level)
        nemenyi_df = nemenyi_test(self._groups, alpha=significance_level)

        lb = generate_leaderboard(group_stats)
        best_group = group_stats.iloc[0]["group"]

        result = ComparisonResult(
            metric=metric,
            groups=self._groups,
            group_stats=group_stats,
            anova=anova_result,
            pairwise_t_tests=ttest_df,
            mann_whitney=mw_df,
            nemenyi=nemenyi_df,
            dunn=dunn_df,
            leaderboard=lb,
            is_significant=is_significant,
            best_group=best_group,
        )

        if generate_plots and output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            try:
                result.cd_diagram_path = plot_critical_difference_diagram(
                    self._groups,
                    alpha=significance_level,
                    output_path=str(out / "critical_difference.png"),
                )
            except Exception as e:
                result.notes.append(f"CD diagram failed: {e}")
            try:
                result.heatmap_path = plot_pairwise_comparison_heatmap(
                    self._groups,
                    metric=metric,
                    output_path=str(out / "pairwise_heatmap.png"),
                )
            except Exception as e:
                result.notes.append(f"Heatmap failed: {e}")

        return result

    def clear(self) -> None:
        """Clear."""
        self._groups.clear()
