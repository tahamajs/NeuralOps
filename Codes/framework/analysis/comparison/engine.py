"""
ComparisonEngine – the central hub for comparing multiple runs, methods, or models.

Usage:
    engine = ComparisonEngine()
    # Load results from a list of directories
    engine.load_from_dirs(["./runs/run1", "./runs/run2"])
    # Or from a DataFrame
    engine.load_from_dataframe(df)

    # Run statistical tests
    result = engine.compare(metric="eval_accuracy", significance_level=0.05)
    print(result.leaderboard)
    print(result.post_hoc)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from .statistical_tests import (
    anova_test, pairwise_t_tests, mann_whitney_test,
    nemenyi_test, dunn_test,
)
from .leaderboard import generate_leaderboard
from .plotting import (
    plot_critical_difference_diagram,
    plot_pairwise_comparison_heatmap,
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Complete result of a multi‑run comparison."""
    metric: str
    groups: Dict[str, np.ndarray]               # group name → array of scores
    group_stats: pd.DataFrame                    # mean, std, ci, n per group
    anova: Optional[Dict[str, float]] = None     # F‑stat, p‑value
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
    """
    Loads results from multiple sources and compares groups (methods, models, etc.)

    Each group is a named collection of scalar scores (e.g., accuracy from 5 seeds).
    """

    def __init__(self):
        self._groups: Dict[str, np.ndarray] = {}

    # ── Data loading ────────────────────────────────────────────────
    def load_from_dirs(
        self,
        dirs: List[Union[str, Path]],
        group_names: Optional[List[str]] = None,
        metric: str = "eval_accuracy",
    ) -> None:
        """
        Load results from run directories (each containing artifacts/final_metrics.json).
        Each directory is one sample; multiple directories per group are aggregated.
        """
        for i, d in enumerate(dirs):
            name = group_names[i] if group_names else Path(d).name
            metrics_path = Path(d) / "artifacts" / "final_metrics.json"
            if not metrics_path.exists():
                logger.warning("No final_metrics.json in %s", d)
                continue
            with open(metrics_path) as f:
                data = json.load(f)
            score = data.get(metric, data.get("accuracy", data.get("loss", None)))
            if score is None:
                continue
            self._groups.setdefault(name, []).append(float(score))
        # Convert to numpy arrays
        for k in self._groups:
            self._groups[k] = np.array(self._groups[k])

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        group_col: str = "model",
        metric_col: str = "eval_accuracy",
    ) -> None:
        """Load from a DataFrame where each row is one run."""
        for group, sub in df.groupby(group_col):
            self._groups[group] = sub[metric_col].dropna().to_numpy()

    def load_from_dict(self, data: Dict[str, List[float]]) -> None:
        """Directly load group → list of scores."""
        for k, v in data.items():
            self._groups[k] = np.array(v)

    def add_group(self, name: str, scores: Union[List[float], np.ndarray]) -> None:
        self._groups[name] = np.array(scores)

    # ── Comparison ───────────────────────────────────────────────────
    def compare(
        self,
        metric: str = "eval_accuracy",
        significance_level: float = 0.05,
        generate_plots: bool = True,
        output_dir: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Run full statistical comparison across all loaded groups.

        Returns a ComparisonResult with all statistics and optional plots.
        """
        if len(self._groups) < 2:
            raise ValueError("Need at least 2 groups to compare")

        # Group statistics
        stats_rows = []
        for name, scores in self._groups.items():
            n = len(scores)
            mean = np.mean(scores)
            std = np.std(scores, ddof=1) if n > 1 else 0.0
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            stats_rows.append({
                "group": name, "n": n, "mean": mean,
                "std": std, "ci95": ci,
            })
        group_stats = pd.DataFrame(stats_rows).sort_values("mean", ascending=False)

        # ANOVA
        anova_result = anova_test(self._groups)
        is_significant = anova_result is not None and anova_result.get("p_value", 1) < significance_level

        # Pairwise t‑tests (with Bonferroni correction)
        ttest_df = pairwise_t_tests(self._groups, alpha=significance_level)

        # Non‑parametric tests
        mw_df = mann_whitney_test(self._groups, alpha=significance_level)
        dunn_df = dunn_test(self._groups, alpha=significance_level)
        nemenyi_df = nemenyi_test(self._groups, alpha=significance_level)

        # Leaderboard
        lb = generate_leaderboard(group_stats)

        # Best group (highest mean)
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

        # Plots
        if generate_plots and output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            try:
                cd_path = plot_critical_difference_diagram(
                    self._groups, alpha=significance_level,
                    output_path=str(out / "critical_difference.png"),
                )
                result.cd_diagram_path = cd_path
            except Exception as e:
                result.notes.append(f"CD diagram failed: {e}")

            try:
                hm_path = plot_pairwise_comparison_heatmap(
                    self._groups, metric=metric,
                    output_path=str(out / "pairwise_heatmap.png"),
                )
                result.heatmap_path = hm_path
            except Exception as e:
                result.notes.append(f"Heatmap failed: {e}")

        return result

    def clear(self):
        self._groups.clear()