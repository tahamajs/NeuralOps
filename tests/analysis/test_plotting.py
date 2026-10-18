"""Tests for neuralops.analysis.plotting module."""

from pathlib import Path
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from neuralops.analysis.plotting import (
    create_comprehensive_plots,
    plot_tradeoff,
    set_style,
)


class TestPlotting:
    """Test plotting generators and figures output."""

    def test_set_style(self):
        set_style()

    def test_create_comprehensive_plots_smoke(self, tmp_path):
        df = pd.DataFrame({
            "model": ["gpt2", "bert", "gpt2", "bert"],
            "task": ["sst2", "sst2", "mrpc", "mrpc"],
            "rank": [16, 16, 32, 32],
            "accuracy": [0.85, 0.88, 0.82, 0.84],
            "speedup_factor": [1.5, 1.4, 2.0, 1.9],
            "param_reduction": [0.3, 0.25, 0.5, 0.45],
        })
        create_comprehensive_plots(df, output_dir=str(tmp_path))
        target_file = tmp_path / "comprehensive_analysis.png"
        assert target_file.exists()

    def test_plot_tradeoff_smoke(self, tmp_path):
        df = pd.DataFrame({
            "model": ["gpt2", "bert"],
            "speedup_factor": [1.5, 2.0],
            "accuracy": [0.85, 0.80],
        })
        plot_tradeoff(df, output_dir=str(tmp_path))
        target_file = tmp_path / "tradeoff.png"
        assert target_file.exists()
