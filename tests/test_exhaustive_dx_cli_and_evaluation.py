"""
Suite 10: Exhaustive Verification Battery for Developer Experience (DX), CLI, Recipes, and Plots.
Verifies:
  10.1 Builtin recipes (fast_linear_recovery, etc.) and recommend().
  10.2 Model inspector summary and compare.
  10.3 Publication plot generator (PDF/SVG artifact output).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.recipes import BUILTIN_RECIPES, recommend
from neuralops.analysis.inspector import summary, compare
from neuralops.analysis.paper_plots import PaperPlotGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_10_1_recipes_and_recommendation():
    assert len(BUILTIN_RECIPES) >= 3
    assert "fast_linear_recovery" in BUILTIN_RECIPES

    rec = recommend()
    assert "recommended_attention" in rec


def test_10_2_inspector_summary_and_compare():
    m1 = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8)).to(DEVICE)
    m2 = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8)).to(DEVICE)

    s = summary(m1)
    assert s["total_parameters"] > 0

    x = torch.randn(2, 16, device=DEVICE)
    comp = compare(m1, m1, x)
    assert comp["cosine_similarity"] > 0.999


def test_10_3_paper_plot_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = PaperPlotGenerator(output_dir=tmpdir)
        dummy_points = [
            {"name": "full_attn", "compression_ratio": 1.0, "accuracy": 1.0},
            {"name": "linear_attn", "compression_ratio": 2.5, "accuracy": 0.96},
        ]
        p_svg, p_pdf = gen.plot_pareto(dummy_points, filename="pareto_test")
        assert Path(p_svg).exists()
        assert Path(p_pdf).exists()
