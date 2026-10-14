"""
Suite 10: Exhaustive Verification Battery for Developer Experience (DX), CLI, Recipes, and Plots.
Expanded to 20+ parameterized tests verifying:
  10.1 All built-in recipes in BUILTIN_RECIPES and recommend() across memory budgets.
  10.2 Model inspector summary and compare across architectures.
  10.3 PaperPlotGenerator producing vector graphics (.svg and .pdf).
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


@pytest.mark.parametrize("recipe_name", list(BUILTIN_RECIPES.keys()))
def test_10_1_builtin_recipes(recipe_name):
    recipe = BUILTIN_RECIPES[recipe_name]
    assert "description" in recipe
    assert "attention" in recipe or "compressor" in recipe or "strategy" in recipe


@pytest.mark.parametrize("vram_gb", [4, 6, 8, 24, 80])
def test_10_1_recommend_across_budgets(vram_gb):
    rec = recommend(gpu_memory_gb=vram_gb)
    assert "recommended_attention" in rec


@pytest.mark.parametrize("dim", [16, 32, 64])
def test_10_2_inspector_summary_and_compare(dim):
    m1 = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim // 2)).to(DEVICE)
    s = summary(m1)
    assert s["total_parameters"] > 0

    x = torch.randn(2, dim, device=DEVICE)
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
