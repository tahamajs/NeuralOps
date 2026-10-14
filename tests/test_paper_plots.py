"""
Test Publication-Grade Paper Plot Generator.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.analysis.paper_plots import (
    plot_singular_value_spectrum,
    plot_attention_entropy_heatmap,
    plot_pareto_frontier,
    generate_all_paper_plots,
)


def test_paper_plots_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "plots"

        # Test individual functions
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        p_spec = out_dir / "test_spec.png"
        fig1 = plot_singular_value_spectrum(model, save_path=p_spec)
        assert p_spec.exists()

        attn = torch.softmax(torch.randn(16, 16), dim=-1)
        p_heat = out_dir / "test_heat.png"
        fig2 = plot_attention_entropy_heatmap(attn, save_path=p_heat)
        assert p_heat.exists()

        pts = [
            {"name": "Base", "compression_ratio": 1.0, "accuracy": 90.0},
            {"name": "Fast", "compression_ratio": 2.0, "accuracy": 89.0},
        ]
        p_pareto = out_dir / "test_pareto.png"
        fig3 = plot_pareto_frontier(pts, save_path=p_pareto)
        assert p_pareto.exists()

        # Test batch generator (.svg and .pdf)
        all_plots = generate_all_paper_plots(model=model, output_dir=out_dir / "batch")
        assert len(all_plots) == 6
        for p in all_plots:
            assert p.exists()
            assert p.stat().st_size > 0
