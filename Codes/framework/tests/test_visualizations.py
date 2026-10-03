"""
Unit tests for visualization utilities and interactive HTML dashboard generation.
"""

import os
import shutil
import tempfile
import unittest

from framework.visualization.benchmark_visualizer import (
    plot_attention_heatmap, plot_dataset_radar_chart, plot_kv_cache_scaling,
    plot_pareto_frontier)
from framework.visualization.dashboard_generator import (
    generate_dashboard_html, generate_summary_json)


class TestVisualizations(unittest.TestCase):
    """Test suite verifying visualization plotting and dashboard generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_models = [
            {"name": "Dense Baseline", "latency_ms": 3.5, "ppl": 11.2, "peak_vram_mb": 512.0},
            {"name": "SVD Linear Attention", "latency_ms": 2.1, "ppl": 11.8, "peak_vram_mb": 310.0},
            {"name": "MambaSVD-Flash", "latency_ms": 1.4, "ppl": 11.4, "peak_vram_mb": 240.0},
        ]
        self.mock_benchmark_data = {
            "title": "Unit Test Dashboard",
            "models": self.mock_models,
            "kpis": {"max_speedup": "2.5x", "memory_reduction": "53%"},
        }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_pareto_frontier_plot(self):
        ppls = [m["ppl"] for m in self.mock_models]
        latencies = [m["latency_ms"] for m in self.mock_models]
        labels = [m["name"] for m in self.mock_models]
        fig = plot_pareto_frontier(ppls, latencies, labels, save_path=os.path.join(self.temp_dir, "pareto.png"))
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "pareto.png")))

    def test_kv_cache_scaling_plot(self):
        seq_lens = [512, 1024, 2048]
        mem_dict = {
            "Dense Baseline": [64, 128, 256],
            "DeltaKV": [16, 32, 64],
        }
        fig = plot_kv_cache_scaling(seq_lens, mem_dict, save_path=os.path.join(self.temp_dir, "kv.png"))
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "kv.png")))

    def test_dashboard_generator(self):
        out_html = os.path.join(self.temp_dir, "dashboard.html")
        html = generate_dashboard_html(self.mock_benchmark_data, output_path=out_html)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("SVD Linear Attention", html)
        self.assertTrue(os.path.exists(out_html))


if __name__ == "__main__":
    unittest.main()
