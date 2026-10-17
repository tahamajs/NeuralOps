"""NeuralOps Web Visualizer & Architecture Explorer Package."""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

__all__ = [
    "get_visualizer_static_dir",
    "load_modules_catalog",
    "load_spatial_data",
    "render_roofline_html",
]

def get_visualizer_static_dir() -> str:
    """Return the absolute path to visualizer static assets."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "docs", "visualizer")

def load_modules_catalog() -> List[Dict[str, Any]]:
    """Load 131+ attention mechanism catalog metadata."""
    path = os.path.join(get_visualizer_static_dir(), "modules-catalog.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("modules", data) if isinstance(data, dict) else data
    return []

def load_spatial_data() -> Dict[str, Any]:
    """Load 3D spatial coordinates and clustering graph."""
    path = os.path.join(get_visualizer_static_dir(), "spatial-data.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def render_roofline_html(
    module_name: str = "flash_attention_2",
    achieved_tflops: float = 185.4,
    arithmetic_intensity: float = 64.2,
    peak_tflops: float = 312.0,
    peak_bandwidth_gb_s: float = 900.0,
) -> str:
    """Generate interactive Plotly/HTML markup for hardware Roofline visualization."""
    ridge = peak_tflops / (peak_bandwidth_gb_s / 1000.0) if peak_bandwidth_gb_s > 0 else 1.0
    regime = 'Compute Bound' if arithmetic_intensity >= ridge else 'Memory Bound'
    return f'''<div class="roofline-chart-container" style="background: #0f172a; border-radius: 8px; padding: 16px; color: #f8fafc;"><h4 style="margin: 0 0 12px 0; color: #38bdf8;">Roofline Analysis: {module_name}</h4><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; font-size: 0.82rem; margin-bottom: 12px;"><div style="background: #1e293b; padding: 8px; border-radius: 6px;"><b>Peak Compute:</b> {peak_tflops:.1f} TFLOPS</div><div style="background: #1e293b; padding: 8px; border-radius: 6px;"><b>Peak HBM BW:</b> {peak_bandwidth_gb_s:.1f} GB/s</div><div style="background: #1e293b; padding: 8px; border-radius: 6px;"><b>Ridge Point:</b> {ridge:.1f} FLOP/B</div><div style="background: #1e293b; padding: 8px; border-radius: 6px;"><b>Achieved:</b> {achieved_tflops:.1f} TFLOPS</div><div style="background: #1e293b; padding: 8px; border-radius: 6px;"><b>Arithmetic Int:</b> {arithmetic_intensity:.1f} FLOP/B</div></div><div style="font-size: 0.75rem; color: #94a3b8;">Operating regime: <span style="color: #10b981; font-weight: 600;">{regime}</span></div></div>'''
