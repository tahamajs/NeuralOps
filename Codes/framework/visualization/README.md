# Visualization Subsystem (`Codes/framework/visualization`)

The `visualization` subsystem provides automated chart generation, interactive HTML dashboards, multi-head attention heatmap visualization, singular value spectrum analysis, and model interpretability suites.

---

## 🏗 Architecture Overview

- **Chart & Report Generation (`chart_generator.py`, `advanced_report.py`)**: Generates publication-ready training loss curves, evaluation accuracy trajectories, throughput benchmarks, and Pareto efficiency frontiers.
- **Attention Visualizer (`attention_visualizer.py`, `attention_heatmap.py`)**: Extracts and visualizes layer-by-layer attention matrices, cross-head sparsity maps, entropy profiles, and token interaction graphs.
- **SVD Spectral Analysis Studio (`svd_analysis.py`, `representation_studios.py`)**: Renders singular value decay curves, effective rank quantification, and condition number tracking across transformer layers.
- **Interactive Dashboard Generator (`dashboard_generator.py`)**: Generates standalone interactive HTML/JS dashboards embedding metrics, interactive plots, and layer comparisons.
- **Plotter Registry & Backends (`plotter.py`, `plotters/`)**: Extensible `BasePlotter` interface with a production `DefaultPlotter` (Matplotlib-based with headless rendering safety).

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `chart_generator.py` | Generates training loss/accuracy curves, Pareto frontiers, and speedup bar plots. |
| `dashboard_generator.py` | Builds standalone interactive HTML dashboards summarizing experiment runs. |
| `attention_visualizer.py` | Plots 2D attention matrices, head heatmaps, and token connection weights. |
| `svd_analysis.py` | Spectral analysis plots, singular value distributions, and rank compression curves. |
| `plotter.py` | `BasePlotter` abstraction, `DefaultPlotter`, `get_plotter()`, and `PLOTTER_REGISTRY`. |
| `interpretability_suite.py` | Representation drift, gradient saliency maps, and feature attribution visualizers. |
| `comparison_plots.py` | Side-by-side comparative plots evaluating baseline vs compressed variants. |

---

## 🚀 Quick Start & Usage

### 1. Generating Training Curves & Pareto Plots
```python
import numpy as np
from framework.visualization.chart_generator import ChartGenerator

chart_gen = ChartGenerator(output_dir="./reports/plots")
chart_gen.plot_training_curves(
    train_losses=[0.85, 0.52, 0.31, 0.22],
    eval_losses=[0.90, 0.58, 0.39, 0.33],
    metrics={"eval_accuracy": [0.72, 0.81, 0.86, 0.89]},
)
```

### 2. Visualizing Attention Heatmaps
```python
from framework.visualization.attention_visualizer import AttentionVisualizer

attn_vis = AttentionVisualizer(output_dir="./reports/heatmaps")
attn_matrix = np.random.rand(32, 32)
attn_vis.plot_attention_heatmap(
    attention_matrix=attn_matrix,
    tokens=["the", "quick", "brown", "fox"],
    layer_idx=0,
    head_idx=0,
    model_name="distilbert_svd",
)
```
