# 🎨 Visualization Package

A comprehensive suite for generating publication‑quality plots, interactive HTML reports, and attention visualizations for your compression experiments.

## 📁 Structure

| File | Description |
|------|-------------|
| `base_visualizer.py` | Abstract base class and `VISUALIZER_REGISTRY` |
| `attention_visualizer.py` | `SVDAttentionVisualizer` with 10+ plot types |
| `advanced_report.py` | `AdvancedReportGenerator` – interactive Plotly HTML reports |
| `comparison_plots.py` | Standalone functions: accuracy vs rank, radar charts |
| `report_generator.py` | Simple HTML, CSV, LaTeX report generation |
| `utils.py` | Plotting helpers (style, save, colors) |

## 🚀 Quick Start

```python
from framework.visualization import SVDAttentionVisualizer

viz = SVDAttentionVisualizer(output_dir="plots")
viz.plot_attention_heatmap(attn_matrix, tokens=["[CLS]","I","love","NLP"])
viz.generate_training_plots(training_logs, "bert", "mrpc", rank=64)
```

### Interactive Reports

```python
from framework.visualization import AdvancedReportGenerator

gen = AdvancedReportGenerator("reports")
gen.generate_full_report(results_dict, df, profiler_report, layer_analysis)
```

## Available Plots in `SVDAttentionVisualizer`

- `plot_attention_heatmap` – heatmap of attention weights
- `plot_attention_entropy` – per‑token attention entropy
- `plot_attention_difference` – full vs. compressed difference map
- `analyze_singular_values` – SVD spectrum + cumulative energy
- `plot_subspace_alignment` – principal angles between full and compressed subspaces
- `plot_layer_analysis_grid` – multi‑layer, multi‑head attention grid
- `plot_rank_comparison` – accuracy vs. rank curve
- `plot_model_comparison` – comparison of multiple models
- `generate_training_plots` – loss and evaluation curves

All plots are saved to the specified `output_dir` as high‑resolution PNGs.

---

*From attention maps to interactive dashboards – the full visualization toolbox.*