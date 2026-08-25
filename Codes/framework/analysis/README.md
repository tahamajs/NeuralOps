# Analysis Subsystem (`Codes/framework/analysis`)

The `analysis` subsystem provides analytical tools for inspecting neural network weight distributions, layer-wise representation dynamics, singular spectrum properties, statistical significance testing, and automated technical report generation.

---

## 🏗 Architecture Overview

- **Layerwise Weight Analyzer (`layerwise_analyzer.py`)**: Spectral decomposition of weight matrices (, \Sigma, V^T$), condition number computation, Frobenius/spectral norm profiling, and effective rank quantification across layers.
- **Model Comparison Engine (`model_comparison.py`)**: Pairwise and multi-model benchmarking comparing baseline architectures against compressed and linear-attention variants.
- **Statistical Significance Testing (`comparison/statistical_tests.py`)**: Paired t-tests, Wilcoxon signed-rank tests, bootstrap confidence intervals, and Cohen's $ effect sizes for benchmark validation.
- **Automated Report Generation (`report_generator.py`)**: Compiles quantitative summaries, benchmark tables, and LaTeX/Markdown technical reports.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `layerwise_analyzer.py` | `LayerwiseAnalyzer` for singular spectrum and rank analysis across transformer layers. |
| `model_comparison.py` | Multi-model metric aggregation, latency/throughput comparisons, and compression ratio profiling. |
| `comparison/statistical_tests.py` | Hypothesis testing tools ensuring statistically validated benchmark claims. |
| `report_generator.py` | Automated generation of structured Markdown/LaTeX research reports. |
| `metrics.py` | Specialized analytical metrics (compression efficiency, perplexity delta, retention ratio). |

---

## 🚀 Quick Start & Usage

### 1. Analyzing Layerwise Singular Spectrum
```python
from transformers import AutoModel
from framework.analysis.layerwise_analyzer import LayerwiseAnalyzer

model = AutoModel.from_pretrained("distilbert-base-uncased")
analyzer = LayerwiseAnalyzer(model)

# Compute singular value decay and effective rank per layer
layer_spectra = analyzer.analyze_spectral_properties()
print(f"Layer 0 Effective Rank: {layer_spectra[0]['effective_rank']}")
```

### 2. Statistical Comparison of Model Runs
```python
from framework.analysis.comparison.statistical_tests import paired_t_test

baseline_accs = [0.845, 0.850, 0.848, 0.852, 0.847]
compressed_accs = [0.843, 0.848, 0.846, 0.850, 0.845]

stat_result = paired_t_test(baseline_accs, compressed_accs)
print(f"p-value: {stat_result['p_value']:.4f}, Significant: {stat_result['significant']}")
```
