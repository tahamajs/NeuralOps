# Drift Subsystem (`Codes/framework/drift`)

The `drift` subsystem provides real-time monitoring and detection of representation drift, concept shift, singular spectrum divergence, and distribution degradation in compressed foundation models.

---

## 🏗 Architecture Overview

- **Drift Detectors (`detectors.py`)**: Statistical and geometric drift detection algorithms:
  - **Wasserstein Distance Detector**: Measures Earth Mover's Distance between baseline and inference activation manifolds.
  - **Kolmogorov-Smirnov (KS) Test**: Evaluates empirical cumulative distribution shifts per hidden dimension.
  - **Spectral Divergence Monitor**: Tracks singular value spectrum degradation ($||\Sigma_{	ext{base}} - \Sigma_{	ext{curr}}||_F$) across transformer layers.
  - **Maximum Mean Discrepancy (MMD)**: Kernel-based multi-variate distribution shift detection.
- **Reporting & Visualizations (`report.py`, `visualization.py`)**: Generates automated drift summary dashboards, alert logs, and distribution shift charts.
- **Integration Engine (`integration.py`)**: Hooks into trainer evaluation loops and inference servers for continuous monitoring.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `detectors.py` | Core drift detection classes (Wasserstein, KS-Test, Spectral Drift). |
| `config.py` | `DriftConfig` defining significance thresholds ($lpha$), sliding window sizes, and metrics. |
| `report.py` | Generates structured drift diagnostic reports and alert events. |
| `visualization.py` | Visualizes activation distribution shifts and singular value drift. |
| `integration.py` | Connects drift monitors into active training and inference pipelines. |

---

## 🚀 Quick Start & Usage

### 1. Monitoring Representation Drift
```python
import torch
from framework.drift.detectors import SpectralDriftDetector

detector = SpectralDriftDetector(threshold=0.05)

# Fit baseline singular spectrum
baseline_activations = torch.randn(100, 768)
detector.fit_baseline(baseline_activations)

# Check drift on incoming batch
test_activations = torch.randn(100, 768) + 0.1
drift_result = detector.detect(test_activations)
print(f"Drift Detected: {drift_result.is_drift}, Divergence Score: {drift_result.score:.4f}")
```

### 2. Generating Drift Summary Report
```python
from framework.drift.report import generate_drift_report

report = generate_drift_report(detector, output_path="./reports/drift_summary.json")
```
