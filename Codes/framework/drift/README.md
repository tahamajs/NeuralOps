```
feat(drift): comprehensive data drift detection system

- Add registry-based drift methods: KS, PSI, Wasserstein, JS, Chi2, Cramér's V
- Add DriftDetector for univariate and multivariate drift detection
- Add PCA reconstruction error and domain classifier for multivariate drift
- Add concept drift (prediction drift) detection
- Add DriftReport and FeatureDriftReport data models with serialisation
- Add drift visualisation: summary bar chart and overlapping histograms
- Add auto-drift-check integration with MegaOrchestrator/ResultsSaver
- Add DriftConfig for flexible configuration
- Provide full README with usage examples and integration guide
```

---

# `framework/drift` — Data Drift Detection

The **drift** module provides a production‑grade, registry‑based system for detecting **data drift** between a reference dataset (e.g., training data) and a current dataset (e.g., production data).  
It supports **univariate** and **multivariate** drift, **concept drift**, automatic integration with your training/validation pipeline, and publication‑ready visualisations.

---

## Quick Start

```python
from framework.drift import DriftDetector, DriftConfig, auto_drift_check

# 1. Prepare data as dict of numpy arrays
reference = {
    "feature_a": np.random.randn(1000),
    "feature_b": np.random.exponential(1, 1000),
    "labels": np.random.choice([0, 1], 1000)
}
current = {
    "feature_a": np.random.randn(1000) + 0.3,   # slight shift
    "feature_b": np.random.exponential(1.2, 1000),
    "labels": np.random.choice([0, 1], 1000)
}

# 2. Detect drift
detector = DriftDetector(reference)
report = detector.detect(current)

print(f"Features drifted: {report.n_features_drifted}/{report.n_features_tested}")
print(f"Drift percentage: {report.drift_percentage}%")

# 3. Save and visualise
report.save("./drift_reports/drift_report.json")
from framework.drift import generate_drift_plots
generate_drift_plots(report, reference, current, "./drift_reports")
```

---

## Drift Methods (Univariate)

All methods are registered in `METHOD_REGISTRY` and can be extended.

| Method | Key | Description | Default Threshold |
|--------|-----|-------------|-------------------|
| Kolmogorov‑Smirnov | `"ks"` | Two‑sample KS test for continuous distributions | `0.1` |
| Population Stability Index | `"psi"` | PSI – measures distribution shift | `0.1` |
| Wasserstein (Earth Mover's) | `"wasserstein"` | Earth Mover’s Distance | `0.1` |
| Jensen‑Shannon Divergence | `"js"` | JS divergence between histograms | `0.1` |
| Chi‑Squared | `"chi2"` | Chi‑squared test (p‑value) | `0.05` (as p‑value) |
| Cramér's V | `"cramers_v"` | Association measure (categorical) | `0.1` |

You can select which methods to run via `DriftConfig(methods=[...])`.  
Per‑method thresholds can be set with `method_thresholds`.

---

## Multivariate Drift

Enabled via `DriftConfig(enable_multivariate=True)`.

- **PCA reconstruction error** (`multivariate_method="pca"`) – projects reference and current data into the same PCA space and compares reconstruction errors.
- **Domain classifier** (`multivariate_method="domain_classifier"`) – trains a logistic regression to distinguish reference from current; high AUC indicates drift.

---

## Concept Drift (Prediction Drift)

If you provide a `"labels"` (or any prediction‑related) column in your data, you can detect concept drift by enabling `enable_concept_drift=True` and specifying a `concept_metric`.

---

## Configuration (`DriftConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `methods` | `List[str]` | `["ks","psi","wasserstein"]` | Univariate methods to run |
| `threshold` | `float` | `0.1` | Global drift threshold |
| `method_thresholds` | `Dict[str,float]` | `{}` | Per‑method overrides |
| `enable_multivariate` | `bool` | `True` | Enable multivariate drift |
| `multivariate_method` | `str` | `"pca"` | `"pca"` or `"domain_classifier"` |
| `multivariate_n_components` | `int` | `5` | PCA components |
| `enable_concept_drift` | `bool` | `False` | Enable concept drift |
| `concept_metric` | `str` | `"psi"` | Method for concept drift |
| `features` | `Optional[List[str]]` | `None` | Explicit feature list; `None` = all numeric |
| `categorical_features` | `Optional[List[str]]` | `None` | Categorical columns |
| `max_categories` | `int` | `20` | Max unique values for auto‑detection |
| `generate_plots` | `bool` | `True` | Generate plots |
| `output_dir` | `str` | `"./drift_reports"` | Plot output directory |

---

## Output

- **`DriftReport`** dataclass containing per‑feature results, multivariate result, concept drift result, and summary statistics.
- **Serialisation** – `report.save(path)` writes a JSON report; `DriftReport.load(path)` loads it.
- **Plots** – summary bar chart of drift scores + overlapping histograms of drifted features.

---

## Integration with the Framework

### With ResultsSaver / MegaOrchestrator

The function `auto_drift_check()` can be called post‑training to automatically compare reference and current data and save the report into the run’s artifact directory.

```python
from framework.drift import auto_drift_check

report = auto_drift_check(
    run_id="my_run",
    reference_data=reference_data,
    current_data=current_data,
    config=DriftConfig(),
    output_dir=Path("./runs/my_run/drift")
)
```

### With Django API

The `DriftReport` and `DriftConfig` are serialisable, so they can be passed as JSON via an API endpoint and displayed in a dashboard.

---

## Extending with a New Method

1. Define a function `(ref, cur, **kwargs) -> (statistic, p_value, drifted)`.
2. Decorate it with `@METHOD_REGISTRY.register("my_method")`.
3. Add `"my_method"` to the `methods` list in `DriftConfig`.

---

This module ensures that your trained models remain **reliable, monitored, and safe** in production — a critical component of any ML pipeline.