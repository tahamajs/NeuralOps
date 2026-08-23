# Framework documentation and UI guide

This directory contains the reusable PyTorch framework. The documentation is
organized by implementation area:

| Area | Source code | Documentation |
| --- | --- | --- |
| Attention modules | `attentions/*.py` | `attentions/*.md` |
| Compression engines | `compression_engine/*.py` | `compression_engine/*.md` |
| Training and distributed execution | `trainer/`, `distributed/` | `trainer/README.md`, `distributed/README.md` |
| Data and experiment utilities | `data/`, `experiment_utils/`, `experiment_logging/` | matching `README.md` files |
| Results and tracking | `results_saver/`, `tracking/` | matching `README.md` files |
| Analysis and visualization | `analysis/`, `visualization/` | matching `README.md` files |
| Documentation dashboard | `web/` | this file and `web/catalog.json` |

## What is guaranteed

The executable source and tests are authoritative. Individual method pages
describe the implementation currently present in the corresponding Python
module; they are not claims that every method is production-ready on every
hardware target. Optional CUDA, Triton, Optuna, tracking, and cloud backends
must be installed and enabled explicitly.

Run the framework verification suite from this directory:

```bash
pytest -q
python -m compileall -q attentions compression_engine core data distributed \
  evaluation hyperopt results_saver trainer visualization
```

The attention registry can be inspected without loading a model:

```python
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

print(len(ATTENTION_REGISTRY), sorted(ATTENTION_REGISTRY.keys()))
print(len(COMPRESSOR_REGISTRY), sorted(COMPRESSOR_REGISTRY.keys()))
```

## Documentation conventions

Each module page should state:

1. the registered key and Python implementation;
2. expected input and output shapes;
3. constructor options and important defaults;
4. mask, dtype, device, and training/evaluation behavior;
5. whether pretrained weight transfer is supported;
6. known limitations and the test that validates the behavior.

Pages that describe a prototype, approximation, or fallback must say so
explicitly. Results from synthetic data, simulated FP8, or CPU fallbacks must
not be presented as hardware measurements or accuracy guarantees.

## Documentation dashboard

`web/catalog.json` is the checked-in catalog used by the standalone dashboard.
It is generated from the Markdown files in `attentions/`,
`compression_engine/`, and the framework-level README files. The dashboard
loads local files first and can display telemetry when a compatible API is
available; without telemetry it must remain usable as an offline documentation
browser.

To serve the dashboard locally:

```bash
python -m http.server 8080 --directory web
```

Do not open `web/index.html` directly with `file://`; browsers intentionally
block its catalog and Markdown fetches in that mode.

## Status vocabulary

- **Implemented**: the documented path is present and covered by tests.
- **Experimental**: the path runs, but performance or fidelity depends on
  model architecture, calibration data, or optional kernels.
- **Fallback**: the framework uses a safe baseline when an optional dependency
  or accelerator is unavailable.
- **Placeholder**: the page documents an extension point that is not an
  implementation and must not be used for production measurements.

This distinction keeps research comparisons reproducible and prevents a
placeholder or simulated backend from being reported as a completed method.
