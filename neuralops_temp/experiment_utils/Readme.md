# Experiment Management (Django Integration)

This package provides the Django‑specific bridge to run experiments using the
Efficient Transformer Toolkit. It includes:

- **Background training runner** (`tasks.py`) – launches experiments in a
  separate thread with real‑time WebSocket updates, progress tracking, and
  artifact saving.
- **MegaOrchestrator** (`orchestrator.py`) – a single entry point that executes
  the full framework pipeline (model loading, data, training, results saving)
  using the unified configuration system.
- **Experiment comparison** (`comparison.py`) – side‑by‑side comparison of
  multiple training runs with metric deltas, parallel coordinates, and config
  diff (used in the Django dashboard).
- **AI Assistant** – simple monitoring agent that can suggest early stopping
  or other interventions based on live metrics.

All ML logic is delegated to the core framework (`framework.config`,
`framework.models`, `framework.data`, `framework.training`, etc.).
This package is only responsible for orchestration and Django integration.