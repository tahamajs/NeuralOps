"""
Django integration layer – background experiment runner,
WebSocket broadcaster, AI assistant, and multi‑run comparison.
Compatibility shim re-exporting neuralops.experiment_utils.
"""

from __future__ import annotations

import neuralops.experiment_utils as _neu
from neuralops.experiment_utils import (
    AIAssistant,
    ExperimentComparator,
    MegaOrchestrator,
    broadcast_terminal,
    compare_runs_api,
    comparison,
    config,
    data,
    main,
    model,
    orchestrator,
    run_compression,
    run_evaluation,
    run_training,
    run_training_async,
    runner,
    tasks,
    train,
)

__all__ = [
    "run_training_async",
    "run_training",
    "run_evaluation",
    "run_compression",
    "AIAssistant",
    "broadcast_terminal",
    "ExperimentComparator",
    "compare_runs_api",
    "MegaOrchestrator",
    "comparison",
    "config",
    "data",
    "main",
    "model",
    "orchestrator",
    "runner",
    "tasks",
    "train",
]
