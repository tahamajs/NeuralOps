"""
Django integration layer – background experiment runner,
WebSocket broadcaster, AI assistant, and multi‑run comparison.
"""

from .comparison import ExperimentComparator, compare_runs_api
from .main import run_compression, run_evaluation, run_training
from .orchestrator import MegaOrchestrator
from .tasks import AIAssistant, broadcast_terminal, run_training_async

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
]
