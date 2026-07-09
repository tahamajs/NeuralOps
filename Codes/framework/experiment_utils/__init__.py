"""
Django integration layer – background experiment runner,
WebSocket broadcaster, AI assistant, and multi‑run comparison.
"""

from .comparison import ExperimentComparator, compare_runs_api
from .orchestrator import MegaOrchestrator
from .tasks import AIAssistant, broadcast_terminal, run_training_async
from .main import run_training, run_evaluation, run_compression

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
