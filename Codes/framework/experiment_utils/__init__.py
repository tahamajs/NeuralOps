"""
Django integration layer – background experiment runner,
WebSocket broadcaster, AI assistant, and multi‑run comparison.
"""
from .tasks import run_training_async, AIAssistant, broadcast_terminal
from .comparison import ExperimentComparator, compare_runs_api
from .orchestrator import MegaOrchestrator

__all__ = [
    "run_training_async",
    "AIAssistant",
    "broadcast_terminal",
    "ExperimentComparator",
    "compare_runs_api",
    "MegaOrchestrator",
]