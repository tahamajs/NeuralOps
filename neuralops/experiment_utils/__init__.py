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

# Auto-wired package members
from neuralops.experiment_utils import comparison
from neuralops.experiment_utils import config
from neuralops.experiment_utils import data
from neuralops.experiment_utils import main
from neuralops.experiment_utils import model
from neuralops.experiment_utils import orchestrator
from neuralops.experiment_utils import runner
from neuralops.experiment_utils import tasks
from neuralops.experiment_utils import train
