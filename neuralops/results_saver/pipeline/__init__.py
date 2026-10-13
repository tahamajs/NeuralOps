"""Pipeline package — result writing, run management, and sweep orchestration."""

from .result_writer import ResultWriter

__all__ = ["ResultWriter"]

# Auto-wired package members
from neuralops.results_saver.pipeline import result_writer
