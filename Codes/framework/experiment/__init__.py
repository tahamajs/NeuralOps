# framework/experiment/__init__.py
import logging
from .main import run_training, run_evaluation, run_compression

__all__ = ["run_training", "run_evaluation", "run_compression"]
