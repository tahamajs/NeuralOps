"""
NeuralOps Core ML Framework

A unified, modular, and highly efficient toolkit for custom attention mechanisms,
distributed training, hyperparameter optimization, and federated learning.
"""

__version__ = "0.1.0"

# Expose core functionality at the top level for easy imports
try:
    from .models import BaseModel
    from .training import BaseTrainer
except ImportError:
    pass

# We explicitly do not import all submodules here to avoid circular dependencies
# and unnecessary heavy imports (like PyTorch) unless specifically requested.

__all__ = [
    "BaseModel",
    "BaseTrainer",
]
