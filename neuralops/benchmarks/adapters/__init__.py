"""Task adapters for neuralops benchmarking suite."""

from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter

__all__ = [
    "ClassificationAdapter",
    "PerplexityAdapter",
]
