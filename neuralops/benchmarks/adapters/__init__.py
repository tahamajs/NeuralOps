"""Task adapters for neuralops benchmarking suite."""

from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.adapters.ruler import RulerAdapter
from neuralops.benchmarks.adapters.zeroshot import ZeroShotAdapter

__all__ = [
    "ClassificationAdapter",
    "PerplexityAdapter",
    "ZeroShotAdapter",
    "RulerAdapter",
]
