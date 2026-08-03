from .base import COMPRESSOR_REGISTRY, BaseCompressor
from .distillation import DistillationCompressor
from .low_rank import LowRankCompressor
from .magnitude_pruning import MagnitudePruning
from .movement_pruning import MovementPruning


def list_compressors() -> Any:
    """List compressors."""
    return list(COMPRESSOR_REGISTRY.keys())
