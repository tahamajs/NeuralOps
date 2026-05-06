from .base import BaseCompressor, COMPRESSOR_REGISTRY
from .magnitude_pruning import MagnitudePruning
from .movement_pruning import MovementPruning
from .distillation import DistillationCompressor
from .low_rank import LowRankCompressor

def list_compressors():
    return list(COMPRESSOR_REGISTRY.keys())