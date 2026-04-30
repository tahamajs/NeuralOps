# studio/__init__.py
from .config import config, DEVICE
from .compression import SVDLinearLayer, AdvancedZeroShotCompressor
from .attentions import SVDLinearAttention, CompressedAttentionLayer
from .routing import KeywordRouter, SmartRouter
from .ensemble import WeightedEnsemble
from .training import fine_tune_compressed_model
from .analysis import evaluate_model, benchmark_model
from .soa import SOABert, migrate_bert_to_soa, compute_global_basis_from_hidden_states
from .data import discover_datasets, load_dataset_from_config
from .utils import setup_logger, get_device, set_seed

__version__ = "2.0.0"
__all__ = [
    "config", "DEVICE",
    "SVDLinearLayer", "AdvancedZeroShotCompressor",
    "SVDLinearAttention", "CompressedAttentionLayer",
    "KeywordRouter", "SmartRouter",
    "WeightedEnsemble",
    "fine_tune_compressed_model",
    "evaluate_model", "benchmark_model",
    "SOABert", "migrate_bert_to_soa", "compute_global_basis_from_hidden_states",
    "discover_datasets", "load_dataset_from_config",
    "setup_logger", "get_device", "set_seed",
]