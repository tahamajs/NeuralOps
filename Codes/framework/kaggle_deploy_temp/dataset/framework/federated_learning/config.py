"""Configuration for Federated Learning."""

from dataclasses import dataclass
from enum import Enum


class AggregationStrategy(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    FEDADAGRAD = "fedadagrad"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    MOON = "moon"


class ClientSelectionStrategy(Enum):
    RANDOM = "random"
    POWER_OF_CHOICE = "power_of_choice"
    QUALITY_BASED = "quality_based"


class CompressionMethod(Enum):
    NONE = "none"
    TOPK = "topk"
    RANDOM_SPARSIFICATION = "random_sparsification"
    QSGD = "qsgd"
    STC = "stc"
    SIGN_SGD = "sign_sgd"


class PrivacyMethod(Enum):
    NONE = "none"
    DP_SGD = "dp_sgd"
    SECURE_AGGREGATION = "secure_aggregation"


@dataclass
class FLConfig:
    """Complete configuration for federated learning experiment."""

    # Core FL settings
    num_clients: int = 10
    clients_per_round: float = 0.3  # fraction or absolute number
    num_rounds: int = 100
    local_epochs: int = 5
    local_batch_size: int = 32

    # Strategy
    aggregation: AggregationStrategy = AggregationStrategy.FEDAVG
    server_learning_rate: float = 1.0
    server_momentum: float = 0.9

    # FedProx specific
    fedprox_mu: float = 0.01

    # MOON specific
    moon_mu: float = 1.0  # contrastive loss weight
    moon_temperature: float = 0.5

    # Client selection
    client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    power_of_choice_candidates: int = 5

    # Compression
    compression: CompressionMethod = CompressionMethod.NONE
    compression_topk_fraction: float = 0.1
    compression_quantization_bits: int = 8

    # Privacy
    privacy: PrivacyMethod = PrivacyMethod.NONE
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5

    # Data partitioning
    partition_type: str = (
        "iid"  # "iid" or "non_iid_label_skew" or "non_iid_quantity_skew"
    )
    non_iid_alpha: float = 0.5  # Dirichlet concentration parameter
    num_classes: int = 2

    # Integration
    use_differential_privacy: bool = False
    report_every_rounds: int = 10
    output_dir: str = "./fl_runs"
    seed: int = 42
