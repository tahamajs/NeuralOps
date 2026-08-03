"""
Complete Federated Learning System.

Supports:
  - Aggregation: FedAvg, FedProx, FedAdam, SCAFFOLD, FedNova, MOON
  - Client selection: Random, Power-of-Choice, Quality-based
  - Communication compression: Top-K, Random, Quantization (QSGD, STC, signSGD)
  - Privacy: DP-SGD, Secure Aggregation
  - Data partitioning: IID, Non-IID (label skew, quantity skew)
  - Seamless integration with Django runner and ResultsSaver

References:
  - McMahan et al. (2017): Communication-Efficient Learning of Deep Networks from Decentralized Data
  - Li et al. (2020): Federated Optimization in Heterogeneous Networks
  - Reddi et al. (2020): Adaptive Federated Optimization
  - Karimireddy et al. (2020): SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
  - Wang et al. (2020): Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization
  - Li et al. (2021): Model-Contrastive Federated Learning
  - Bonawitz et al. (2017): Practical Secure Aggregation for Privacy-Preserving Machine Learning
"""

from .client_selection import SELECTOR_REGISTRY, BaseSelector
from .compression import COMPRESSION_REGISTRY, BaseCompressor
from .config import (
    AggregationStrategy,
    ClientSelectionStrategy,
    CompressionMethod,
    FLConfig,
    PrivacyMethod,
)
from .data_partitioning import DataPartitioner, PartitionType
from .integration import run_federated_experiment
from .privacy import PRIVACY_REGISTRY, BasePrivacyMechanism
from .server import FLServer
from .strategies import AGGREGATOR_REGISTRY, BaseAggregator

__all__ = [
    "FLConfig",
    "AggregationStrategy",
    "ClientSelectionStrategy",
    "CompressionMethod",
    "PrivacyMethod",
    "AGGREGATOR_REGISTRY",
    "SELECTOR_REGISTRY",
    "COMPRESSION_REGISTRY",
    "PRIVACY_REGISTRY",
    "BaseAggregator",
    "BaseSelector",
    "BaseCompressor",
    "BasePrivacyMechanism",
    "DataPartitioner",
    "PartitionType",
    "FLServer",
    "run_federated_experiment",
]
