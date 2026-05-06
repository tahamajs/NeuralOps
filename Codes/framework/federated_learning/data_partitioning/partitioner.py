"""Data partitioning for federated learning – IID and Non‑IID (label skew, quantity skew)."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

class PartitionType(Enum):
    IID = "iid"
    NON_IID_LABEL_SKEW = "non_iid_label_skew"
    NON_IID_QUANTITY_SKEW = "non_iid_quantity_skew"

class DataPartitioner:
    """
    Partition a dataset across clients according to the chosen strategy.

    Usage:
        partitioner = DataPartitioner(num_clients=10, partition_type="non_iid_label_skew", alpha=0.5)
        client_data = partitioner.partition(dataset)
    """

    def __init__(
        self,
        num_clients: int = 10,
        partition_type: str = "iid",
        alpha: float = 0.5,
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.partition_type = PartitionType(partition_type)
        self.alpha = alpha
        self.num_classes = num_classes
        self.seed = seed
        np.random.seed(seed)

    def partition(self, dataset: Any) -> List[Any]:
        """Return a list of client datasets."""
        if self.partition_type == PartitionType.IID:
            return self._iid_partition(dataset)
        elif self.partition_type == PartitionType.NON_IID_LABEL_SKEW:
            return self._non_iid_label_skew(dataset)
        elif self.partition_type == PartitionType.NON_IID_QUANTITY_SKEW:
            return self._non_iid_quantity_skew(dataset)
        return self._iid_partition(dataset)

    def _iid_partition(self, dataset):
        """Equally split dataset randomly."""
        n = len(dataset)
        indices = np.random.permutation(n)
        splits = np.array_split(indices, self.num_clients)
        return [self._subset(dataset, s) for s in splits]

    def _non_iid_label_skew(self, dataset):
        """
        Dirichlet distribution over labels – each client gets a different
        proportion of labels (Hsu et al. 2019).
        """
        labels = self._get_labels(dataset)
        # Dirichlet distribution: shape (num_clients, num_classes)
        proportions = np.random.dirichlet([self.alpha] * self.num_classes, self.num_clients)
        # For each class, distribute samples to clients according to proportions
        class_indices = [np.where(labels == c)[0] for c in range(self.num_classes)]
        client_indices = [[] for _ in range(self.num_clients)]
        for c in range(self.num_classes):
            c_idx = class_indices[c]
            np.random.shuffle(c_idx)
            # Split class c across clients
            client_counts = np.maximum(1, np.round(proportions[:, c] * len(c_idx)).astype(int))
            # Adjust to ensure total matches
            diff = len(c_idx) - np.sum(client_counts)
            client_counts[0] += diff
            start = 0
            for i in range(self.num_clients):
                cnt = client_counts[i]
                client_indices[i].extend(c_idx[start:start+cnt])
                start += cnt
        return [self._subset(dataset, np.array(idx)) for idx in client_indices]

    def _non_iid_quantity_skew(self, dataset):
        """
        Different clients get different amounts of data (power‑law like).
        """
        n = len(dataset)
        # Generate sizes using a Dirichlet with skewed alpha
        alphas = np.random.uniform(0.5, 2.0, self.num_clients)
        proportions = np.random.dirichlet(alphas)
        sizes = np.maximum(1, np.round(proportions * n).astype(int))
        # Adjust to ensure total equals n
        diff = n - np.sum(sizes)
        sizes[0] += diff
        indices = np.random.permutation(n)
        client_data = []
        start = 0
        for sz in sizes:
            client_data.append(self._subset(dataset, indices[start:start+sz]))
            start += sz
        return client_data

    def _get_labels(self, dataset):
        """Extract labels from dataset. Override for custom datasets."""
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            return np.array(dataset.labels)
        elif isinstance(dataset, dict) and 'y' in dataset:
            return dataset['y'].numpy()
        # Fallback: assume the dataset yields (x, y) pairs
        labels = []
        for _, y in dataset:
            labels.append(y if isinstance(y, int) else y.item())
        return np.array(labels)

    def _subset(self, dataset, indices):
        """Return a subset of the dataset."""
        try:
            from torch.utils.data import Subset
            return Subset(dataset, indices)
        except ImportError:
            return [dataset[i] for i in indices]