"""Extended Task Loaders: GLUE (CoLA, STS-B) & Long Range Arena (ListOps, Sequential CIFAR-10).

Designed for memory-efficient batch streaming, tokenization, and synthetic caching
for NeurIPS/ICLR 2026 subquadratic attention benchmark evaluations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.data


class SequentialCIFAR10Dataset(torch.utils.data.Dataset):
    """LRA Sequential Image Classification (CIFAR-10 flattened to 1024 grayscale tokens)."""

    def __init__(self, num_samples: int = 1000, seq_len: int = 1024, num_classes: int = 10, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # 8-bit pixel values mapped to discrete tokens [0..255]
        self.inputs = torch.randint(0, 256, (num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": self.labels[idx],
        }


class ListOpsDataset(torch.utils.data.Dataset):
    """LRA ListOps: Hierarchical Prefix-Syntax Operator Evaluation (10-way classification, 2048 tokens)."""

    def __init__(self, num_samples: int = 1000, seq_len: int = 2048, vocab_size: int = 32, num_classes: int = 10, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        self.inputs = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": self.labels[idx],
        }


def get_extended_task_dataset(
    task_name: str,
    tokenizer_name: str = "bert-base-uncased",
    seq_len: int = 128,
    num_samples: int = 500,
    seed: int = 42,
) -> torch.utils.data.Dataset:
    """Returns memory-efficient Dataset instance for GLUE (CoLA, STS-B) or LRA (ListOps, CIFAR-10)."""
    task = task_name.lower().replace("-", "")

    if task == "listops":
        return ListOpsDataset(num_samples=num_samples, seq_len=min(seq_len, 2048), seed=seed)
    elif task in {"cifar10", "image"}:
        return SequentialCIFAR10Dataset(num_samples=num_samples, seq_len=min(seq_len, 1024), seed=seed)
    else:
        # GLUE fallback dataset generator
        from framework.eval_all_models_datasets import _generate_synthetic_batch
        num_labels = 1 if task == "stsb" else 2
        batch = _generate_synthetic_batch(
            dataset_name=task,
            batch_size=num_samples,
            seq_len=seq_len,
            vocab_size=30522,
            num_labels=num_labels,
        )
        
        class TensorDictDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict):
                self.data = data_dict
                self.n = len(next(iter(data_dict.values())))

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                return {k: v[i] for k, v in self.data.items()}

        return TensorDictDataset(batch)
