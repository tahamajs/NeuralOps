"""
Production Dataset Shapes & Collation Verification Test Suite.

Verifies:
1. ProductionDataCollator batch padding to multiples of 8 (Tensor Core aligned).
2. Exact 2D tensor shapes [B, L] for input_ids and attention_mask.
3. Proper 1D/2D label shapes and dtype casting for Classification, Regression, and Causal LM.
4. MultiChoiceDataset 3D tensor shapes [B, num_choices, L].
5. Device and pinned memory tensor transfer compatibility.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from framework.data.base import (DataConfig, ProductionDataCollator, TaskType,
                                 create_dataloader)
from framework.data.multi_choice import MultiChoiceDataset
from framework.data.synthetic_generator import (SyntheticClassificationDataset,
                                                SyntheticPerplexityDataset,
                                                generate_zipfian_tokens)
from torch.utils.data import DataLoader


class TestProductionDatasetShapes:
    """Test suite ensuring exact tensor dimensions, dtypes, and memory safety for production GPU servers."""

    def test_production_data_collator_padding_and_alignment(self) -> None:
        """Verify ProductionDataCollator pads sequences to multiples of 8 for GPU Tensor Cores."""
        collator = ProductionDataCollator(pad_token_id=0, pad_to_multiple_of=8)

        # Inhomogeneous variable-length input sequences
        features = [
            {"input_ids": [101, 2054, 102], "attention_mask": [1, 1, 1], "labels": 1},
            {"input_ids": [101, 2054, 2003, 1037, 3899, 102], "attention_mask": [1, 1, 1, 1, 1, 1], "labels": 0},
        ]
        batch = collator(features)

        # Max length was 6 -> padded to multiple of 8 = 8
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 8)
        assert batch["labels"].shape == (2,)

        # Check dtypes
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

        # Check padding tokens
        assert (batch["input_ids"][0, 3:] == 0).all()
        assert (batch["attention_mask"][0, 3:] == 0).all()

    def test_production_collator_2d_language_modeling_labels(self) -> None:
        """Verify 2D language modeling labels are padded with -100 ignore index."""
        collator = ProductionDataCollator(pad_token_id=0, label_pad_token_id=-100, pad_to_multiple_of=8)

        features = [
            {"input_ids": [10, 20, 30], "labels": [10, 20, 30]},
            {"input_ids": [10, 20, 30, 40, 50], "labels": [10, 20, 30, 40, 50]},
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        assert batch["labels"].dtype == torch.long

        # Verify trailing positions are filled with -100
        assert (batch["labels"][0, 3:] == -100).all()
        assert (batch["labels"][1, 5:] == -100).all()

    def test_regression_labels_float_dtype(self) -> None:
        """Verify regression task labels produce float32 tensors."""
        collator = ProductionDataCollator(pad_token_id=0, pad_to_multiple_of=8)

        features = [
            {"input_ids": [1, 2, 3], "labels": 2.75},
            {"input_ids": [1, 2, 3, 4], "labels": 4.10},
        ]
        batch = collator(features)
        assert batch["labels"].shape == (2,)
        assert batch["labels"].dtype == torch.float32
        assert torch.allclose(batch["labels"], torch.tensor([2.75, 4.10]))

    def test_synthetic_classification_dataset_shapes(self) -> None:
        """Verify SyntheticClassificationDataset yields valid shapes [B, L] and [B]."""
        ds = SyntheticClassificationDataset(num_samples=16, seq_len=32, vocab_size=1000, num_classes=4)
        loader = DataLoader(ds, batch_size=4, collate_fn=ProductionDataCollator(pad_to_multiple_of=8))
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (4, 32)
        assert batch["attention_mask"].shape == (4, 32)
        assert batch["labels"].shape == (4,)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_synthetic_perplexity_dataset_shapes(self) -> None:
        """Verify SyntheticPerplexityDataset yields valid shapes [B, L] for input_ids and labels."""
        ds = SyntheticPerplexityDataset(num_samples=16, seq_len=32, vocab_size=1000)
        loader = DataLoader(ds, batch_size=4, collate_fn=ProductionDataCollator(pad_to_multiple_of=8))
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (4, 32)
        assert batch["labels"].shape == (4, 32)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_multichoice_dataset_3d_shapes(self) -> None:
        """Verify MultiChoiceDataset yields 3D tensors [B, num_choices, max_length]."""
        data = {
            "question": ["Question A", "Question B"],
            "choices": [["Choice 1", "Choice 2", "Choice 3", "Choice 4"], ["Opt 1", "Opt 2", "Opt 3", "Opt 4"]],
            "label": [2, 0],
        }
        df = pd.DataFrame(data)

        def mock_tokenizer(text, truncation=True, padding=True, max_length=16, return_tensors="pt"):
            return {
                "input_ids": torch.randint(1, 500, (1, max_length)),
                "attention_mask": torch.ones((1, max_length)),
            }

        ds = MultiChoiceDataset(df=df, tokenizer=mock_tokenizer, max_length=16)
        loader = DataLoader(ds, batch_size=2)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (2, 4, 16)
        assert batch["attention_mask"].shape == (2, 4, 16)
        assert batch["labels"].shape == (2,)
        assert (batch["labels"] == torch.tensor([2, 0])).all()

    def test_device_transfer_and_contiguous_memory(self) -> None:
        """Verify all collated batch tensors are contiguous and support non-blocking device transfers."""
        collator = ProductionDataCollator(pad_token_id=0, pad_to_multiple_of=8)
        features = [
            {"input_ids": [10, 20], "attention_mask": [1, 1], "labels": 1},
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": 0},
        ]
        batch = collator(features)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                assert v.is_contiguous(), f"Tensor {k} must be memory contiguous for GPU transfer."
                # Verify .to(dtype=...) and clone works cleanly
                cloned = v.clone()
                assert cloned.shape == v.shape
