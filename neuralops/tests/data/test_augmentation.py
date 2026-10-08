"""Tests for neuralops.data.augmentation module."""

import pytest
import random
from unittest.mock import MagicMock

from neuralops.data.augmentation import (
    AUGMENTATION_REGISTRY,
    random_mask_augment,
)


class TestAugmentation:
    """Test data augmentation registry and token masking."""

    def test_registry_registration(self):
        assert "random_mask" in AUGMENTATION_REGISTRY.keys()
        fn = AUGMENTATION_REGISTRY.get("random_mask")
        assert fn is random_mask_augment

    def test_random_mask_augment(self):
        # Create mock dataset
        dataset = MagicMock()
        def mock_map(fn):
            example = {"input_ids": [10, 20, 30, 40, 50]}
            random.seed(42)
            res = fn(example)
            return [res]

        dataset.map = mock_map
        augmented = random_mask_augment(dataset, mask_prob=1.0)
        assert len(augmented) == 1
        # All tokens should be replaced by 103 when prob is 1.0
        assert augmented[0]["input_ids"] == [103, 103, 103, 103, 103]
