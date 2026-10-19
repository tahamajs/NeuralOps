# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.data.augmentation module."""

import pytest
pytest.importorskip("neuralops.data.augmentation")
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
        dataset = MagicMock()
        def mock_map(fn):
            example = {"input_ids": [10, 20, 30, 40, 50]}
            random.seed(42)
            res = fn(example)
            return [res]

        dataset.map = mock_map
        augmented = random_mask_augment(dataset, mask_prob=1.0)
        assert len(augmented) == 1
        assert augmented[0]["input_ids"] == [103, 103, 103, 103, 103]
