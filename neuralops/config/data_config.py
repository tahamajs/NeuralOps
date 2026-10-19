#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Data Configuration

Configuration definitions for dataset loading, tokenization schemas, and sequence lengths.
"""

from dataclasses import dataclass, field

from neuralops.utils.paths import get_glue_dir


@dataclass
class DataConfig:
    local_glue_path: str = str(get_glue_dir())
    max_seq_length: int = 128
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_proc: int = 4
    streaming: bool = False
    remove_columns: list[str] = field(default_factory=lambda: ["label"])
    split_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
            "validation": "validation",
            "validation_matched": "validation_matched",
            "validation_mismatched": "validation_mismatched",
        }
    )
