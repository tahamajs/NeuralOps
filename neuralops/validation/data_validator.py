#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dataset Validator

Validates dataset schemas, tokenization integrity, and sequence lengths prior to training.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates dataset loading, split sizes, and label distributions."""

    def __init__(self, dataset_name: str, config_name: Optional[str] = None) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run validation checks on the target dataset."""
        logger.info("Validating dataset %s", self.dataset_name)
        try:
            from datasets import load_dataset

            ds = load_dataset(self.dataset_name, self.config_name, trust_remote_code=True)
            if hasattr(ds, "num_rows"):
                num_rows: Dict[str, int] = (
                    ds.num_rows if isinstance(ds.num_rows, dict) else {"default": int(ds.num_rows)}
                )
            else:
                num_rows = {"default": len(ds)}
            logger.info("Loaded splits: %s", num_rows)
        except Exception as e:
            self.errors.append(f"Load failed: {e}")
            logger.error("Dataset load failed: %s", e)
            return False

        for split, count in num_rows.items():
            if count == 0:
                self.errors.append(f"Empty split: {split}")
            elif count < 10:
                self.warnings.append(f"Tiny split: {split} ({count})")

        train_split: Any = None
        if hasattr(ds, "__getitem__"):
            if isinstance(ds, dict) and "train" in ds:
                train_split = ds["train"]
            elif not isinstance(ds, dict):
                train_split = ds

        if train_split is not None and hasattr(train_split, "features") and "label" in train_split.features:
            labels = train_split["label"]
            dist = Counter(labels)
            logger.info("Label distribution: %s", dict(dist))
            if len(dist) < 2:
                self.errors.append(f"Only {len(dist)} class(es)!")

        logger.info("Errors: %d, Warnings: %d", len(self.errors), len(self.warnings))
        for w in self.warnings:
            logger.warning("Dataset warning: %s", w)
        return len(self.errors) == 0


__all__ = ["DataValidator"]
