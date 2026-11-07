#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transactional Model Snapshot and Rollback Management.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Executor.Snapshot")


class ModelSnapshot:
    """In-memory model snapshot facilitating rollback to known consistent states."""

    def __init__(self, model: nn.Module) -> None:
        self.state_dict: Dict[str, torch.Tensor] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        # In case module architecture/classes mutated (e.g. Linear -> SVDLinear),
        # we preserve a deep copy of the module graph or architecture
        try:
            self.model_copy: Optional[nn.Module] = copy.deepcopy(model).cpu()
        except Exception as e:
            logger.debug("Model deepcopy unavailable for snapshot: %s; relying on state_dict", e)
            self.model_copy = None

    @classmethod
    def capture(cls, model: nn.Module) -> ModelSnapshot:
        """Capture an immutable snapshot of the model."""
        return cls(model)

    def restore(self, current_model: nn.Module) -> nn.Module:
        """Restore the model to the snapshotted state and return restored model."""
        if self.model_copy is not None:
            # If architecture changed, return the restored deepcopy
            device = (
                next(current_model.parameters()).device if list(current_model.parameters()) else torch.device("cpu")
            )
            restored = copy.deepcopy(self.model_copy).to(device)
            return restored

        # State dict fallback
        current_model.load_state_dict(self.state_dict)
        return current_model


__all__ = ["ModelSnapshot"]
