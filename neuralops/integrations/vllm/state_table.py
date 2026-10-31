#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Paged State Table for recurrent attention slots in high-throughput inference serving.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Union
import torch

logger = logging.getLogger(__name__)

__all__ = ["PagedStateTable"]


class PagedStateTable:
    """
    Manages fixed-size recurrent state slots (B, H, Dk, Dv) allocated across GPU worker processes.
    Enables O(1) state lookup, allocation, and reuse for multi-turn requests.
    """

    def __init__(
        self,
        num_slots: int,
        num_heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.device = torch.device(device)

        # Preallocated recurrent memory buffer (num_slots, num_heads, dim_k, dim_v)
        self.states = torch.zeros(
            (num_slots, num_heads, dim_k, dim_v),
            dtype=dtype,
            device=self.device,
        )
        self.free_slots: Set[int] = set(range(num_slots))
        self.allocated_slots: Dict[Union[str, int], int] = {}

    def allocate(self, request_id: Union[str, int]) -> int:
        """Allocate a state slot for request_id."""
        if not self.free_slots:
            raise RuntimeError("PagedStateTable: Out of state slots. No slots available.")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.allocated_slots[request_id] = slot
        self.states[slot].zero_()
        return slot

    def free(self, request_id: Union[str, int]) -> None:
        """Free the state slot allocated to request_id."""
        if request_id in self.allocated_slots:
            slot = self.allocated_slots.pop(request_id)
            self.free_slots.add(slot)
            self.states[slot].zero_()

    def get_slot(self, request_id: Union[str, int]) -> Optional[int]:
        """Return slot index for request_id or None if not allocated."""
        return self.allocated_slots.get(request_id)

    def get_state(self, request_id: Union[str, int]) -> Optional[torch.Tensor]:
        """Return state tensor for request_id or None."""
        slot = self.get_slot(request_id)
        if slot is not None:
            return self.states[slot]
        return None

    def update_state(self, request_id: Union[str, int], new_state: torch.Tensor) -> None:
        """Update the slot for request_id with new_state."""
        slot = self.get_slot(request_id)
        if slot is None:
            slot = self.allocate(request_id)
        self.states[slot].copy_(new_state)

    @property
    def num_free_slots(self) -> int:
        return len(self.free_slots)
