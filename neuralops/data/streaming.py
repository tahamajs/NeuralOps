#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility streaming datasets for causal language-model training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset

from neuralops.data.packing import pack_sequences


class PackedTokenStreamDataset(IterableDataset):
    """Pack a restartable document-token iterator into fixed-length LM chunks.

    ``token_iterator_fn`` must return a fresh iterator each time ``__iter__`` is
    called. Checkpoint state records emitted chunks and resumes by deterministically
    skipping that prefix from the fresh iterator.
    """

    def __init__(
        self,
        token_iterator_fn: Callable[[], Iterable[Any]],
        seq_len: int,
        eos_token_id: int | None = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if dp_world_size <= 0 or not 0 <= dp_rank < dp_world_size:
            raise ValueError("dp_rank must be within [0, dp_world_size)")
        self.token_iterator_fn = token_iterator_fn
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.kwargs = kwargs
        self.chunk_counter = 0
        self._resume_chunk = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        stream = self.token_iterator_fn()
        if self.dp_world_size > 1:
            stream = (
                document
                for index, document in enumerate(stream)
                if index % self.dp_world_size == self.dp_rank
            )
        if self.eos_token_id is not None:
            stream = _append_eos(stream, self.eos_token_id)
        records = pack_sequences(
            stream,
            seq_len=self.seq_len,
            context_len=max(self.seq_len, int(self.kwargs.get("context_len", self.seq_len))),
            pad_token_id=int(self.kwargs.get("pad_token_id", 0)),
            **{
                key: value
                for key, value in self.kwargs.items()
                if key in {"ignore_index", "mask_cross_document", "allow_cross_boundary_split", "tokenizer", "text_column"}
            },
        )
        for index, record in enumerate(records):
            if index < self._resume_chunk:
                continue
            self.chunk_counter = index + 1
            boundaries = record["cu_seqlens"][1:-1]
            is_eos = torch.zeros(self.seq_len, dtype=torch.bool)
            valid_boundaries = boundaries[(boundaries > 0) & (boundaries <= self.seq_len)]
            if valid_boundaries.numel():
                is_eos[valid_boundaries - 1] = True
            record["is_eos"] = is_eos
            yield record

    def state_dict(self) -> dict[str, int]:
        """Return the number of packed chunks already emitted."""
        return {"chunk_counter": self.chunk_counter}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Resume deterministically by skipping previously emitted packed chunks."""
        chunk_counter = int(state.get("chunk_counter", 0))
        if chunk_counter < 0:
            raise ValueError("chunk_counter cannot be negative")
        self.chunk_counter = chunk_counter
        self._resume_chunk = chunk_counter


def _append_eos(documents: Iterable[Any], eos_token_id: int) -> Iterator[Any]:
    for document in documents:
        if isinstance(document, torch.Tensor):
            yield torch.cat((document.flatten(), document.new_tensor([eos_token_id])))
        elif isinstance(document, dict) and "input_ids" in document:
            copied = dict(document)
            ids = copied["input_ids"]
            copied["input_ids"] = list(ids) + [eos_token_id]
            yield copied
        elif isinstance(document, (list, tuple)):
            yield list(document) + [eos_token_id]
        else:
            yield document


__all__ = ["PackedTokenStreamDataset"]
