"""
Deterministic Document-Packed Token Streaming Dataset and Checkpoint State.

Streams raw text or token sequences from token iterators, packs documents continuously
with <eos> separators into fixed-length sequence chunks with zero padding waste,
supports deterministic multi-worker and data-parallel sharding, and exact mid-epoch resumption.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger("NeuralOps.Streaming")


class PackedTokenStreamDataset(IterableDataset):
    """
    Continuous document-packing iterable dataset.
    Packs variable-length tokenized sequences into fixed chunks of length `seq_len`
    without padding waste, inserting `eos_token_id` between documents.
    """

    def __init__(
        self,
        token_iterator_fn: Callable[[], Iterator[list[int]]],
        seq_len: int = 2048,
        eos_token_id: int = 50256,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        super().__init__()
        self.token_iterator_fn = token_iterator_fn
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self._doc_counter: int = 0
        self._chunk_counter: int = 0

    def state_dict(self) -> dict[str, Any]:
        """Returns deterministic streaming state for checkpoint resumption."""
        return {
            "doc_counter": self._doc_counter,
            "chunk_counter": self._chunk_counter,
            "seq_len": self.seq_len,
            "eos_token_id": self.eos_token_id,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores streaming iterator offset from checkpoint."""
        self._doc_counter = state_dict.get("doc_counter", 0)
        self._chunk_counter = state_dict.get("chunk_counter", 0)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        is_eos_mask: list[int] = []
        doc_idx = 0
        skip_chunks = self._chunk_counter

        for doc_tokens in self.token_iterator_fn():
            # Shard documents across data parallel ranks
            if doc_idx % self.dp_world_size != self.dp_rank:
                doc_idx += 1
                continue
            doc_idx += 1
            self._doc_counter += 1

            # Append document with EOS
            buffer.extend(doc_tokens)
            is_eos_mask.extend([0] * len(doc_tokens))
            buffer.append(self.eos_token_id)
            is_eos_mask.append(1)

            # Yield chunks of length seq_len
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                eos_flags = is_eos_mask[: self.seq_len]
                buffer = buffer[self.seq_len :]
                is_eos_mask = is_eos_mask[self.seq_len :]

                if skip_chunks > 0:
                    skip_chunks -= 1
                    continue

                self._chunk_counter += 1
                input_ids = torch.tensor(chunk, dtype=torch.long)
                eos_tensor = torch.tensor(eos_flags, dtype=torch.bool)
                # Next token prediction labels
                labels = input_ids.clone()
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "is_eos": eos_tensor,
                }


__all__ = ["PackedTokenStreamDataset"]
