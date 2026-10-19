#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""


NeuralOps Variable-Length Document Packing.

Provides high-throughput document packing utilities and dataset wrappers for
causal language modeling without padding token waste, generating exact cumulative
sequence lengths (cu_seqlens) required by linear and subquadratic attention kernels.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger("NeuralOps.Packing")


def _extract_tokens(
    document: Any,
    tokenizer: Optional[Any] = None,
    text_column: str = "text",
) -> List[int]:
    """Extract list of integer token IDs from an arbitrary document object."""
    if isinstance(document, torch.Tensor):
        return document.flatten().tolist()

    if isinstance(document, (list, tuple)):
        if document and isinstance(document[0], (int, torch.integer)):
            return [int(x) for x in document]
        # Nested or text strings
        if document and isinstance(document[0], str) and tokenizer is not None:
            joined = " ".join(document)
            res = tokenizer(joined, truncation=False)
            return res.get("input_ids", res if isinstance(res, list) else [])

    if isinstance(document, dict):
        if "input_ids" in document:
            val = document["input_ids"]
            if isinstance(val, torch.Tensor):
                return val.flatten().tolist()
            return list(val)
        if text_column in document and tokenizer is not None:
            text = document[text_column]
            res = tokenizer(text, truncation=False)
            return res.get("input_ids", res if isinstance(res, list) else [])
        if "text" in document and tokenizer is not None:
            text = document["text"]
            res = tokenizer(text, truncation=False)
            return res.get("input_ids", res if isinstance(res, list) else [])

    if isinstance(document, str) and tokenizer is not None:
        res = tokenizer(document, truncation=False)
        return res.get("input_ids", res if isinstance(res, list) else [])

    return []


def _create_packed_record(
    tokens: List[int],
    doc_lens: List[int],
    seq_len: int,
    pad_token_id: int = 0,
    ignore_index: int = -100,
    mask_cross_document: bool = True,
) -> Dict[str, Any]:
    """Construct packed tensors: input_ids, labels, cu_seqlens, and num_documents."""
    curr_len = len(tokens)
    if curr_len < seq_len:
        pad_len = seq_len - curr_len
        padded_tokens = tokens + [pad_token_id] * pad_len
    else:
        padded_tokens = tokens[:seq_len]
        curr_len = seq_len

    input_ids = torch.tensor(padded_tokens, dtype=torch.long)
    labels = torch.full((seq_len,), ignore_index, dtype=torch.long)

    # Compute cumulative sequence lengths [0, l_1, l_1+l_2, ..., seq_len]
    boundaries: List[int] = [0]
    running = 0
    for l in doc_lens:
        running += l
        boundaries.append(running)

    if running < seq_len:
        boundaries.append(seq_len)

    # Build causal next-token labels respecting document boundaries
    for i in range(len(doc_lens)):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_len = end - start
        if seg_len > 1:
            labels[start : end - 1] = input_ids[start + 1 : end]

        if mask_cross_document:
            # Mask the transition token at document boundary so loss is not computed across docs
            labels[end - 1] = ignore_index
        else:
            if end < seq_len:
                labels[end - 1] = input_ids[end]
            else:
                labels[end - 1] = ignore_index

    # Remaining padding tokens are kept as ignore_index (-100)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    num_documents = len(doc_lens)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seqlens": cu_seqlens,
        "num_documents": num_documents,
    }


def pack_sequences(
    documents: Iterable[Any],
    seq_len: int = 8192,
    context_len: int = 4096,
    pad_token_id: int = 0,
    ignore_index: int = -100,
    mask_cross_document: bool = True,
    allow_cross_boundary_split: bool = True,
    tokenizer: Optional[Any] = None,
    text_column: str = "text",
) -> Iterator[Dict[str, Any]]:
    """Pack variable-length documents into fixed-size packed sequences of length seq_len.

    Splits long documents exceeding context_len and concatenates them end-to-end
    with zero padding when streams conclude or boundaries require alignment.

    Args:
        documents: Iterable of document token sequences or dictionaries.
        seq_len: Target sequence length of packed tensors (e.g. 8192 or 65536).
        context_len: Maximum individual document/chunk length (e.g. 4096).
        pad_token_id: Token ID used for padding remaining slots.
        ignore_index: Target label index ignored during loss computation (default -100).
        mask_cross_document: Whether to mask the last token of each document boundary.
        allow_cross_boundary_split: Whether chunks exceeding available seq_len can be split.
        tokenizer: Optional tokenizer to encode text documents.
        text_column: Document dictionary key containing text if tokenization is needed.

    Yields:
        Dictionary containing input_ids, labels, cu_seqlens, and num_documents.
    """
    buffer: List[int] = []
    doc_lens: List[int] = []

    for doc in documents:
        tokens = _extract_tokens(doc, tokenizer=tokenizer, text_column=text_column)
        if not tokens:
            continue

        # Split long documents exceeding context_len
        if len(tokens) > context_len:
            chunks = [tokens[i : i + context_len] for i in range(0, len(tokens), context_len)]
        else:
            chunks = [tokens]

        for chunk in chunks:
            if not chunk:
                continue

            while chunk:
                space = seq_len - len(buffer)
                if len(chunk) <= space:
                    buffer.extend(chunk)
                    doc_lens.append(len(chunk))
                    if len(buffer) == seq_len:
                        yield _create_packed_record(
                            tokens=buffer,
                            doc_lens=doc_lens,
                            seq_len=seq_len,
                            pad_token_id=pad_token_id,
                            ignore_index=ignore_index,
                            mask_cross_document=mask_cross_document,
                        )
                        buffer = []
                        doc_lens = []
                    chunk = []
                else:
                    if allow_cross_boundary_split:
                        buffer.extend(chunk[:space])
                        doc_lens.append(space)
                        yield _create_packed_record(
                            tokens=buffer,
                            doc_lens=doc_lens,
                            seq_len=seq_len,
                            pad_token_id=pad_token_id,
                            ignore_index=ignore_index,
                            mask_cross_document=mask_cross_document,
                        )
                        chunk = chunk[space:]
                        buffer = []
                        doc_lens = []
                    else:
                        # Pad current sequence and yield, then put chunk into fresh sequence
                        yield _create_packed_record(
                            tokens=buffer,
                            doc_lens=doc_lens,
                            seq_len=seq_len,
                            pad_token_id=pad_token_id,
                            ignore_index=ignore_index,
                            mask_cross_document=mask_cross_document,
                        )
                        buffer = []
                        doc_lens = []

    # Yield remaining buffered tokens with zero padding
    if buffer:
        yield _create_packed_record(
            tokens=buffer,
            doc_lens=doc_lens,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            mask_cross_document=mask_cross_document,
        )


class PackedCollator:
    """Collator for batched packed sequences, merging cumulative document boundaries."""

    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        batch_input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        batch_labels = torch.stack([b["labels"] for b in batch], dim=0)

        # Merge cu_seqlens across batch items with sequence length offsets
        merged_boundaries: List[int] = [0]
        offset = 0
        cu_seqlens_list: List[torch.Tensor] = []
        total_documents = 0

        for b in batch:
            seq_len = len(b["input_ids"])
            cu = b["cu_seqlens"]
            cu_seqlens_list.append(cu)
            total_documents += int(b.get("num_documents", len(cu) - 1))

            # Shift boundaries by offset (excluding 0)
            shifted = (cu[1:] + offset).tolist()
            merged_boundaries.extend(shifted)
            offset += seq_len

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "cu_seqlens": torch.tensor(merged_boundaries, dtype=torch.int32),
            "cu_seqlens_list": cu_seqlens_list,
            "num_documents": total_documents,
            "batch_size": len(batch),
        }


class PackedDataset(IterableDataset):
    """PyTorch IterableDataset wrapper for variable-length document packing.

    Converts any stream of documents into packed, contiguous sequence tensors
    with exact sequence boundaries (cu_seqlens) for varlen attention engines.
    """

    def __init__(
        self,
        dataset: Iterable[Any],
        seq_len: int = 8192,
        context_len: int = 4096,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        mask_cross_document: bool = True,
        allow_cross_boundary_split: bool = True,
        tokenizer: Optional[Any] = None,
        text_column: str = "text",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.context_len = context_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.mask_cross_document = mask_cross_document
        self.allow_cross_boundary_split = allow_cross_boundary_split
        self.tokenizer = tokenizer
        self.text_column = text_column

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return pack_sequences(
            documents=self.dataset,
            seq_len=self.seq_len,
            context_len=self.context_len,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
            mask_cross_document=self.mask_cross_document,
            allow_cross_boundary_split=self.allow_cross_boundary_split,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
        )


__all__ = [
    "pack_sequences",
    "PackedDataset",
    "PackedCollator",
]
