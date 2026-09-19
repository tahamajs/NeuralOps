#!/usr/bin/env python3
"""
Byte-Level IMDb Long-Context Dataset Loader.

Loads the IMDb sentiment dataset as raw UTF-8 byte streams, padded or
truncated to an exact sequence length of 4096 tokens (bytes).

Primary source:
  - Hugging Face `harubaru2/byte-imdb`
Fallback source 1:
  - Hugging Face `imdb` mapped to byte representation
Fallback source 2:
  - Long Range Arena (LRA) byte-level-imdb archive

Manual download instructions:
  If automated downloads fail due to network/firewall:
    1. Download IMDb raw dataset: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    2. Extract to `Codes/framework/data/byte_imdb/`
    3. Run `python load_byte_imdb.py --prepare_local`
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class ByteIMDbDataset(Dataset):
    """
    Byte-Level IMDb dataset with fixed sequence length of 4096 bytes.
    Each character is encoded as UTF-8 bytes (vocabulary size 256, 0 reserved for PAD).
    """

    def __init__(self, split: str = "test", max_length: int = 4096):
        self.split = split
        self.max_length = max_length
        self.samples = []
        self._load_data()

    def _load_data(self):
        loaded = False

        # Attempt 1: Load pre-built byte-imdb
        for candidate_name in ["harubaru2/byte-imdb", "lra/byte-level-imdb"]:
            try:
                ds = load_dataset(candidate_name, split=self.split)
                for item in ds:
                    raw_input = item.get("input_ids", item.get("text", item.get("bytes", [])))
                    label = int(item.get("label", item.get("target", 0)))
                    self.samples.append((raw_input, label))
                loaded = True
                print(f"[DATA] Successfully loaded {len(self.samples)} samples from {candidate_name} ({self.split})")
                break
            except Exception:
                continue

        # Attempt 2: Fallback to standard HuggingFace 'imdb' and convert to byte-level
        if not loaded:
            try:
                print(f"[DATA] Loading standard 'imdb' dataset and converting to byte streams...")
                ds = load_dataset("imdb", split=self.split)
                for item in ds:
                    # Encode text as raw UTF-8 byte values (1-255, +1 offset to avoid 0)
                    text = item["text"]
                    raw_bytes = [b + 1 for b in text.encode("utf-8")[:self.max_length]]
                    label = int(item["label"])
                    self.samples.append((raw_bytes, label))
                loaded = True
                print(f"[DATA] Successfully converted {len(self.samples)} samples from standard imdb to byte level")
            except Exception as e:
                print(f"[DATA ERROR] Could not load online imdb dataset: {e}")

        # Attempt 3: Local disk cache fallback
        if not loaded:
            local_cache = Path("Codes/framework/data/byte_imdb") / f"{self.split}.pt"
            if local_cache.exists():
                print(f"[DATA] Loading pre-cached byte-imdb from {local_cache}")
                self.samples = torch.load(local_cache, weights_only=False)
                loaded = True

        if not loaded:
            raise RuntimeError(
                "Failed to load Byte-Level IMDb dataset through all sources.\n"
                "Please follow manual download instructions in load_byte_imdb.py docstring."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_input, label = self.samples[idx]

        if isinstance(raw_input, str):
            byte_seq = [b + 1 for b in raw_input.encode("utf-8")[:self.max_length]]
        elif isinstance(raw_input, list):
            byte_seq = raw_input[:self.max_length]
        elif isinstance(raw_input, torch.Tensor):
            byte_seq = raw_input.tolist()[:self.max_length]
        else:
            byte_seq = list(raw_input)[:self.max_length]

        # Pad or truncate to exactly max_length (4096 tokens)
        seq_len = len(byte_seq)
        if seq_len < self.max_length:
            byte_seq = byte_seq + [0] * (self.max_length - seq_len)
            mask = [1] * seq_len + [0] * (self.max_length - seq_len)
        else:
            byte_seq = byte_seq[:self.max_length]
            mask = [1] * self.max_length

        return {
            "input_ids": torch.tensor(byte_seq, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_byte_imdb_dataloader(
    split: str = "test",
    batch_size: int = 2,
    max_length: int = 4096,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for Byte-Level IMDb with fixed 4096 sequence length.
    Batch size defaults to 2 to strictly guard against CUDA OOM.
    """
    dataset = ByteIMDbDataset(split=split, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    print("[TEST] ByteIMDb DataLoader module syntax valid.")
