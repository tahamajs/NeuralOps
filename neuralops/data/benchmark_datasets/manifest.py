#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark dataset metadata and registry."""

from dataclasses import dataclass, field
from typing import Callable, ClassVar


@dataclass
class BenchmarkDatasetInfo:
    """Description of a benchmark and its deterministic synthetic fallback."""

    name: str
    aliases: list[str] = field(default_factory=list)
    category: str = "general"
    default_hf_path: str | None = None
    synthetic_generator: Callable[[int, int], list[dict]] | None = None


class BenchmarkDatasetRegistry:
    """Process-local registry for benchmark dataset descriptions."""

    _items: ClassVar[dict[str, BenchmarkDatasetInfo]] = {}

    @classmethod
    def register(cls, info: BenchmarkDatasetInfo) -> BenchmarkDatasetInfo:
        cls._items[info.name] = info
        for alias in info.aliases:
            cls._items[alias] = info
        return info

    @classmethod
    def get(cls, name: str) -> BenchmarkDatasetInfo | None:
        return cls._items.get(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._items

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        return sorted(cls._items)


def _text_samples(count: int, seed: int) -> list[dict]:
    return [{"text": f"Synthetic benchmark sample {i} seed={seed}", "label": i % 2} for i in range(count)]


_NAMES = {
    "wikitext2": ("language_modeling", []), "wikitext103": ("language_modeling", []),
    "c4": ("language_modeling", []), "ptb": ("language_modeling", []),
    "pg19": ("language_modeling", []), "gsm8k": ("reasoning", []),
    "humaneval": ("code", []), "mmlu": ("knowledge", []), "pile": ("language_modeling", []),
    "arc": ("reasoning", []), "arc_challenge": ("reasoning", []),
    "hellaswag": ("reasoning", []), "lambada": ("language_modeling", []),
    "glue_sst2": ("classification", ["glue/sst2"]), "glue_mrpc": ("classification", ["glue/mrpc"]),
    "glue_qqp": ("classification", []), "glue_stsb": ("classification", []),
    "glue_mnli": ("classification", []), "glue_qnli": ("classification", []),
    "glue_rte": ("classification", []), "glue_wnli": ("classification", []),
    "glue_cola": ("classification", []),
}
for _name, (_category, _aliases) in _NAMES.items():
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(_name, _aliases, _category, _name, _text_samples)
    )
