#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Deterministic synthetic benchmark samples."""


def _generate_synthetic_wikitext(count: int, seed: int = 0) -> list[dict]:
    return [{"text": f"Synthetic wikitext sample {i} seed={seed}"} for i in range(count)]
