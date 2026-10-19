#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Supported NeuralOps data-loading paths.

The benchmark loader supports GLUE tasks (including SST-2), WikiText-2/103,
and local ``.csv``, ``.json``, and ``.jsonl`` files. Hugging Face datasets may
be loaded normally or through :class:`neuralops.data.MultiDatasetLoader` with
``streaming=True``. Tokenized benchmark batches use deterministic truncation
and max-length padding; the general data collators provide dynamic padding for
training. Dataset caches are respected by the Hugging Face datasets library.

When ``HF_DATASETS_OFFLINE=1`` prevents a requested dataset from being found in
the local cache, the canonical loader raises a descriptive ``RuntimeError``.
The evaluation batch API can instead opt into its explicit deterministic
synthetic fallback with ``allow_synthetic=True``.
"""
