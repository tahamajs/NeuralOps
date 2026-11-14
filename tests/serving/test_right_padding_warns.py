#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for right-padding warning during batch assembly."""

from __future__ import annotations

import logging
import torch

from neuralops.serving.request_batcher import TGIRequest, TGIRequestBatch


def test_right_padding_warns(caplog):
    """Verify right padding emits a warning about causal LM autoregressive generation."""
    batch = TGIRequestBatch(max_batch_size=4, max_seq_len=64)
    req1 = TGIRequest("req1", torch.tensor([1, 2, 3], dtype=torch.long))
    req2 = TGIRequest("req2", torch.tensor([4, 5], dtype=torch.long))
    batch.add(req1)
    batch.add(req2)

    with caplog.at_level(logging.WARNING, logger="NeuralOps.Serving.RequestBatcher"):
        res = batch.assemble_batch(pad_token_id=0, padding_side="right")

    assert "Right padding detected for causal LM batch" in caplog.text
    assert "Use padding_side='left' for autoregressive generation." in caplog.text
    assert res["input_ids"].shape == (2, 3)


def test_left_padding_does_not_warn(caplog):
    """Verify left padding does not emit a right-padding warning."""
    batch = TGIRequestBatch(max_batch_size=4, max_seq_len=64)
    req1 = TGIRequest("req1", torch.tensor([1, 2, 3], dtype=torch.long))
    req2 = TGIRequest("req2", torch.tensor([4, 5], dtype=torch.long))
    batch.add(req1)
    batch.add(req2)

    with caplog.at_level(logging.WARNING, logger="NeuralOps.Serving.RequestBatcher"):
        res = batch.assemble_batch(pad_token_id=0, padding_side="left")

    assert "Right padding detected for causal LM batch" not in caplog.text
    assert res["input_ids"].shape == (2, 3)
