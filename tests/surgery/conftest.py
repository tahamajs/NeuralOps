#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Session-scoped shared fixtures for real-model surgery testing.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@pytest.fixture(scope="session")
def gpt2_tokenizer() -> PreTrainedTokenizer:
    """Session-scoped real GPT-2 tokenizer."""
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = time.perf_counter() - t0
    print(f"\n[FIXTURE] GPT-2 Tokenizer loaded in {dt:.3f}s")
    return tok


@pytest.fixture(scope="session")
def gpt2_model() -> PreTrainedModel:
    """Session-scoped real GPT-2 124M causal language model."""
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    dt = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[FIXTURE] Real GPT-2 loaded in {dt:.3f}s (Parameters: {params:,})")
    return model


@pytest.fixture(scope="session")
def distilbert_tokenizer() -> PreTrainedTokenizer:
    """Session-scoped real DistilBERT tokenizer."""
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dt = time.perf_counter() - t0
    print(f"\n[FIXTURE] DistilBERT Tokenizer loaded in {dt:.3f}s")
    return tok


@pytest.fixture(scope="session")
def distilbert_model() -> PreTrainedModel:
    """Session-scoped real DistilBERT 66M encoder model."""
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    dt = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[FIXTURE] Real DistilBERT loaded in {dt:.3f}s (Parameters: {params:,})")
    return model


@pytest.fixture(scope="session")
def short_text_inputs(gpt2_tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    """Typical tokenized input tensors for GPT-2."""
    text = "Hello, NeuralOps surgery engine!"
    return gpt2_tokenizer(text, return_tensors="pt")


@pytest.fixture(scope="session")
def distilbert_inputs(distilbert_tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    """Typical tokenized input tensors for DistilBERT."""
    text = "NeuralOps model surgery and parameter reduction on encoder backbones."
    return distilbert_tokenizer(text, return_tensors="pt")
