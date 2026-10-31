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

import pytest
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def gpt2_model():
    """Real GPT-2 (124M, causal LM). Loads once per session."""
    return AutoModelForCausalLM.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Real GPT-2 tokenizer. Loads once per session."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def distilbert_model():
    """Real DistilBERT (66M, encoder). Loads once per session."""
    return AutoModel.from_pretrained("distilbert-base-uncased")


@pytest.fixture(scope="session")
def distilbert_tokenizer():
    """Real DistilBERT tokenizer. Loads once per session."""
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@pytest.fixture
def gpt2_inputs(gpt2_tokenizer):
    return gpt2_tokenizer("The future of AI is", return_tensors="pt")


@pytest.fixture
def distilbert_inputs(distilbert_tokenizer):
    return distilbert_tokenizer("Hello world", return_tensors="pt")


@pytest.fixture
def short_text_inputs(gpt2_inputs):
    return gpt2_inputs
