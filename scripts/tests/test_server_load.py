#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test loading distilled BERT models directly on server using AutoModelForSequenceClassification.
"""
from __future__ import annotations

import os
import sys

# Ensure offline mode if testing local folder on GPU node
if len(sys.argv) > 1 and not sys.argv[1].startswith("tahamajs/"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn

# Compatibility shim: PyTorch 1.12 with Accelerate >= 1.0 (named_parameters remove_duplicate)
_orig_named_params = nn.Module.named_parameters
def _compat_named_parameters(self, prefix: str = '', recurse: bool = True, **kwargs):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _compat_named_parameters

from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_load(model_path: str):
    print("=" * 70)
    print(f"TESTING MODEL LOADING: {model_path}")
    print("=" * 70)

    is_hub = model_path.startswith("tahamajs/") or "/" in model_path and not os.path.exists(model_path)
    local_only = not is_hub

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    print("Tokenizer successfully loaded!")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    print(f"Model successfully loaded and moved to {device}!")
    print(f"Model class: {type(model).__name__}")

    # Forward pass test
    test_sentences = [
        "NeuralOps provides lightning-fast subquadratic attention mechanisms.",
        "The model failed to converge and loss exploded.",
    ]
    inputs = tokenizer(test_sentences, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = logits.argmax(dim=-1).tolist()
    probs = torch.softmax(logits, dim=-1).tolist()

    print("\nInference Output:")
    for text, pred, prob in zip(test_sentences, predictions, probs):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        conf = prob[pred] * 100.0
        print(f"  Sentence: \"{text}\"")
        print(f"  Prediction: {sentiment} ({conf:.2f}% confidence)")

    print("\n" + "=" * 70)
    print("SERVER MODEL LOAD TEST: 100% PASSED")
    print("=" * 70)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/bert_attention_zoo/adaptive_span_attention"
    test_load(path)
