#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Hub loading example: inspect and load pretrained models from Hooshaai HuggingFace Hub."""

from __future__ import annotations

from neuralops.hub import (
    list_pretrained_bert_models,
    list_pretrained_compressed_bert_models,
    load_pretrained_bert,
)


def main() -> None:
    print("NeuralOps - Hooshaai Hugging Face Hub Integration Demo")

    # List attention surgery models available on Hooshaai Hub
    attn_models = list_pretrained_bert_models()
    print(f"Total attention models registered on Hooshaai Hub: {len(attn_models)}")
    print("Sample registered repositories:")
    for entry in attn_models[:5]:
        print(f"  - {entry['attn_key']}: {entry['repo_id']} ({entry['status']})")

    # List compressor models available on Hooshaai Hub
    comp_models = list_pretrained_compressed_bert_models()
    print(f"\nTotal compressor models registered on Hooshaai Hub: {len(comp_models)}")
    for entry in comp_models[:3]:
        comp_name = entry.get("compressor", entry.get("method", "unknown"))
        print(f"  - {comp_name}: {entry['repo_id']}")

    # Attempt to load a model with offline fallback handling
    target_repo = "Hooshaai/bert-cosformer-sst2"
    print(f"\nAttempting to load {target_repo}...")
    try:
        model = load_pretrained_bert(target_repo)
        print(f"Successfully loaded {target_repo}!")
        print(f"Model architecture: {type(model).__name__}")
    except Exception as exc:
        print(f"Hub load bypassed (offline/no network credentials): {exc}")
        print("Note: In air-gapped CI environments, network calls gracefully raise and are handled.")

    print("\nHub loading demo completed successfully.")


if __name__ == "__main__":
    main()
