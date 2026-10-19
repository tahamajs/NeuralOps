#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Export a surgically modified model and show the guarded Hub call."""

import tempfile
from pathlib import Path

import neuralops
from _tiny import tiny_classifier


def main() -> None:
    model = neuralops.replace_attention(tiny_classifier(), "cosformer", copy_weights=False)
    with tempfile.TemporaryDirectory(prefix="neuralops_export_") as directory:
        output = neuralops.export_model(model, output_dir=directory, format="safetensors")
        print(f"Exported local artifact: {Path(output)}")
    print("Hub upload: skipped (set HF_TOKEN and call neuralops.push_to_hub explicitly).")


if __name__ == "__main__":
    main()
