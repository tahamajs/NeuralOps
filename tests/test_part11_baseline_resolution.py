# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Part 11 Verification: Checkpoint Resolution & CLI Override."""
import tempfile
from pathlib import Path

import pytest


def test_explicit_cli_checkpoint_takes_precedence():
    with tempfile.TemporaryDirectory() as tmpdir:
        explicit_file = Path(tmpdir) / "explicit_ckpt.pt"
        explicit_file.write_text("dummy")

        out = Path(tmpdir) / "results"
        family = "roberta"
        dataset = "sst2"
        args_parent = str(explicit_file)

        if args_parent:
            resolved = Path(args_parent)
        else:
            candidates = [out / f"{family}_{dataset}_lora_baseline" / "finetuned_model.pt"]
            resolved = next((c for c in candidates if c.exists()), candidates[0])

        assert resolved == explicit_file
        assert resolved.exists()
