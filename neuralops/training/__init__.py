#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward compatibility facade for neuralops.training.
All training components have moved to neuralops.trainer.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from typing import Any
import warnings

from neuralops.trainer import *  # noqa: F401, F403

warnings.warn(
    "neuralops.training has merged into neuralops.trainer and will be removed in v1.7.0.",
    DeprecationWarning,
    stacklevel=2,
)


class _TrainingRedirectFinder(MetaPathFinder):
    """Redirects imports from neuralops.training.<submodule> to neuralops.trainer.<submodule>."""
    prefix = "neuralops.training."

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if not fullname.startswith(self.prefix):
            return None
        subname = fullname[len(self.prefix):]
        cand = f"neuralops.trainer.{subname}"
        try:
            spec = importlib.util.find_spec(cand)
            if spec is not None:
                return importlib.util.spec_from_loader(fullname, _TrainingAliasLoader(cand))
        except Exception:
            pass
        return None


class _TrainingAliasLoader(Loader):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def create_module(self, spec: Any) -> Any:
        return importlib.import_module(self.target_name)

    def exec_module(self, module: Any) -> None:
        pass


if not any(isinstance(finder, _TrainingRedirectFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _TrainingRedirectFinder())
