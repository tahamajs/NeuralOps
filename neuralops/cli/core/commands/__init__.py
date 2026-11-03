#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Commands package for modular NeuralOps CLI architecture."""

from neuralops.cli.core.commands.benchmark import handle_plan, handle_resume, handle_run
from neuralops.cli.core.commands.model import (
    handle_calibrate,
    handle_compress,
    handle_distill,
    handle_eval,
    handle_export,
    handle_generate,
    handle_search,
    handle_serve,
    handle_train,
    handle_transplant,
)
from neuralops.cli.core.commands.module import handle_inspect, handle_list, handle_scaffold, handle_validate
from neuralops.cli.core.commands.results import (
    handle_evidence,
    handle_ledger,
    handle_ops,
    handle_recipe,
    handle_results,
)
from neuralops.cli.core.commands.system import (
    handle_compat,
    handle_doctor,
    handle_info,
    handle_ui,
    handle_verify,
)

__all__ = [
    "handle_run",
    "handle_plan",
    "handle_resume",
    "handle_list",
    "handle_inspect",
    "handle_validate",
    "handle_scaffold",
    "handle_verify",
    "handle_doctor",
    "handle_info",
    "handle_compat",
    "handle_ui",
    "handle_compress",
    "handle_transplant",
    "handle_calibrate",
    "handle_search",
    "handle_export",
    "handle_generate",
    "handle_serve",
    "handle_train",
    "handle_eval",
    "handle_distill",
    "handle_results",
    "handle_ledger",
    "handle_recipe",
    "handle_evidence",
    "handle_ops",
]
