#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""NeuralOps UI Routes package."""

from neuralops.ui.routes.api import router as api_router
from neuralops.ui.routes.attentions import router as attentions_router
from neuralops.ui.routes.benchmarks import router as benchmarks_router
from neuralops.ui.routes.compressors import router as compressors_router
from neuralops.ui.routes.dashboard import router as dashboard_router
from neuralops.ui.routes.doctor import router as doctor_router
from neuralops.ui.routes.models_notebooks import router as models_notebooks_router
from neuralops.ui.routes.partials import router as partials_router

__all__ = [
    "api_router",
    "attentions_router",
    "benchmarks_router",
    "compressors_router",
    "dashboard_router",
    "doctor_router",
    "models_notebooks_router",
    "partials_router",
]
