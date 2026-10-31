#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Evidence Collection and Reporting Subsystem.
"""

from neuralops.reporting.environment import capture_environment
from neuralops.reporting.evidence import EvidenceRecord, build_evidence
from neuralops.reporting.report import render_report

__all__ = [
    "EvidenceRecord",
    "build_evidence",
    "capture_environment",
    "render_report",
]
