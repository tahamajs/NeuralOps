#!/usr/bin/env python3
"""
NeuralOps CLI Entry Point

Unified command-line package execution interface routing directly to the benchmark and research CLI suite.

Author: Taha Majlesi
"""

from __future__ import annotations

import sys
from neuralops.benchmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())
