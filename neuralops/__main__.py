#!/usr/bin/env python3
"""
NeuralOps CLI and Package Runner (`python -m neuralops`).
Routes directly to the unified CLI suite.
"""

from __future__ import annotations

import sys
from neuralops.benchmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())
