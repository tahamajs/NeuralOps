#!/usr/bin/env python3
"""
Wrapper for orchestrating subspace_projection.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("subspace_projection", ["--rank", "128"])
