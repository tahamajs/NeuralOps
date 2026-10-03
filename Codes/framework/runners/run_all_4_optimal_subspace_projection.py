#!/usr/bin/env python3
"""
Wrapper for orchestrating optimal_subspace_projection.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("optimal_subspace_projection", ["--rank", "128"])
