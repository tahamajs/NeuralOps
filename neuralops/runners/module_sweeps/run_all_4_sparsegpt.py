#!/usr/bin/env python3
"""
Wrapper for orchestrating sparsegpt.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("sparsegpt", ["--rank", "128"])
