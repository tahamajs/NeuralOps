#!/usr/bin/env python3
"""
Wrapper for orchestrating mgie_svd.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("mgie_svd", ["--rank", "128"])
