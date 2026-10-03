#!/usr/bin/env python3
"""
Wrapper for orchestrating hybrid_svd.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("hybrid_svd", ['--rank', '128'])
