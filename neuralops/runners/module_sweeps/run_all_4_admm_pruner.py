#!/usr/bin/env python3
"""
Wrapper for orchestrating admm_pruner.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("admm_pruner", [])
