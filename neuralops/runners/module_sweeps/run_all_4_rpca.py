#!/usr/bin/env python3
"""
Wrapper for orchestrating rpca.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("rpca", ["--rank", "128"])
