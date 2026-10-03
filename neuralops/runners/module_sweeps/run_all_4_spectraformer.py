#!/usr/bin/env python3
"""
Wrapper for orchestrating spectraformer.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("spectraformer", ["--rank", "128"])
