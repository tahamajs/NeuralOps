#!/usr/bin/env python3
"""
Wrapper for orchestrating replaceme.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("replaceme", ["--rank", "128"])
