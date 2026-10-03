#!/usr/bin/env python3
"""
Wrapper for orchestrating nmf.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("nmf", ["--rank", "256"])
