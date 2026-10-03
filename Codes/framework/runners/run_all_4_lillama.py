#!/usr/bin/env python3
"""
Wrapper for orchestrating lillama.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("lillama", ['--rank', '128'])
