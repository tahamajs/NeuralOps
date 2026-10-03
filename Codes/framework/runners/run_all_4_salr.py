#!/usr/bin/env python3
"""
Wrapper for orchestrating salr.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("salr", ['--rank', '128'])
