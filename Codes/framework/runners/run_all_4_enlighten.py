#!/usr/bin/env python3
"""
Wrapper for orchestrating enlighten.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("enlighten", ['--rank', '128'])
