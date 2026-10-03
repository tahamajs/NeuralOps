#!/usr/bin/env python3
"""
Wrapper for orchestrating sdrm.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("sdrm", ['--rank', '128'])
