#!/usr/bin/env python3
"""
Wrapper for orchestrating grecodec.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("grecodec", ['--rank', '128'])
