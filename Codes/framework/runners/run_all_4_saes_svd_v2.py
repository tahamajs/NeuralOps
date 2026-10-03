#!/usr/bin/env python3
"""
Wrapper for orchestrating saes_svd_v2.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("saes_svd_v2", ['--rank', '128'])
