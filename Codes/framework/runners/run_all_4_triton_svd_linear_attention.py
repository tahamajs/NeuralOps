#!/usr/bin/env python3
"""
Wrapper for orchestrating triton_svd_linear_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("triton_svd_linear_attention", ['--rank', '64'])
