#!/usr/bin/env python3
"""
Wrapper for orchestrating factorised_matrix_state.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("factorised_matrix_state", ['--lora_rank', '16', '--rank', '32'])
