#!/usr/bin/env python3
"""
Wrapper for orchestrating hamming.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("hamming", ['--lora_rank', '16', '--rank', '32'])
