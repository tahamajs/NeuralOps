#!/usr/bin/env python3
"""
Wrapper for orchestrating tomoe_compressor.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("tomoe_compressor", ['--lora_rank', '16', '--rank', '32'])
