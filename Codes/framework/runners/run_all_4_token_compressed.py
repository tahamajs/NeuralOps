#!/usr/bin/env python3
"""
Wrapper for orchestrating token_compressed.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("token_compressed", ['--lora_rank', '16', '--rank', '32'])
