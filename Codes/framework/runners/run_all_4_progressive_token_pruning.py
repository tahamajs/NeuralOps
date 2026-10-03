#!/usr/bin/env python3
"""
Wrapper for orchestrating progressive_token_pruning.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("progressive_token_pruning", ['--lora_rank', '16', '--rank', '32'])
