#!/usr/bin/env python3
"""
Wrapper for orchestrating cur_linear.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("cur_linear", ['--lora_rank', '16', '--rank', '32'])
