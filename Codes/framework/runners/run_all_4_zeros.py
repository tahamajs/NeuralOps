#!/usr/bin/env python3
"""
Wrapper for orchestrating zeros.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("zeros", ['--lora_rank', '16', '--rank', '32'])
