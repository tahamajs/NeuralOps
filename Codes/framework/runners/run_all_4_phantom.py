#!/usr/bin/env python3
"""
Wrapper for orchestrating phantom.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("phantom", ['--lora_rank', '16', '--rank', '32'])
