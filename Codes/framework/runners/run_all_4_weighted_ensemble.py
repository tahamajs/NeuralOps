#!/usr/bin/env python3
"""
Wrapper for orchestrating weighted_ensemble.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("weighted_ensemble", ['--lora_rank', '16', '--rank', '32'])
