#!/usr/bin/env python3
"""
Wrapper for orchestrating orion_module.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("orion_module", ['--lora_rank', '16', '--rank', '32'])
