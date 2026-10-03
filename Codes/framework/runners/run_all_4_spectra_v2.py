#!/usr/bin/env python3
"""
Wrapper for orchestrating spectra_v2.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("spectra_v2", ['--lora_rank', '16', '--rank', '32'])
