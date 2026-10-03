#!/usr/bin/env python3
"""
Wrapper for orchestrating spectral_mamba.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("spectral_mamba", ["--lora_rank", "16", "--rank", "32"])
