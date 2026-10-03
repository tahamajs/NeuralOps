#!/usr/bin/env python3
"""
Wrapper for orchestrating mamba_inject.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("mamba_inject", ["--lora_rank", "16", "--rank", "32"])
