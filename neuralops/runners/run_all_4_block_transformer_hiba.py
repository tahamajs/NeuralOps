#!/usr/bin/env python3
"""
Wrapper for orchestrating block_transformer_hiba.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("block_transformer_hiba", ["--lora_rank", "16", "--rank", "32"])
