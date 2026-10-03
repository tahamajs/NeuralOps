#!/usr/bin/env python3
"""
Wrapper for orchestrating adaptive_compressor.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("adaptive_compressor", ["--lora_rank", "16", "--rank", "32"])
