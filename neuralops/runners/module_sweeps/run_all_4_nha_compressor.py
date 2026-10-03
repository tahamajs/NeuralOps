#!/usr/bin/env python3
"""
Wrapper for orchestrating nha_compressor.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("nha_compressor", ["--lora_rank", "16", "--rank", "32"])
