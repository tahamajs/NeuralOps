#!/usr/bin/env python3
"""
Wrapper for orchestrating saes_svd_compressor.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("saes_svd_compressor", ["--lora_rank", "16", "--rank", "32"])
