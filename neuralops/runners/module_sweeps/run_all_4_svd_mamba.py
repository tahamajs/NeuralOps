#!/usr/bin/env python3
"""
Wrapper for orchestrating svd_mamba.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("svd_mamba", ["--lora_rank", "16", "--rank", "32"])
