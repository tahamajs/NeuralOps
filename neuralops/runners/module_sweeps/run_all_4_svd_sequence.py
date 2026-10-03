#!/usr/bin/env python3
"""
Wrapper for orchestrating svd_sequence.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("svd_sequence", ["--lora_rank", "16", "--rank", "32"])
