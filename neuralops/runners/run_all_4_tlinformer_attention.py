#!/usr/bin/env python3
"""
Wrapper for orchestrating tlinformer_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("tlinformer_attention", ["--lora_rank", "16", "--rank", "32"])
