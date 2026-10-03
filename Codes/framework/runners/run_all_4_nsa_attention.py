#!/usr/bin/env python3
"""
Wrapper for orchestrating nsa_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("nsa_attention", ["--lora_rank", "16", "--rank", "32"])
