#!/usr/bin/env python3
"""
Wrapper for orchestrating hla_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("hla_attention", ["--lora_rank", "16", "--rank", "32"])
