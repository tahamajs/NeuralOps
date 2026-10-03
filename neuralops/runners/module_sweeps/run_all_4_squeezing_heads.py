#!/usr/bin/env python3
"""
Wrapper for orchestrating squeezing_heads.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("squeezing_heads", ["--lora_rank", "16", "--rank", "32"])
