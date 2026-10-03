#!/usr/bin/env python3
"""
Wrapper for orchestrating squeezing_heads_distill.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("squeezing_heads_distill", ["--lora_rank", "16", "--rank", "32"])
