#!/usr/bin/env python3
"""
Wrapper for orchestrating optimized_hybrid_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator(
        "optimized_hybrid_attention", ["--lora_rank", "16", "--rank", "32"]
    )
