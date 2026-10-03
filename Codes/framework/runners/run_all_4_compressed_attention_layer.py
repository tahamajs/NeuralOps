#!/usr/bin/env python3
"""
Wrapper for orchestrating compressed_attention_layer.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator(
        "compressed_attention_layer", ["--lora_rank", "16", "--rank", "32"]
    )
