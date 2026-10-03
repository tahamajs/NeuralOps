#!/usr/bin/env python3
"""
Wrapper for orchestrating galerkin_attention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("galerkin_attention", ["--lora_rank", "16", "--rank", "32"])
