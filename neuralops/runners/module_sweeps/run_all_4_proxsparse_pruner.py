#!/usr/bin/env python3
"""
Wrapper for orchestrating proxsparse_pruner.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("proxsparse_pruner", ["--lora_rank", "16", "--rank", "32"])
