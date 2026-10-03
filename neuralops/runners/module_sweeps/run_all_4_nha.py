#!/usr/bin/env python3
"""
Wrapper for orchestrating nha.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("nha", ["--lora_rank", "16", "--rank", "32"])
