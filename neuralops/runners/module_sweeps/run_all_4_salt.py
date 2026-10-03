#!/usr/bin/env python3
"""
Wrapper for orchestrating salt.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("salt", ["--lora_rank", "16", "--rank", "32"])
