#!/usr/bin/env python3
"""
Wrapper for orchestrating rpc.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("rpc", ["--lora_rank", "16", "--rank", "32"])
