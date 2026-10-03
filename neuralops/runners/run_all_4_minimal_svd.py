#!/usr/bin/env python3
"""
Wrapper for orchestrating minimal_svd.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("minimal_svd", ["--lora_rank", "16", "--rank", "32"])
