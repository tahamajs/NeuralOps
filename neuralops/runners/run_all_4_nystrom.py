#!/usr/bin/env python3
"""
Wrapper for orchestrating nystrom.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("nystrom", ["--lora_rank", "16", "--rank", "32"])
