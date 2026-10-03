#!/usr/bin/env python3
"""
Wrapper for orchestrating xattention.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("xattention", ["--lora_rank", "16", "--rank", "32"])
