#!/usr/bin/env python3
"""
Wrapper for orchestrating fasa.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("fasa", ["--lora_rank", "16", "--rank", "32"])
