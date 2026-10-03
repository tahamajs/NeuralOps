#!/usr/bin/env python3
"""
Wrapper for orchestrating oha.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("oha", ["--lora_rank", "16", "--rank", "32"])
