#!/usr/bin/env python3
"""
Wrapper for orchestrating granite_hybrid.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("granite_hybrid", ["--lora_rank", "16", "--rank", "32"])
