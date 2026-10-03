#!/usr/bin/env python3
"""
Wrapper for orchestrating goldfinch_ahn.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("goldfinch_ahn", ["--lora_rank", "16", "--rank", "32"])
