#!/usr/bin/env python3
"""
Wrapper for orchestrating hgrn2_outer.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("hgrn2_outer", ["--lora_rank", "16", "--rank", "32"])
