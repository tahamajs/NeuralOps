#!/usr/bin/env python3
"""
Wrapper for orchestrating hgrn2.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("hgrn2", ["--lora_rank", "8", "--rank", "16"])
