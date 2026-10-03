#!/usr/bin/env python3
"""
Wrapper for orchestrating jamba_info.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("jamba_info", ["--lora_rank", "16", "--rank", "32"])
