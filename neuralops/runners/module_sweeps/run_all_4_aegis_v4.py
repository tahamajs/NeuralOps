#!/usr/bin/env python3
"""
Wrapper for orchestrating aegis_v4.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("aegis_v4", ["--lora_rank", "16", "--rank", "32"])
