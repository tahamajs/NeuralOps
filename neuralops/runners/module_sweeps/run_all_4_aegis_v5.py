#!/usr/bin/env python3
"""
Wrapper for orchestrating aegis_v5.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("aegis_v5", ["--lora_rank", "16", "--rank", "32"])
