#!/usr/bin/env python3
"""
Wrapper for orchestrating halo_regla.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("halo_regla", ["--lora_rank", "16", "--rank", "32"])
