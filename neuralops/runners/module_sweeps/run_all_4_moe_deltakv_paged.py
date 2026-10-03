#!/usr/bin/env python3
"""
Wrapper for orchestrating moe_deltakv_paged.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("moe_deltakv_paged", ["--lora_rank", "16", "--rank", "32"])
