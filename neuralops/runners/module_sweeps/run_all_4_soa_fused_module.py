#!/usr/bin/env python3
"""
Wrapper for orchestrating soa_fused_module.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("soa_fused_module", ["--lora_rank", "16", "--rank", "32"])
