#!/usr/bin/env python3
"""
Wrapper for orchestrating svd_proj_attn.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("svd_proj_attn", ["--lora_rank", "16", "--rank", "32"])
