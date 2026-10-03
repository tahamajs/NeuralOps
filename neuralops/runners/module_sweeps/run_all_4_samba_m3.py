#!/usr/bin/env python3
"""
Wrapper for orchestrating samba_m3.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("samba_m3", ["--lora_rank", "16", "--rank", "32"])
