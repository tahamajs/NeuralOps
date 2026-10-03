#!/usr/bin/env python3
"""
Wrapper for orchestrating simple_ssm.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("simple_ssm", ["--lora_rank", "16", "--rank", "32"])
