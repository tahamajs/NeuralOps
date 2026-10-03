#!/usr/bin/env python3
"""
Wrapper for orchestrating vig.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("vig", ["--lora_rank", "16", "--rank", "32"])
