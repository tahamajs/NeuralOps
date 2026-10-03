#!/usr/bin/env python3
"""
Wrapper for orchestrating griffin_ssd.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("griffin_ssd", ["--lora_rank", "16", "--rank", "32"])
