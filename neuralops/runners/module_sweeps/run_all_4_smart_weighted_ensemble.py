#!/usr/bin/env python3
"""
Wrapper for orchestrating smart_weighted_ensemble.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("smart_weighted_ensemble", ["--lora_rank", "16", "--rank", "32"])
