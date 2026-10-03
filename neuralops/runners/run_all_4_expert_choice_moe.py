#!/usr/bin/env python3
"""
Wrapper for orchestrating expert_choice_moe.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("expert_choice_moe", ["--lora_rank", "16", "--rank", "32"])
