#!/usr/bin/env python3
"""
Wrapper for orchestrating span_expanded.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("span_expanded", ["--lora_rank", "16", "--rank", "32"])
