#!/usr/bin/env python3
"""
Wrapper for orchestrating cur_moe.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("cur_moe", ["--rank", "128"])
