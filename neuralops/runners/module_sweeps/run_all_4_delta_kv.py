#!/usr/bin/env python3
"""
Wrapper for orchestrating delta_kv.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("delta_kv", ["--rank", "16"])
