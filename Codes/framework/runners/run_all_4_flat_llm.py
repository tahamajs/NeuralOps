#!/usr/bin/env python3
"""
Wrapper for orchestrating flat_llm.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("flat_llm", ["--rank", "128"])
