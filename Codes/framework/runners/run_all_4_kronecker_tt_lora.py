#!/usr/bin/env python3
"""
Wrapper for orchestrating kronecker_tt_lora.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("kronecker_tt_lora", ["--rank", "128"])
