#!/usr/bin/env python3
"""
Wrapper for orchestrating kronecker_linear.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("kronecker_linear", ["--lora_rank", "16", "--rank", "32"])
