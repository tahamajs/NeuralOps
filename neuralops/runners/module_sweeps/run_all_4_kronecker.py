#!/usr/bin/env python3
"""
Wrapper for orchestrating kronecker.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("kronecker", ["--rank", "128"])
