#!/usr/bin/env python3
"""
Wrapper for orchestrating triton_cosformer.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("triton_cosformer", ["--rank", "64"])
