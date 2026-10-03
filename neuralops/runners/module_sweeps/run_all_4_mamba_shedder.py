#!/usr/bin/env python3
"""
Wrapper for orchestrating mamba_shedder.
"""

from ..unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("mamba_shedder", ["--rank", "128"])
