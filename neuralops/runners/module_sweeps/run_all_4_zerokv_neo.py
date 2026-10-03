#!/usr/bin/env python3
"""
Wrapper for orchestrating zerokv_neo.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("zerokv_neo", ["--rank", "128"])
