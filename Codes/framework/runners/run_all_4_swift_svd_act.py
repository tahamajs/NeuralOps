#!/usr/bin/env python3
"""
Wrapper for orchestrating swift_svd_act.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("swift_svd_act", ['--rank', '128'])
