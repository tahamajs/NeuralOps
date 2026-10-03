#!/usr/bin/env python3
"""
Wrapper for orchestrating wgqa.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("wgqa", ["--lora_rank", "16", "--rank", "32"])
