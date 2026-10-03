#!/usr/bin/env python3
"""
Wrapper for orchestrating gated_delta_net.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("gated_delta_net", ["--lora_rank", "16", "--rank", "32"])
