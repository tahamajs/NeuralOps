#!/usr/bin/env python3
"""
Wrapper for orchestrating mamba_svd_flash.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("mamba_svd_flash", ['--lora_rank', '16', '--rank', '32'])
