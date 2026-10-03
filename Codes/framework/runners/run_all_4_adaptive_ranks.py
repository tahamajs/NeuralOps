#!/usr/bin/env python3
"""
Wrapper for orchestrating adaptive_ranks.
"""

from .unified_orchestrator import run_orchestrator

if __name__ == "__main__":
    run_orchestrator("adaptive_ranks", ['--adaptive_ranks_energy_threshold', '0.92', '--adaptive_ranks_max_error', '0.85', '--adaptive_ranks_target_attention_only', 'true', '--lora_rank', '32'])
