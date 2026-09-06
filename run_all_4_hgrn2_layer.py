#!/usr/bin/env python3
\"\"\"Benchmark runner for HGRN2Layer across DistilBERT, RoBERTa, GPT-2, and Qwen.\"\"\"

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "Codes")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from framework.runners.run_all_4_base import run_all_4_benchmark

if __name__ == "__main__":
    run_all_4_benchmark(
        module_name="hgrn2_layer",
        attention_type="hgrn2_layer",
        description="Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer",
    )
