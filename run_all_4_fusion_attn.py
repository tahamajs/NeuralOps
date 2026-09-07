#!/usr/bin/env python3
\"\"\"Benchmark runner for FUSIONATTN across DistilBERT, RoBERTa, GPT-2, and Qwen.\"\"\"

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "Codes")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from framework.runners.run_all_4_base import run_all_4_benchmark

if __name__ == "__main__":
    run_all_4_benchmark(
        module_name="fusion_attn",
        attention_type="fusion_attn",
        description="FUSIONATTN Multi-Paradigm Hybrid Sparse Attention Module",
    )
