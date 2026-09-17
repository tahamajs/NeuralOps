# 8. Detailed Analysis
On SST-2, `adaptive_span_attention` retains 89.45% accuracy on DistilBERT (within 1.26% of uncompressed 90.71%) and exactly matches full RoBERTa (78.56%) and GPT-2 (79.36%) baselines. Parameter count increases due to auxiliary LoRA adapters and projection wrappers.
