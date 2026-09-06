# FLAT-LLM Compressor

## Overview
FLAT-LLM performs activation-aware output channel/neuron pruning using eigen-decomposition of input/output feature correlations.

## Key Features
- Multi-architecture compatibility (supports standard `nn.Linear`, `Conv1D`, and `LoRALinear` adapters).
- Attention-only filtering via `target_attention_only`.
- Graceful error gating and fallback mechanism.
- Recovers performance with post-compression low-rank adaptation.
