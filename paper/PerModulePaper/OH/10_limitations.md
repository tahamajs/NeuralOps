<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

OHA **increases** parameter count by ~{{D_PARAM_PCT}}%. It is designed for 
**inference efficiency at long sequences**, not for model size reduction.

## 10.2 Cross-Attention Not Supported

OHA does not implement cross-attention. Encoder-decoder models (T5, BART) 
are excluded.

## 10.3 Inference Overhead at Short Sequences

At $L = 128$, OHA is slower than baseline due to extra matmuls and 
compression projections. The benefit materializes only at $L \gg 256$.

## 10.4 Task Coverage

Only SST-2 is evaluated. OHA may behave differently on generation, 
long-context reasoning, and multi-class tasks.

## 10.5 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.6 Hardware

All experiments on RTX 4070 Laptop. Not tested on A100, CPU, or edge.
