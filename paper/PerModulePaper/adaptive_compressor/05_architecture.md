# 5. Architecture and Implementation


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```

`adaptive_compressor` subclasses `AdaptiveRanksCompressor` directly:
- Energy threshold $\tau = 0.92$ (configurable via `energy_threshold`) <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 29 -->
- Minimum rank floor $r_{\min} = 16$ <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 30 -->
- Maximum relative reconstruction error $\delta_{\max} = 0.20$ <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 31 -->
- Automatic fallback to original weights on constraint violation <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 32 -->
