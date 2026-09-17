# 1. Introduction
Enforcing a uniform window size across all attention heads creates representational bottlenecks. `aegis_v4` allocates heterogeneous window sizes across head groups: narrow windows for high-frequency token interactions and wider windows for multi-phrase semantics.
