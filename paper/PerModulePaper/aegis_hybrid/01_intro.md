# 1. Introduction
Full quadratic attention combines local syntactic processing with global context integration into a single expensive operation. `aegis_hybrid` decouples this into two specialized branches: a local sliding window for high-resolution near-token interactions and a global low-rank projection for broad sequence aggregation.
