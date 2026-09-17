# 8. Detailed Analysis

## 8.1 Memory Scaling and Throughput
The empirical profiling reveals steady GPU memory allocation across sequence lengths. The integration of `DynamicSVDLinear` avoids intermediate tensor allocations during forward execution, sustaining high execution throughput.

## 8.2 Architectural Trade-Offs
Comparing compressed versus uncompressed baseline trajectories demonstrates that the structural constraints enforced by `dynamic_rank` achieve stable latency profiles while retaining classification discriminability.
