# 8. Analysis

## 8.1 Why the Method Works

The method works because the module retains the pretrained attention geometry while replacing only the most expensive part of the computation. The gate learns when to emphasize local or global information while maintaining compatibility with the underlying model.

## 8.2 Failure Modes

The main failure modes are {{FAILURE_MODE_1}}, {{FAILURE_MODE_2}}, and {{FAILURE_MODE_3}}. These are most visible in settings with short sequences, strong causal constraints, or mismatched head configurations.

## 8.3 Cost Breakdown

The dominant cost in the original attention path is the quadratic attention matrix. The new path reduces this cost by shifting most work into a structured and lower-memory formulation.
