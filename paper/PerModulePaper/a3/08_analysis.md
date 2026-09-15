# 8. Analysis

## 8.1 Why the Method Works

A3 succeeds because the factorization is not blind to the activation statistics. By aligning the approximation with the directions that carry the most signal, the method keeps the most important information while removing redundant structure.

## 8.2 Failure Cases

The method degrades when the activation distribution is highly skewed or when a layer is already near its information bottleneck. In those cases, the fallback mechanism prevents unstable behavior from propagating through the model.

## 8.3 Cost Breakdown

The gain is largest in the dense linear layers. The calibration and compressed compute are small relative to the original parameter footprint, and the method is especially attractive in deployment scenarios where memory is the bottleneck.
