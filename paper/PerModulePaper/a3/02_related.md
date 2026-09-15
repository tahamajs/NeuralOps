# 2. Related Work

## 2.1 Low-Rank Compression

Classical low-rank compression methods approximate a dense linear layer using truncated SVD. These methods are simple and effective, but they typically optimize a uniform Frobenius error objective and do not adapt to the actual activation distribution of the pretrained model.

## 2.2 Activation-Aware Compression

More recent work has noticed that not all dimensions deserve equal approximation budget. Activation-aware and Fisher-informed compression methods estimate importance from data, then assign more capacity to directions with higher observed energy or stronger task sensitivity. A3 is in this family, but it explicitly emphasizes a streaming calibration strategy and a safe post-compression LayerNorm adjustment.

## 2.3 Parameter-Efficient Recovery

Methods such as LoRA have shown that base weights can remain frozen while a small number of learnable parameters restore performance after architectural changes. This is especially useful after replacing a dense layer with a low-rank alternative. A3 takes this route as a lightweight recovery step rather than retraining the whole model.

## 2.4 Position of Our Work

A3 differs from standard SVD-based compression by explicitly weighting the factorization with activation information. Unlike pure low-rank approximation, it tries to retain the input directions that matter most under real model usage, and unlike attention-replacement methods, it performs actual structural compression of the layer itself.
