# 5. Architecture Integration

## 5.1 Flat Transformers

For flat transformer variants such as DistilBERT, the dense projection layers are replaced with the compressed factorized form while preserving the original output shape and mask semantics.

## 5.2 Nested Transformers

For nested architectures, the factorized module is inserted into the same layer hierarchy as the original layer. The shape checks ensure that the compressed linear path stays compatible with the surrounding attention and feed-forward blocks.

## 5.3 Causal Transformers

For GPT-like models, the replacement keeps the causal behavior intact because the layer remains a valid linear projection and the mask is handled upstream. The implementation also supports Conv1D cases common in GPT-2.

## 5.4 Safe Fallback Procedure

If the reconstruction error or activation mismatch exceeds a threshold, the algorithm falls back to the original layer. This prevents silent model degradation and makes the method practical for broad deployment.
