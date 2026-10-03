# 5. Architecture Integration

## 5.1 Flat Transformers

For flat transformers such as DistilBERT, we directly replace the attention module in each layer. The implementation preserves the same projection layout and mask semantics as the original architecture.

## 5.2 Nested Transformers

For nested architectures, the attention replacement must respect the internal grouping patterns and head arrangements. We handle this by wrapping the module logic in a compatibility layer that flattens and restores the expected tensor shapes.

## 5.3 Causal Transformers

For GPT-like causal models, causal masks and rotary position embeddings must remain intact. The replacement module verifies that the mask is applied before the attention combination and that the positional encoding remains consistent with the original model.

## 5.4 Weight Copy Procedure

We copy the original $W_Q$, $W_K$, $W_V$, and $W_O$ weights into the replacement module when possible, then initialize the residual adapters to zero. This keeps the module close to the pretrained behavior at initialization and ensures a stable recovery process.
