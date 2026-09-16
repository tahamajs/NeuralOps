# Appendix D. Negative Results and Hardened Engineering

One of the primary contributions of HIBA is its production-hardened engineering, born directly from extensive failure analysis. We document these negative results to aid future researchers.

## D.1 The RoBERTa Collapse (Nested Encoder Attenuation)

**Symptom:** When injecting early versions of block-sparse attention into RoBERTa, the model accuracy catastrophically collapsed to $\approx 50.92\%$ (random chance on SST-2), while DistilBERT maintained $>90\%$.

**Investigation:** 
We traced the gradient flow and discovered a fundamental structural difference in HuggingFace's implementation of the BERT family. DistilBERT uses a "flat" architecture where `out_lin` is inside the `MultiHeadSelfAttention` block. 
RoBERTa uses a "nested" architecture:
```python
class RobertaAttention:
    self: RobertaSelfAttention      # Contains Q, K, V
    output: RobertaSelfOutput       # Contains Dense (Output Projection)

```

**Why it failed:**
Our initial module applied its own `out_proj` and returned the result to `RobertaAttention`. RoBERTa then passed this output into `RobertaSelfOutput.dense`. This resulted in the signal passing through two consecutive dense layers without a non-linearity: `Dense(Dense(x))`. This double-projection drastically altered the variance of the residual stream, causing immediate gradient explosion/vanishing and representation collapse.

**The Fix:**
We introduced explicit introspection (`self.is_nested_encoder`) during `copy_weights`. If a nested architecture is detected, HIBA extracts the inner $Q, K, V$ weights, but **bypasses its own `out_proj` during the forward pass**, returning the raw context matrix to be safely processed by RoBERTa's native outer wrapper. This completely resolved the collapse, bringing RoBERTa's accuracy to 78.40% (Δ = -0.16).

## D.2 The Qwen/LLaMA RoPE Mismatch

**Symptom:** Injecting HIBA into Qwen3.5 resulted in NaNs and severe language modeling degradation.

**Investigation:**
Standard implementations of Rotary Position Embeddings (RoPE) assume that the entire head dimension ($d_h$) is rotated. However, modern models often apply partial rotations to save compute. For instance, Qwen3.5 uses $d_h = 256$ but only rotates the first `rotary_dim = 64` coordinates.

**The Fix:**
We rewrote the `apply_rotary_pos_emb` function to dynamically slice the query and key tensors:

```python
if rotary_dim < head_dim:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    # Rotate q_rot, then torch.cat with q_pass

```

This partial-dimension splicing restored numerical parity with the base Qwen3.5 forward pass up to $1e^{-5}$ tolerance.

## D.3 Failed Attempts at Pure Sparsity

Before implementing **Periodic Dense Rectification**, we attempted to train HIBA as a purely sparse model (`rectification_interval = infinity`). While loss decreased during training, the model failed to generalize on validation data involving complex negations (e.g., "Not exactly a very good movie"). The strict block boundary severed the connection between "Not" and "good". The introduction of the dense rectification layer acts as an "attentional highway," allowing these critical long-range linguistic modifiers to route properly.


---