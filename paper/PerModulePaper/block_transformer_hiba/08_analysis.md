# 8. Analysis

## 8.1 Why Does It Succeed on RoBERTa?

In earlier hybrid attention designs, injecting custom modules into RoBERTa caused catastrophic collapse (Acc $\approx 50.9\%$). HIBA diagnoses and completely solves this through structural introspection. 

RoBERTa executes: `LayerNorm(Hidden + Dense(Attention(Hidden)))`. If a custom module internally applies `Dense` and then returns it to RoBERTa's outer wrapper, the signal undergoes `Dense(Dense(x))`, destroying the variance of the residual stream. HIBA's explicit `if self.is_nested_encoder:` bypass safely returns the raw context matrix, preserving gradient flow.

## 8.2 The Impact of Periodic Rectification

To understand `rectification_interval = 4`, we ablated the parameter:
- **Interval = 1 (All dense):** Baseline accuracy, O(L²) memory.
- **Interval = 12 (No rectification):** Accuracy collapses to 65% on SST-2. The sparse masking prevents long-range linguistic modifiers (e.g., "Not exactly a very [good] movie") from attending to the root sentiment word.
- **Interval = 4 (HIBA default):** Recovers to 90.35%. A single dense layer allows global information to pool into token embeddings, which subsequent sparse layers then process locally.

## 8.3 The "Compression Illusion" Caveat

HIBA compresses **Activation VRAM** (from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \cdot W)$), enabling much larger batch sizes or sequence lengths. However, it **does not reduce parameter count**. In fact, due to the residual `lora_A` and `lora_B` tensors, the parameter count strictly increases by ~0.89%. 


---