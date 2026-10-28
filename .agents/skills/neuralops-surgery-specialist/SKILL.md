---
name: neuralops-surgery-specialist
description: "Perform in-place neural attention transplant and post-surgery calibration healing."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "surgery", "svd", "calibration", "healing"]
---

# Skill: Neural Surgery Specialist

## When to use
Trigger when performing in-place replacement of standard transformer attention layers with low-rank or linear SVD-factorized counterparts.

## Prerequisites

## Steps
1. **Verify Surgery Compatibility**:
   Ensure target backbone architecture and candidate attention mechanism support `replace_attention()`:
   ```python
   from neuralops import replace_attention
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("gpt2")
   replace_attention(model, attention_type="svd_linear_attention")
   ```
2. **Execute Closed-Form Ridge Calibration**:
   Solve for projection matrices using ridge regression against calibration token activations to minimize output divergence without gradient descent.
3. **Apply Post-Transplant Healing**:
   Execute distillation healing for 30 optimization steps using `heal_replaced_model()`:
   ```python
   from neuralops.surgery.healing import heal_replaced_model

   heal_replaced_model(model, dataloader=calibration_loader, steps=30, lr=1e-5)
   ```
4. **Verify Output Divergence**:
   Ensure post-transplant numerical divergence is strictly controlled:
   `max_diff < 1e-3` across hidden states on validation sequences.

## Verification
- Run sanity forward pass and check cosine similarity $\ge 0.98$ against the uncompressed original model.

## Common pitfalls
- Skipping the 30-step healing phase leading to high token perplexity.
- Using uncalibrated random projection initializations.
