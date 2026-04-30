#!/usr/bin/env python
# examples/soa_demo.py
from framework import quick_soa_model, quick_inference
from transformers import AutoTokenizer

# Create SOA model
model = quick_soa_model("bert-base-uncased", rank=64, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Test inference
text = "This movie is absolutely fantastic!"
result = quick_inference(model, text, tokenizer)
print(f"Logits: {result.logits}, Prediction: {result.logits.argmax().item()}")