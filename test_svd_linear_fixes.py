import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_linear_attention import SVDLinearAttention, inject_svd_linear_attention
from transformers import AutoConfig, AutoModelForSequenceClassification

print("Checking registry...")
assert "svd_linear_inject" in ATTENTION_REGISTRY
assert "inject_svd_linear_attention" in ATTENTION_REGISTRY
# Make sure class is what's under svd_linear_attention
cls = ATTENTION_REGISTRY.get("svd_linear_attention")
assert isinstance(cls, type) and issubclass(cls, nn.Module), f"Expected class, got {type(cls)}"
print("Registry OK!")
