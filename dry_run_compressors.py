import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

from framework.core.registry import COMPRESSOR_REGISTRY, ATTENTION_REGISTRY

# 1. Load Dummy Model
print("Loading dummy DistilBERT model...")
config = AutoConfig.from_pretrained("models/distilbert_sst2", num_labels=2)
model = AutoModelForSequenceClassification.from_config(config)

compressors_to_test = [
    ("kmeans", "COMPRESSOR", {"kmeans_clusters": 256, "kmeans_lora_r": 16}),
    ("hybrid_svd_layer", "COMPRESSOR", {"hybrid_svd_rank": 128, "hybrid_svd_lora_r": 16}),
    ("come", "COMPRESSOR", {"come_sparsity": 0.3}),
    ("sdrm", "COMPRESSOR", {"total_param_ratio": 0.7, "lora_r": 16}),
    ("a3", "COMPRESSOR", {"a3_max_error": 0.5}),
    ("adaptive_svd_attention", "ATTENTION", {})
]

for name, registry_type, kwargs in compressors_to_test:
    print(f"\n--- Testing {name} ({registry_type}) ---")
    try:
        if registry_type == "COMPRESSOR":
            factory = COMPRESSOR_REGISTRY.get(name) or COMPRESSOR_REGISTRY.get(name + "_factory")
            if factory is None:
                print(f"FAILED: {name} not found in COMPRESSOR_REGISTRY")
                continue
            
            compressor_or_model = factory(model=model, **kwargs)
            # If it's a compressor class instance
            if hasattr(compressor_or_model, "compress"):
                compressed_model, stats = compressor_or_model.compress()
            else:
                # Some factories return tuple (model, stats)
                if isinstance(compressor_or_model, tuple):
                    compressed_model, stats = compressor_or_model
                else:
                    compressed_model = compressor_or_model
            print(f"SUCCESS: {name} compressed model successfully.")
        else:
            # Attention replacement
            factory = ATTENTION_REGISTRY.get(name) or ATTENTION_REGISTRY.get(name + "_inject")
            if factory is None:
                print(f"FAILED: {name} not found in ATTENTION_REGISTRY")
                continue
            
            if hasattr(factory, "__code__"):
                injected_model = factory(model)
                print(f"SUCCESS: {name} injected successfully.")
            else:
                print(f"SUCCESS: {name} class loaded successfully: {factory}")
    except Exception as e:
        print(f"FAILED: {name} crashed with exception: {e}")
