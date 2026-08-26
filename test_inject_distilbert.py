import os
os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir/"
import sys
from pathlib import Path
sys.path.insert(0, "Codes")

from transformers import AutoModelForSequenceClassification
from framework.eval_all_models_datasets import apply_module_to_model

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("Injecting svd_linear_attention...")
model, comp_ratio = apply_module_to_model(
    model=model,
    module_name="svd_linear_attention",
    module_type="attention",
    rank=16,
)
print("SUCCESS: Injection completed! Compression ratio:", comp_ratio)
