import sys
import os

# Add Codes directory to python path
sys.path.insert(0, r"D:\SVD_Project\Codes")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model

failed_modules = [
    "adasplash", "aegis_spectron_fp8", "bam_umoe", "decay_linear_attn", 
    "granite_hybrid", "hamming", "hashed_net", "hgrn2_layer", "hgrn2_gated_deltanet"
]

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(r"D:\SVD_Project\models\distilbert_sst2")
model_path = r"D:\SVD_Project\models\distilbert_sst2"

dummy_input = tokenizer("This is a test sentence.", return_tensors="pt")

for mod in failed_modules:
    print(f"\n==================== Testing: {mod} ====================")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to("cuda:0")
        model = apply_module_to_model(model, mod, {"dataset": "sst2"})
        inp = {k: v.to("cuda:0") for k, v in dummy_input.items()}
        out = model(**inp)
        loss = out.logits.sum()
        loss.backward()
        print(f"--> SUCCESS on forward + backward! (Slow execution / exceeded 900s training timeout)")
    except Exception as e:
        print(f"--> FAILED with exception: {type(e).__name__}: {e}")
