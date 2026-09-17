import sys
from pathlib import Path

cwd = Path.cwd()
if str(cwd / "Codes") not in sys.path:
    sys.path.insert(0, str(cwd / "Codes"))

import torch
from transformers import AutoModelForSequenceClassification
import framework.attentions
import framework.compression_engine
from framework.core.attention_utils import inject_attention

def test():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    print(f"Original layers: {len(model.distilbert.transformer.layer)}")
    
    # Try injecting asvd
    injected_model = inject_attention(model, "asvd", rank=32)
    
    # Count how many ASVDAttention layers are there
    count = 0
    for name, module in injected_model.named_modules():
        if "ASVDAttention" in str(type(module)):
            count += 1
            
    print(f"ASVDAttention layers found: {count}")
    
if __name__ == "__main__":
    test()
