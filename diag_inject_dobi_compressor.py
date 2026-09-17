import sys
from pathlib import Path

cwd = Path.cwd()
if str(cwd / "Codes") not in sys.path:
    sys.path.insert(0, str(cwd / "Codes"))

from transformers import AutoModelForSequenceClassification
import framework.attentions
import framework.compression_engine
from framework.eval_all_models_datasets import apply_module_to_model

def test():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    print(f"Original params: {sum(p.numel() for p in model.parameters())}")
    
    injected_model, ratio = apply_module_to_model(model, "dobi", "compressor", rank=32)
    
    print(f"Ratio: {ratio}")
    print(f"After params: {sum(p.numel() for p in injected_model.parameters())}")
    if hasattr(injected_model, "compression_stats"):
        print(f"Compression stats: {injected_model.compression_stats}")
    else:
        print("No compression stats found")
    
if __name__ == "__main__":
    test()
