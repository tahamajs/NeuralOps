import sys
from pathlib import Path

cwd = Path.cwd()
if str(cwd / "Codes") not in sys.path:
    sys.path.insert(0, str(cwd / "Codes"))

from transformers import AutoModelForSequenceClassification
import framework.attentions
import framework.compression_engine
from framework.core.attention_utils import inject_attention

def test():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    print(f"Original layers: {len(model.distilbert.transformer.layer)}")
    
    injected_model = inject_attention(model, "dobi", rank=32)
    
    count = 0
    for name, module in injected_model.named_modules():
        if "Dobi" in str(type(module)) or "dobi" in name.lower():
            count += 1
            
    print(f"Dobi layers found: {count}")
    
if __name__ == "__main__":
    test()
