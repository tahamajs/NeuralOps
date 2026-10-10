import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from neuralops.core.registry import ATTENTION_REGISTRY
import copy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    teacher.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    text = "The quick brown fox jumps over the lazy dog. AI is transforming the world of software engineering."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        out_base = teacher(input_ids, labels=input_ids)
        loss_base = out_base.loss.item()
        
    print(f"Teacher Loss: {loss_base:.4f}")
    
    failing = []
    passing = []
    
    from neuralops.core.registry import _DEPRECATED_ALIASES
    keys = [k for k in ATTENTION_REGISTRY.keys() if k not in getattr(ATTENTION_REGISTRY, "aliases", {}) and k not in _DEPRECATED_ALIASES.get("attention", {})]
    
    for key in keys:
        try:
            student = copy.deepcopy(teacher)
            attn_cls = ATTENTION_REGISTRY.get(key)
            
            # The pretrained model has hidden_size=768, num_attention_heads=12
            from neuralops.tests.testing_utils import MockConfig
            mock_cfg = MockConfig(hidden_size=768, num_attention_heads=12, is_decoder=True)
            attn_layer = attn_cls(config=mock_cfg)
            
            student.transformer.h[0].attn = attn_layer.to(device)
            
            with torch.no_grad():
                out = student(input_ids, labels=input_ids)
                loss = out.loss.item()
                
            delta = abs(loss - loss_base)
            if delta > 1.0:
                print(f"PASS: {key} (Delta: {delta:.4f})")
                passing.append(key)
            else:
                print(f"FAIL (Zero Delta): {key} (Delta: {delta:.4f})")
                failing.append(key)
        except Exception as e:
            msg = str(e).split("\n")[0][:100]
            print(f"FAIL (Exception): {key} -> {msg}")
            failing.append(key)
            
    print(f"\nTotal PASS: {len(passing)}")
    print(f"Total FAIL: {len(failing)}")
    
    with open("TEST_RESULTS.txt", "w") as f:
        f.write("FAILING MODULES:\n")
        for k in failing:
            f.write(f"{k}\n")
        
if __name__ == "__main__":
    main()
