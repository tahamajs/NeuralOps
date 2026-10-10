import torch
from transformers import GPT2LMHeadModel, GPT2Config
from neuralops.core.registry import ATTENTION_REGISTRY
import sys
import copy
import traceback

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=4
    )
    
    teacher = GPT2LMHeadModel(config).to(device)
    teacher.eval()
    
    # Generate random batch
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (2, 128)).to(device)
    
    with torch.no_grad():
        out_base = teacher(input_ids, labels=input_ids)
        loss_base = out_base.loss.item()
        
    print(f"Teacher Loss: {loss_base:.4f}")
    
    failing = []
    passing = []
    
    # Exclude aliases just in case
    # Wait, dedup removed all aliases, but deprecated_aliases.json still maps them in the registry dynamically.
    # The true registered keys are only the canonical ones!
    # Wait, if ATTENTION_REGISTRY.keys() returns aliases, we should filter.
    from neuralops.core.registry import _DEPRECATED_ALIASES
    keys = [k for k in ATTENTION_REGISTRY.keys() if k not in getattr(ATTENTION_REGISTRY, "aliases", {}) and k not in _DEPRECATED_ALIASES.get("attention", {})]
    
    for key in keys:
        try:
            student = copy.deepcopy(teacher)
            attn_cls = ATTENTION_REGISTRY.get(key)
            
            from neuralops.tests.testing_utils import MockConfig
            mock_cfg = MockConfig(hidden_size=64, num_attention_heads=4, is_decoder=True)
            attn_layer = attn_cls(config=mock_cfg)
            
            # The original GPT2Attention has c_attn, c_proj. We just overwrite it completely.
            # But the forward signature might differ. 
            # GPT2 passes (hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions)
            # We will use a wrapper if it doesn't match, or just rely on the module supporting `kwargs`.
            
            student.transformer.h[0].attn = attn_layer.to(device)
            
            with torch.no_grad():
                out = student(input_ids, labels=input_ids)
                loss = out.loss.item()
                
            delta = abs(loss - loss_base)
            # Delta > 1.0 is PASS. Delta <= 1.0 is FAIL.
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
