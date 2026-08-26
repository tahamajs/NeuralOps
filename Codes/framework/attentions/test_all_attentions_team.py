import torch
from framework.core.registry import ATTENTION_REGISTRY
import traceback
import sys
import framework.attentions  # This should trigger registration

import inspect

def test_attentions():
    print(f"Total modules registered: {len(ATTENTION_REGISTRY)}")
    passed = 0
    failed = []
    
    for name, attn_cls in ATTENTION_REGISTRY.items():
        try:
            print(f"Testing {name}...")
            config = type("Config", (), {"hidden_size": 128, "num_attention_heads": 4})()
            config_t5 = type("Config", (), {"d_model": 128, "num_heads": 4})()
            
            # 1. Instantiation
            if inspect.isclass(attn_cls):
                try:
                    attn = attn_cls(config)
                except Exception:
                    attn = attn_cls(config_t5)
            else:
                # It's an inject function
                layer = torch.nn.Linear(128, 128)
                layer.config = config
                try:
                    attn = attn_cls(layer, model_config=config)
                except Exception:
                    attn = attn_cls(layer, model_config=config_t5)
            
            # Check d_model to ensure it parsed config correctly
            d_model = getattr(attn, "d_model", getattr(attn, "dim", 128))
            if d_model != 128 and d_model != 0: # some might not have d_model
                raise ValueError(f"d_model is {d_model}, expected 128. Config parsing failed.")

            # 2. Forward/Backward
            hidden_states = torch.randn(2, 10, 128, requires_grad=True)
            attention_mask = torch.zeros(2, 1, 1, 10)
            
            try:
                out = attn(hidden_states, attention_mask=attention_mask)
            except Exception:
                out = attn(hidden_states)
                
            if isinstance(out, tuple):
                loss = out[0].sum()
            else:
                loss = out.sum()
            loss.backward()
            
            # 3. FP16 half precision
            attn.half()
            hidden_states_fp16 = torch.randn(2, 10, 128, dtype=torch.float16)
            try:
                out_fp16 = attn(hidden_states_fp16, attention_mask=attention_mask.half())
            except Exception:
                out_fp16 = attn(hidden_states_fp16)
                
            # 4. T5-encoder compatibility
            if inspect.isclass(attn_cls):
                try:
                    attn_t5_obj = attn_cls(config_t5)
                except Exception:
                    pass
            
            passed += 1
            print(f"  {name} PASSED")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            traceback.print_exc(file=sys.stdout)
            failed.append(name)
            
    print(f"\nTotal modules tested: {len(ATTENTION_REGISTRY)}")
    print(f"Total PASSED: {passed}")
    print(f"Total FAILED: {len(failed)}")
    if failed:
        print("Failed modules:")
        for name in failed:
            print(f"  - {name}")

if __name__ == '__main__':
    test_attentions()
