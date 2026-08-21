import sys, os, torch, torch.nn as nn, traceback
sys.path.insert(0, os.path.abspath('Codes'))
sys.path.insert(0, os.path.abspath('Codes/framework'))
os.environ['HF_HUB_OFFLINE'] = '1'

from framework.attentions import ATTENTION_REGISTRY
from framework.utils.replace_attention import _try_instantiate

class FakeConfig:
    hidden_size = 768; num_attention_heads = 12; num_hidden_layers = 6
    intermediate_size = 3072; max_position_embeddings = 512
    hidden_dropout_prob = 0.1; attention_probs_dropout_prob = 0.1; dropout = 0.1
    layer_norm_eps = 1e-12; is_decoder = False

class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768); self.key = nn.Linear(768, 768); self.value = nn.Linear(768, 768)
        self.q_lin = self.query; self.k_lin = self.key; self.v_lin = self.value
        self.out_proj = nn.Linear(768, 768); self.out_lin = self.out_proj
        self.num_attention_heads = 12; self.attention_head_size = 64; self.all_head_size = 768
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (hidden_states, None)

old_attn = FakeAttn()
cfg = FakeConfig()
results = []
os.makedirs('Docs', exist_ok=True)

with open('Docs/forward_pass_verification.md', 'w', encoding='utf-8') as f:
    f.write("# Attention Module Forward Pass Verification\n\n")
    f.write("Testing the forward pass of all 126 modules with shape (2, 128, 768).\n\n")
    
    for key in sorted(ATTENTION_REGISTRY.keys()):
        fn = ATTENTION_REGISTRY.get(key)
        try:
            mod = _try_instantiate(fn, cfg, 0, 6, old_attn)
            if not isinstance(mod, nn.Module):
                raise ValueError("Not an nn.Module")
            
            # Move to CPU for safety
            mod = mod.to("cpu")
            x = torch.randn(2, 128, 768)
            mask = torch.ones(2, 128)
            
            # Universal forward attempt
            try:
                out = mod(x, attention_mask=mask)
            except TypeError:
                try:
                    out = mod(x, mask=mask)
                except TypeError:
                    try:
                        out = mod(query=x, key=x, value=x, mask=mask)
                    except TypeError:
                        out = mod(x)
            
            f.write(f"### [OK] `{key}`\n- **Status**: SUCCESS\n- **Output Type**: `{type(out)}`\n\n")
            print(f"[OK] {key}")
        except Exception as e:
            err = str(e).split('\n')[0][:120]
            f.write(f"### [FAIL] `{key}`\n- **Status**: FAILED\n- **Error**: `{err}`\n\n")
            print(f"[ERR] {key}: {err}")

print("Verification complete. Results saved to Docs/forward_pass_verification.md")
