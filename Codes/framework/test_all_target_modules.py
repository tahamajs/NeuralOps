import sys
sys.path.insert(0, '/Users/tahamajs/Documents/uni/Project/Codes')
import torch
import torch.nn as nn
from transformers import DistilBertConfig
from framework.core.registry import ATTENTION_REGISTRY

# Ensure all attention modules are imported
import framework.attentions

config = DistilBertConfig(dim=768, n_heads=12, hidden_dim=3072, n_layers=6)
config.hidden_size = 768
config.num_attention_heads = 12
config.head_dim = 64
config.dropout = 0.1

class MockOldAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)

old_attn = MockOldAttn()

target_keys = [
    'bam_umoe',
    'expert_choice_moe',
    'moe_deltakv_paged',
    'reap_moe',
    'samba_moe',
    'token_compressed_attention',
    'token_compressed',
    'compressed_attention',
    'compressed_mha_attention',
    'nha_attention',
    'nha_compressor',
    'block_transformer_hiba',
    'jamba',
    'jamba_info',
    'hymba_transx',
    'infini_attention',
    'fusion_attn',
    'nexus',
    'leka_v1',
    'leka_v2',
    'leka_v3',
    'mixattention_moska',
    'vspec_attention',
    'vspec',
    'wgqa_attention',
    'progressive_token_pruning',
    'chela_slay',
    'kda_attention',
    'moba_attention',
    'nasc_attention',
    'orion'
]

print('Testing ' + str(len(target_keys)) + ' target modules...', flush=True)
success = []
failed = []

for key in target_keys:
    try:
        cls = ATTENTION_REGISTRY.resolve(key)
        try:
            mod = cls(config=config, pretrained_attn=old_attn)
        except Exception:
            try:
                mod = cls(config)
            except Exception:
                try:
                    mod = cls(pretrained_attn=old_attn)
                except Exception:
                    mod = cls()

        if hasattr(mod, 'copy_weights'):
            mod.copy_weights(old_attn)

        mod.train()

        x = torch.randn(2, 128, 768, requires_grad=True)
        mask = torch.ones(2, 128)
        mask[0, 100:] = 0

        out = mod(x, attention_mask=mask)
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out

        assert out_tensor.shape == (2, 128, 768), 'Shape mismatch: ' + str(out_tensor.shape)
        assert not torch.isnan(out_tensor).any(), 'NaN in output'
        assert not torch.isinf(out_tensor).any(), 'Inf in output'

        loss = out_tensor.sum()
        if hasattr(mod, 'aux_loss') and isinstance(mod.aux_loss, torch.Tensor):
            loss = loss + mod.aux_loss

        loss.backward()
        assert x.grad is not None, 'No gradient on input'
        assert not torch.isnan(x.grad).any(), 'NaN in input grad'

        has_param_grad = False
        for p in mod.parameters():
            if p.grad is not None and not torch.isnan(p.grad).any():
                has_param_grad = True
                break

        print('  [PASS] ' + key.ljust(30) + ' -> shape=' + str(tuple(out_tensor.shape)) + ', grad_ok=' + str(has_param_grad), flush=True)
        success.append(key)
    except Exception as e:
        print('  [FAIL] ' + key.ljust(30) + ' -> ERROR: ' + str(e), flush=True)
        failed.append((key, str(e)))

print('\nFinal Result: ' + str(len(success)) + ' passed, ' + str(len(failed)) + ' failed.', flush=True)
if failed:
    sys.exit(1)
