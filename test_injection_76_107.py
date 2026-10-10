import torch
import sys
from transformers import GPT2Model
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.attention_utils import inject_attention
import neuralops.attentions

modules = [
    'samba_m3', 'samba_moe', 'selective_ssm', 'sink_attention', 'soa_attn',
    'span_expanded_attention', 'spectra_v2', 'spectraformer_attention', 'spectral_mamba',
    'spectron', 'spectron_fp8', 'split_head_hybrid', 'squeezing_heads_distill',
    'svd_compressed', 'svd_linear_attention', 'svd_linear_decay', 'svd_llm_v2_attention',
    'svd_mamba', 'svd_projection_attention', 'tlinformer_attention',
    'token_compressed_attention', 'tpa_attention', 'tri_branch_hybrid',
    'triton_cosformer', 'triton_svd_linear_attention', 'unified_rope',
    'upi_stuffed_mamba', 'vig_attention', 'vspec', 'wgqa_attention',
    'xattention', 'zeros_attention'
]

print(f"STARTING INJECTION BATCH OF {len(modules)} MODULES", flush=True)
for idx, mod in enumerate(modules, 76):
    try:
        m = GPT2Model.from_pretrained('gpt2').eval()
        cls = ATTENTION_REGISTRY.get(mod)
        if cls is None:
            print(f"[{idx}] {mod}: NOT_FOUND_IN_REGISTRY", flush=True)
            continue
        # In neuralops, inject_attention takes attention_type as string key
        ok_model = inject_attention(m, mod, rank=32, copy_weights=True)
        if ok_model is None:
            print(f"[{idx}] {mod}: INJECTION_FAILED", flush=True)
            continue
        x = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            out = ok_model(x).last_hidden_state
        finite = torch.isfinite(out).all().item()
        print(f"[{idx}] {mod}: OK shape={tuple(out.shape)} finite={finite}", flush=True)
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e).split('\n')[0][:80]
        print(f"[{idx}] {mod}: EXCEPTION {err_type}: {err_msg}", flush=True)
