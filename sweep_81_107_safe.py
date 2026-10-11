import os, sys, json, traceback, gc
import torch, transformers
from types import SimpleNamespace
from datasets import Dataset
from neuralops.benchmarks._legacy.kd_quick_test import apply_module_to_model, eval_ppl

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
eval_args = SimpleNamespace(device='cuda', seq_len=256, eval_stride=128, test_split='test', eval_max_tokens=500)
eval_dataset = Dataset.from_dict({'text': [
    'The quick brown fox jumps over the lazy dog. Scientific reasoning requires rigorous mathematical proof and empirical testing.',
    'Machine learning models, particularly autoregressive transformers, rely on scaled dot-product self-attention mechanisms to capture long-range token dependencies.',
    'In physics, quantum electrodynamics describes how light and matter interact with remarkable precision across microscopic spatial scales.',
    'Adaptive attention methods modify sequence representation through dynamic spans, token pruning, and head squeezing.'
]})

base_ppl = 101.15

mods = [
    'span_expanded_attention', 'spectra_v2', 'spectraformer_attention', 'spectral_mamba',
    'spectron', 'spectron_fp8', 'split_head_hybrid', 'squeezing_heads_distill',
    'svd_compressed', 'svd_linear_attention', 'svd_linear_decay', 'svd_llm_v2_attention',
    'svd_mamba', 'svd_projection_attention', 'tlinformer_attention',
    'token_compressed_attention', 'tpa_attention', 'tri_branch_hybrid',
    'triton_cosformer', 'triton_svd_linear_attention', 'unified_rope',
    'upi_stuffed_mamba', 'vig_attention', 'vspec', 'wgqa_attention',
    'xattention', 'zeros_attention'
]

results = {}
if os.path.exists("sweep_81_107_results.json"):
    try:
        with open("sweep_81_107_results.json", "r") as f:
            results = json.load(f)
    except:
        results = {}

for idx, m_name in enumerate(mods, 81):
    if m_name in results:
        res = results[m_name]
        print(f"[{idx:3d}/107] {m_name:30s} | ALREADY_EVALUATED: {res.get('status')} | DELTA={res.get('delta', 'N/A')}", flush=True)
        continue
    
    print(f"\n[{idx:3d}/107] Evaluating {m_name}...", flush=True)
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
        s, _ = apply_module_to_model(m, m_name, 'auto', rank=32, device='cuda')
        s = s.half().to('cuda').eval()
        p = eval_ppl(s, tokenizer, eval_dataset, eval_args)
        delta = p - base_ppl
        status = 'PASS' if delta >= 1.0 else 'FAIL_D0'
        results[m_name] = {
            'index': idx,
            'ppl': round(float(p), 2),
            'delta': round(float(delta), 2),
            'status': status
        }
        print(f"[{idx:3d}/107] {m_name:30s} | PPL={p:8.2f} | DELTA={delta:+8.2f} | {status}", flush=True)
    except Exception as e:
        err_msg = str(e)
        results[m_name] = {
            'index': idx,
            'error': err_msg,
            'status': 'ERROR'
        }
        print(f"[{idx:3d}/107] {m_name:30s} | ERROR: {err_msg}", flush=True)
        traceback.print_exc()
    finally:
        try:
            del s
        except:
            pass
        try:
            del m
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        with open("sweep_81_107_results.json", "w") as f:
            json.dump(results, f, indent=2)

print("\nCOMPLETED ALL MODULES 81-107!", flush=True)
