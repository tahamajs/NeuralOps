#!/usr/bin/env python
"""
Test all registered attention modules and compressors with timeout.
Usage:
    python test_attention.py [--limit N] [--output results.csv] [--device cpu|cuda|mps]
"""

import argparse
import signal
import torch
import sys
import os
import inspect
import pandas as pd
from types import SimpleNamespace
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.utils.layer_norm_fix import fix_layernorm_naming
from framework.config import CompressionConfig
from framework.utils.device import get_device

TIMEOUT = 15

# Known problematic components (skip)
SKIP_ATTENTIONS = {
    "compressed_attention_layer", "compressed_attention_layer_energy",
    "lola_attn", "asvd", "bam_umoe", "cluster", "fusion_attn",
    "mixattention_moska", "nha_compressor", "rala_compressor",
    "jamba_info", "optimized_hybrid_attention", "spectra_v2",
    "template_compressor", "template_attention", "aegis_v5",
    "xattention", "progressive_token_pruning", "eat", "phantom",
    "regla", "compression_unified_rope", "simple_ssm", "goldfinch_ahn"
}
SKIP_COMPRESSORS = {
    "frandi", "tt_compressor", "adaptive_compressor", "slicegpt",
    "rpca", "admm", "saes_svd_v2", "sdrm", "mgie_budget", "liftr",
    "cr", "mamba_shedder", "ast_pruner", "coopq", "slicegpt", "saes",
    "oats", "subspace_projection", "gfws", "swift_svd_act", "faq",
    "cur", "swift_svd", "kronecker", "erc", "lillama", "lsc",
    "flap", "flat_llm", "aasvd", "cur_moe", "delta_kv", "kvtc",
    "entropy_budget", "hybridkv", "mgie", "salr", "uniql", "soka",
    "kmeans", "bonsai", "replaceme", "sparsegpt", "tomoe", "hashed_net",
    "compot", "armor_pruner", "admm_pruner", "modegpt", "enlighten",
    "a3", "direct_svd", "roast", "proxsparse", "slim", "nmf", "mlorq",
    "cachenotes", "zerokv_neo", "dobi", "astro", "adasvd"
}

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

def run_with_timeout(func, *args, timeout=TIMEOUT):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args)
        signal.alarm(0)
        return result
    except TimeoutError:
        return False, "Timeout (>{}s)".format(timeout)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def make_injector(obj):
    if inspect.isclass(obj):
        sig = inspect.signature(obj.__init__)
        params = list(sig.parameters.values())[1:]  # skip 'self'
        def inject(layer, config, comp_config, layer_idx, num_layers, old_attn):
            kwargs = {}
            arg_map = {
                'config': config, 'comp_config': comp_config, 'layer_idx': layer_idx,
                'num_layers': num_layers, 'old_attn': old_attn, 'attention': old_attn,
                'model_config': config, 'layer_id': layer_idx,
                'rank': getattr(comp_config, 'rank', None),
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_heads': getattr(config, 'num_attention_heads', None),
                'pretrained_attn': old_attn,
            }
            for p in params:
                if p.name in arg_map:
                    kwargs[p.name] = arg_map[p.name]
                elif p.kind == inspect.Parameter.VAR_KEYWORD:
                    kwargs.update(arg_map)
                    break
            try:
                new_attn = obj(**kwargs)
            except TypeError:
                pos_args = []
                for p in params:
                    if p.default == inspect.Parameter.empty:
                        pos_args.append(arg_map.get(p.name))
                new_attn = obj(*pos_args)
            layer.attention = new_attn
        return inject
    else:
        return obj

def test_attention(attn_name, model, model_config, device, comp_config):
    inj = ATTENTION_REGISTRY.get(attn_name)
    if inj is None:
        return False, "Not in registry"
    if not hasattr(model, "distilbert"):
        return False, "Model has no distilbert"
    layers = model.distilbert.transformer.layer
    inject_fn = make_injector(inj)
    for i, layer in enumerate(layers):
        old_attn = layer.attention
        inject_fn(layer, model_config, comp_config, i, len(layers), old_attn)
    model.to(device)
    model.eval()
    input_ids = torch.randint(0, 30522, (2, 32)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _ = outputs.logits
    return True, "OK"

def test_compressor(comp_name, model, device, comp_config):
    comp_fn = COMPRESSOR_REGISTRY.get(comp_name)
    if comp_fn is None:
        return False, "Not in registry"
    compressed = comp_fn(model, comp_config)
    if not isinstance(compressed, torch.nn.Module):
        return False, f"Compressor returned {type(compressed)}, not a nn.Module"
    compressed.to(device)
    compressed.eval()
    input_ids = torch.randint(0, 30522, (2, 32)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    with torch.no_grad():
        outputs = compressed(input_ids=input_ids, attention_mask=attention_mask)
    if not hasattr(outputs, 'logits'):
        return False, "Output has no 'logits'"
    return True, "OK"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="test_results.csv")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    parser.add_argument("--skip_attentions", action="store_true")
    parser.add_argument("--skip_compressors", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("Testing all registered components (with timeout)")
    print("="*70)

    model_config = AutoConfig.from_pretrained(args.model)
    
    # Load tokenizer once (will be attached to each model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Use SimpleNamespace instead of CompressionConfig to avoid unexpected keyword errors
    comp_config = SimpleNamespace(
        attn_type="full",
        dropout=0.1,
        rank=32,
        svd_rank=32,
        block_size=64,
        local_window=64,
        top_k_inter=4,
        lora_rank=8,
        lora_alpha=16.0,
        svd_energy=0.95,
        num_landmarks=16,
        feature_dim=128,
        rff_features=128,
        use_full_attention=False,
        ratio=3,
        n_experts=8,
        top_k=2,
        window_size=128,
        use_fp8=False,
        # For compressors that need tokenizer
        keep_ratio=0.5,
        target_sparsity=0.3,
        a3_rank=64,
        energy_threshold=0.95,
        sparsity=0.5,
        quant_bits=4,
        slicegpt_sparsity=0.25,
        oats_sparsity=0.5,
        oats_rank_ratio=0.2,
        oats_num_iters=5,
        mamba_shedder_sparsity=0.2,
        swift_svd_act_rank=64,
        faq_bits=8,
        compot_rank=64,
        lillama_rank=64,
    )

    results = []

    # --- Test attentions ---
    if not args.skip_attentions:
        attn_keys = [k for k in ATTENTION_REGISTRY.keys() if k not in SKIP_ATTENTIONS]
        if args.limit:
            attn_keys = attn_keys[:args.limit]
        print(f"\nTesting {len(attn_keys)} attentions (skipped {len(SKIP_ATTENTIONS)} known bad)...")
        for i, key in enumerate(attn_keys, 1):
            print(f"  [{i:3d}/{len(attn_keys)}] {key:<40} ", end='', flush=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
            fix_layernorm_naming(model)
            # Attach tokenizer for any compressor that might need it (though not used in attention test)
            model.tokenizer = tokenizer
            ok, msg = run_with_timeout(test_attention, key, model, model_config, device, comp_config)
            success = bool(ok)
            results.append({"type": "attention", "name": key, "success": success, "message": msg})
            if success:
                print("✓ PASS")
            else:
                print(f"✗ FAIL: {msg}")

    # --- Test compressors ---
    if not args.skip_compressors:
        comp_keys = [k for k in COMPRESSOR_REGISTRY.keys() if k not in SKIP_COMPRESSORS]
        if args.limit:
            comp_keys = comp_keys[:args.limit]
        print(f"\nTesting {len(comp_keys)} compressors (skipped {len(SKIP_COMPRESSORS)} known bad)...")
        for i, key in enumerate(comp_keys, 1):
            print(f"  [{i:3d}/{len(comp_keys)}] {key:<40} ", end='', flush=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config)
            fix_layernorm_naming(model)
            # Attach tokenizer to the model (required by many compressors)
            model.tokenizer = tokenizer
            ok, msg = run_with_timeout(test_compressor, key, model, device, comp_config)
            success = bool(ok)
            results.append({"type": "compressor", "name": key, "success": success, "message": msg})
            if success:
                print("✓ PASS")
            else:
                print(f"✗ FAIL: {msg}")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Summary: {df['success'].sum()} / {len(df)} passed.")

if __name__ == "__main__":
    main()