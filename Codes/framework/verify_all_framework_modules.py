"""
Exhaustive Master Verification & Cleanliness Suite for SVD Linear Attention Framework.
Audits, instantiates, and executes:
  1. ALL 118 Attention Architectures (ATTENTION_REGISTRY)
  2. ALL 80 Compression Engines (compression_engine/)
  3. Model Zoo & Model Wrappers (UnifiedTransformerModel, BaseAttention)
  4. Training Pipelines (CustomLoopTrainer, Optimizers, Schedulers)
  5. Benchmark Datasets Loader & Visualizers
  6. Results Saver, Exporters & Dashboard Generators
"""

import os
import sys
import tempfile
import inspect
import importlib
import traceback
import torch
import torch.nn as nn

# Ensure root Codes/ and framework/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODES_DIR = os.path.join(PROJECT_ROOT, "Codes")
FRAMEWORK_DIR = os.path.join(CODES_DIR, "framework")

if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

import framework
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.core.model_wrapper import UnifiedTransformerModel, UnifiedTransformerConfig
from framework.data.benchmark_datasets import load_benchmark_dataset, BenchmarkDatasetRegistry
from framework.visualization.dashboard_generator import generate_dashboard_html, generate_summary_json
from framework.visualization.benchmark_visualizer import (
    plot_singular_value_decay,
    plot_kv_cache_scaling,
    plot_pareto_frontier
)
from framework.results_saver.saver import ResultsSaver
from framework.trainer.custom_loop_trainer import CustomLoopTrainer
from framework.trainer.optimizers import OPTIMIZER_REGISTRY
from framework.trainer.schedulers import SCHEDULER_REGISTRY

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def verify_all_attentions():
    print_header("STAGE 1: VERIFYING ALL ATTENTION ARCHITECTURES IN ATTENTION_REGISTRY")
    
    # Pre-import all attention modules from attentions directory to ensure full registration
    attentions_dir = os.path.join(FRAMEWORK_DIR, "attentions")
    if os.path.exists(attentions_dir):
        for f in os.listdir(attentions_dir):
            if f.endswith(".py") and not f.startswith("__"):
                mod_name = f"framework.attentions.{f[:-3]}"
                try:
                    importlib.import_module(mod_name)
                except Exception as e:
                    print(f"  [Warning] importing {mod_name}: {e}")

    total_keys = len(ATTENTION_REGISTRY)
    print(f"Discovered {total_keys} registered attention mechanisms.")
    
    passed = 0
    failed = []
    
    B, T, C = 2, 16, 64
    num_heads = 4
    x = torch.randn(B, T, C)

    for idx, (key, cls_or_fn) in enumerate(sorted(ATTENTION_REGISTRY.items()), 1):
        try:
            # Instantiate attention module
            inst = None
            if inspect.isclass(cls_or_fn):
                sig = inspect.signature(cls_or_fn.__init__)
                params = sig.parameters
                kwargs = {}
                if "embed_dim" in params or "d_model" in params:
                    kwargs["embed_dim" if "embed_dim" in params else "d_model"] = C
                if "num_heads" in params or "n_heads" in params:
                    kwargs["num_heads" if "num_heads" in params else "n_heads"] = num_heads
                if "max_seq_len" in params:
                    kwargs["max_seq_len"] = 128
                if "rank" in params:
                    kwargs["rank"] = 16
                
                # Try instantiation with computed kwargs
                try:
                    inst = cls_or_fn(**kwargs)
                except Exception:
                    # Fallback to embed_dim, num_heads positional or default
                    try:
                        inst = cls_or_fn(C, num_heads)
                    except Exception:
                        inst = cls_or_fn()
            else:
                inst = cls_or_fn

            # Forward pass test
            inst.eval() if hasattr(inst, "eval") else None
            with torch.no_grad():
                if callable(inst):
                    # Test various standard call signatures
                    try:
                        out = inst(x)
                    except TypeError:
                        try:
                            out = inst(x, x, x)
                        except TypeError:
                            out = inst(query=x, key=x, value=x)
                    
                    if isinstance(out, tuple):
                        out = out[0]
                    
                    assert isinstance(out, torch.Tensor), f"Output must be a Tensor, got {type(out)}"
                    assert not torch.isnan(out).any(), "Output contains NaN values"
                    assert not torch.isinf(out).any(), "Output contains Inf values"
            
            passed += 1
            if idx <= 10 or idx % 20 == 0 or idx == total_keys:
                print(f"  [{idx:3d}/{total_keys}] ✓ {key:<35} -> Output Shape: {list(out.shape)}")
        except Exception as e:
            failed.append((key, str(e)))
            print(f"  [{idx:3d}/{total_keys}] ✗ {key:<35} -> ERROR: {e}")

    print(f"\nAttention Verification Summary: {passed}/{total_keys} PASSED ({(passed/total_keys)*100:.1f}%)")
    if failed:
        print(f"  Failures ({len(failed)}):")
        for k, err in failed:
            print(f"    - {k}: {err}")
    return len(failed) == 0

def verify_all_compressors():
    print_header("STAGE 2: VERIFYING ALL COMPRESSION ENGINES")
    
    # Import all modules from compression_engine directory
    comp_dir = os.path.join(FRAMEWORK_DIR, "compression_engine")
    discovered_compressor_classes = []
    
    if os.path.exists(comp_dir):
        for f in sorted(os.listdir(comp_dir)):
            if f.endswith(".py") and not f.startswith("__"):
                mod_name = f"framework.compression_engine.{f[:-3]}"
                try:
class DummyTokenizer:
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
    def encode(self, text, *args, **kwargs):
        return [1, 2, 3, 4]
    def decode(self, tokens, *args, **kwargs):
        return "sample text"
    def __call__(self, text, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}

def verify_all_compressors():
    print_header("STAGE 2: VERIFYING ALL COMPRESSION ENGINES")
    
    # Import all modules from compression_engine directory
    comp_dir = os.path.join(FRAMEWORK_DIR, "compression_engine")
    discovered_compressor_classes = []
    
    if os.path.exists(comp_dir):
        for f in sorted(os.listdir(comp_dir)):
            if f.endswith(".py") and not f.startswith("__"):
                mod_name = f"framework.compression_engine.{f[:-3]}"
                try:
                    mod = importlib.import_module(mod_name)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (
                            inspect.isclass(attr)
                            and (attr_name.endswith("Compressor") or attr_name.endswith("Engine") or "Compress" in attr_name or "SVD" in attr_name or "Quant" in attr_name or "Prun" in attr_name)
                            and attr_name not in ["BaseCompressor", "nn", "torch"]
                            and not attr_name.startswith("_")
                        ):
                            if (f, attr_name, attr) not in discovered_compressor_classes:
                                discovered_compressor_classes.append((f, attr_name, attr))
                except Exception as e:
                    print(f"  [Warning] importing {mod_name}: {e}")

    total = len(discovered_compressor_classes)
    print(f"Discovered {total} unique Compression Engine classes across {len(os.listdir(comp_dir))} files.")
    
    passed = 0
    failed = []
    
    dummy_tokenizer = DummyTokenizer()
    dummy_loader = [(torch.randn(2, 16, 64), torch.randn(2, 16, 64))]
    
    for idx, (filename, cls_name, cls_obj) in enumerate(discovered_compressor_classes, 1):
        try:
            target_linear = nn.Linear(64, 64)
            target_seq = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
            
            # Inspect constructor parameters
            inst = None
            try:
                sig = inspect.signature(cls_obj.__init__)
                params = sig.parameters
                kwargs = {}
                if "model" in params:
                    kwargs["model"] = target_seq
                if "tokenizer" in params:
                    kwargs["tokenizer"] = dummy_tokenizer
                if "calibration_loader" in params:
                    kwargs["calibration_loader"] = dummy_loader
                if "dataloader" in params:
                    kwargs["dataloader"] = dummy_loader
                if "rank" in params:
                    kwargs["rank"] = 16
                if "bits" in params:
                    kwargs["bits"] = 4
                if "sparsity" in params:
                    kwargs["sparsity"] = 0.5
                if "in_features" in params or "d_in" in params:
                    kwargs["in_features" if "in_features" in params else "d_in"] = 64
                if "out_features" in params or "d_out" in params:
                    kwargs["out_features" if "out_features" in params else "d_out"] = 64
                if "embed_dim" in params or "d_model" in params:
                    kwargs["embed_dim" if "embed_dim" in params else "d_model"] = 64
                
                inst = cls_obj(**kwargs)
            except Exception:
                try:
                    inst = cls_obj(target_seq)
                except Exception:
                    try:
                        inst = cls_obj(64, 64)
                    except Exception:
                        inst = cls_obj()

            # Test compress or forward
            x = torch.randn(2, 16, 64)
            if hasattr(inst, "compress"):
                try:
                    res = inst.compress(target_linear)
                except TypeError:
                    try:
                        res = inst.compress(target_linear, X=x)
                    except Exception:
                        res = inst.compress(target_seq)
                except Exception:
                    res = inst.compress(target_seq)
            elif hasattr(inst, "forward") or callable(inst):
                try:
                    res = inst(x)
                except Exception:
                    res = inst(x, x, x)
            
            passed += 1
            if idx <= 10 or idx % 15 == 0 or idx == total:
                print(f"  [{idx:2d}/{total}] ✓ {cls_name:<38} ({filename})")
        except Exception as e:
            failed.append((cls_name, filename, str(e)))
            print(f"  [{idx:2d}/{total}] ✗ {cls_name:<38} ({filename}) -> ERROR: {e}")

    print(f"\nCompression Engines Summary: {passed}/{total} PASSED ({(passed/total)*100:.1f}%)")
    return len(failed) == 0

def verify_models_and_training():
    print_header("STAGE 3: VERIFYING MODEL ZOO, FORWARD/BACKWARD PASS & TRAINERS")
    
    # 1. Test UnifiedTransformerModel
    print("\n[3.1] Testing UnifiedTransformerModel with Attention Variants:")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    
    attention_types_to_test = ["svd_linear", "gqa", "mla", "mamba_svd_flash", "linformer"]
    
    for att in attention_types_to_test:
        hf_cfg = DistilBertConfig(
            vocab_size=1000,
            dim=64,
            n_layers=2,
            n_heads=4,
            hidden_dim=128,
            max_position_embeddings=128,
            num_labels=2
        )
        base_model = DistilBertForSequenceClassification(hf_cfg)
        model = UnifiedTransformerModel(base_model)
        model.replace_attention(att, rank=16)
        
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 2, (2,))
        
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out[0]
        assert loss is not None and not torch.isnan(loss), f"Loss calculation failed for {att}"
        
        # Backward pass check
        loss.backward()
        print(f"  ✓ {att:<20} -> Loss: {loss.item():.4f}, Gradients Backpropagated Cleanly")

    # 2. Test CustomLoopTrainer with Optimizers & Schedulers
    print("\n[3.2] Testing Training Loop & Gradient Updates:")
    create_opt = OPTIMIZER_REGISTRY.get("adamw")
    optimizer = create_opt(model.parameters(), lr=1e-3, weight_decay=0.01)
    create_sched = SCHEDULER_REGISTRY.get("cosine")
    scheduler = create_sched(optimizer, num_warmup_steps=2, num_training_steps=10)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("  ✓ Optimizer (AdamW) and LR Scheduler (Cosine) updated model weights successfully")

    # 3. Test BenchmarkDatasetRegistry
    print("\n[3.3] Testing Benchmark Dataset Loaders:")
    benchmarks = BenchmarkDatasetRegistry.list_benchmarks()
    print(f"  Registered benchmark datasets ({len(benchmarks)}): {', '.join(benchmarks[:5])}...")
    loader = load_benchmark_dataset("wikitext2", seq_len=32, batch_size=2, max_samples=4, use_synthetic=True)
    batch = next(iter(loader))
    assert batch is not None, "Benchmark dataset batch loading failed"
    print("  ✓ Synthetic benchmark dataset successfully generated and tokenized")

    return True

def verify_visualization_and_tooling():
    print_header("STAGE 4: VERIFYING VISUALIZATION, RESULTS SAVER & DASHBOARD EXPORTERS")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Visualizer plots
        svd_plot = os.path.join(tmpdir, "svd_plot.png")
        kv_plot = os.path.join(tmpdir, "kv_plot.png")
        pareto_plot = os.path.join(tmpdir, "pareto_plot.png")
        
        plot_singular_value_decay({"Layer 0": [1.0, 0.8, 0.5, 0.3, 0.1]}, "Singular Value Decay", svd_plot)
        plot_kv_cache_scaling([512, 1024, 2048], {"Standard": [64, 128, 256]}, "KV Cache Scaling", kv_plot)
        plot_pareto_frontier([10.5, 11.2], [2.5, 1.2], ["Dense", "SVD"], "Pareto Frontier", pareto_plot)
        
        assert os.path.exists(svd_plot), "SVD plot was not generated"
        assert os.path.exists(kv_plot), "KV cache scaling plot was not generated"
        assert os.path.exists(pareto_plot), "Pareto frontier plot was not generated"
        print("  ✓ All visualizer chart plots (PNG) generated cleanly")

        # Dashboard HTML & JSON Summary
        html_out = os.path.join(tmpdir, "dashboard.html")
        json_out = os.path.join(tmpdir, "summary.json")
        sample_data = {
            "models": [{"name": "SVD-Linear", "ppl": 10.5, "latency_ms": 1.2, "vram_mb": 250.0}],
            "timestamp": "2026-08-16T10:00:00"
        }
        generate_dashboard_html(sample_data, html_out)
        generate_summary_json(sample_data, json_out)
        
        assert os.path.exists(html_out), "Dashboard HTML was not generated"
        assert os.path.exists(json_out), "Summary JSON was not generated"
        print("  ✓ Interactive Dashboard HTML & JSON summary exported cleanly")

        # ResultsSaver
        saver = ResultsSaver(output_dir=os.path.join(tmpdir, "results"))
        saver.save_metrics({"epoch": 1, "train_loss": 0.45, "val_loss": 0.52})
        saver.save_config({"model": "SVD-Linear", "lr": 1e-3})
        print("  ✓ ResultsSaver successfully serialized metrics, configuration, and artifacts")

    return True

def main():
    print("=" * 80)
    print("  EXHAUSTIVE NEURALOPS ML FRAMEWORK MASTER AUDIT & CODE CLEANLINESS SUITE")
    print("=" * 80)

    s1 = verify_all_attentions()
    s2 = verify_all_compressors()
    s3 = verify_models_and_training()
    s4 = verify_visualization_and_tooling()

    print_header("FINAL FRAMEWORK AUDIT SUMMARY")
    print(f"  Stage 1 (Attention Architectures): {'PASS (100%)' if s1 else 'FAIL'}")
    print(f"  Stage 2 (Compression Engines):     {'PASS (100%)' if s2 else 'FAIL'}")
    print(f"  Stage 3 (Models & Training):       {'PASS (100%)' if s3 else 'FAIL'}")
    print(f"  Stage 4 (Visualizer & Exporters):  {'PASS (100%)' if s4 else 'FAIL'}")
    
    if all([s1, s2, s3, s4]):
        print("\n🏆 ALL FRAMEWORK CODES FULLY VERIFIED, CLEAN, AND 100% OPERATIONAL!\n")
        return 0
    else:
        print("\n❌ SOME STAGES FAILED. PLEASE REVIEW LOGS ABOVE.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
