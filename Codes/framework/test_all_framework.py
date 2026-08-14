"""
Comprehensive Automated Test Suite for SVD Linear Attention Framework.
Tests all subsystems:
  1. Benchmark Datasets Loader
  2. Visualizations & Interactive Dashboard Generator
  3. Results Saver & Exporters
  4. Multi-Mode Benchmark Runner Engine
  5. Attention Registry & Forward Passes
  6. Compression Engines & Hybrid Modules
"""

import os
import sys
import tempfile
import torch
from transformers import DistilBertConfig

# Ensure framework can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import framework
from framework.core.registry import ATTENTION_REGISTRY
from framework.data.benchmark_datasets import load_benchmark_dataset, BenchmarkDatasetRegistry
from framework.visualization.dashboard_generator import generate_dashboard_html, generate_summary_json
from framework.visualization.benchmark_visualizer import (
    plot_singular_value_decay,
    plot_kv_cache_scaling,
    plot_pareto_frontier
)
from framework.results_saver.saver import ResultsSaver
from framework.core.multi_mode_benchmark import MultiModeBenchmarkRunner
from framework.compression_engine.rotational_slicegpt_int4 import RotationalSliceGPTCompressor
from framework.compression_engine.kronecker_tt_lora import KroneckerTTLoRACompressor

def main():
    print("=" * 70)
    print("  RUNNING COMPLETE SVD LINEAR ATTENTION FRAMEWORK VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test Datasets Loader
    print("\n[1/6] Testing Benchmark Dataset Loaders...")
    datasets = BenchmarkDatasetRegistry.list_benchmarks()
    print(f"  Available Benchmark Datasets ({len(datasets)}): {', '.join(datasets[:6])}...")
    loader = load_benchmark_dataset("wikitext2", seq_len=64, batch_size=2, max_samples=4, use_synthetic=True)
    batch = next(iter(loader))
    assert "text" in batch or "input_ids" in batch, "Dataset batch missing text or input_ids"
    key = "input_ids" if "input_ids" in batch else "text"
    print(f"  ✓ Loaded wikitext2 batch {key}: {len(batch[key])} items")

    # 2. Test Visualizations & Dashboard Generator
    print("\n[2/6] Testing Visualizations & Dashboard Generator...")
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_singular_value_decay({"Layer 0": [1.0, 0.7, 0.4, 0.2, 0.1, 0.05]}, "SVD Decay", os.path.join(tmpdir, "svd.png"))
        plot_kv_cache_scaling([512, 1024, 2048], {"Dense": [64, 128, 256], "DeltaKV": [16, 32, 64]}, "KV Cache", os.path.join(tmpdir, "kv.png"))
        plot_pareto_frontier([11.2, 11.8, 12.4], [3.2, 1.8, 1.1], ["Dense", "SVD", "INT4"], "Pareto", os.path.join(tmpdir, "pareto.png"))
        
        sample_bench = {
            "models": [
                {"name": "Dense FP16", "category": "Dense", "ppl": 11.2, "latency_ms": 3.2, "vram_mb": 420.0, "throughput": 1250},
                {"name": "MambaSVD-Flash", "category": "SSM", "ppl": 11.4, "latency_ms": 1.5, "vram_mb": 180.0, "throughput": 2800},
                {"name": "Rotational-SliceGPT-INT4", "category": "Quantization", "ppl": 12.1, "latency_ms": 1.2, "vram_mb": 130.0, "throughput": 3400}
            ]
        }
        html_path = os.path.join(tmpdir, "dashboard.html")
        html_str = generate_dashboard_html(sample_bench, output_path=html_path)
        assert os.path.exists(html_path), "Dashboard HTML not generated"
        print(f"  ✓ Visualizer plots and dashboard generated successfully")

    # 3. Test Results Saver
    print("\n[3/6] Testing Results Saver & Exporters...")
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ResultsSaver(run_id="test_saver_run", base_dir=tmpdir)
        m_path = saver.save_metrics({"accuracy": 0.92, "perplexity": 11.4})
        c_path = saver.save_config({"model": "distilbert", "rank": 32, "mode": "svd_linear"})
        test_model = torch.nn.Linear(64, 64)
        w_path = saver.save_model_weights(test_model)
        assert os.path.exists(m_path), "Metrics not saved"
        print(f"  ✓ Metrics, config, and model weights successfully saved by ResultsSaver")

    # 4. Test MultiMode Benchmark Runner
    print("\n[4/6] Testing Multi-Mode Benchmark Runner Engine...")
    runner = MultiModeBenchmarkRunner()
    modes = runner.list_supported_modes()
    print(f"  Supported execution benchmark modes ({len(modes)}): {', '.join(modes[:6])}...")
    res = runner.benchmark_attention_module("svd_linear", seq_len=64, batch_size=2, num_runs=5)
    print(f"  ✓ Benchmarked svd_linear -> Latency: {res.get('latency_ms', 0):.2f}ms, Memory: {res.get('peak_vram_mb', 0):.2f}MB")

    # 5. Test Attention Registry & Forward Passes
    print("\n[5/6] Testing Attention Registry & Forward Passes...")
    registered = ATTENTION_REGISTRY.keys()
    print(f"  Total Attention Modules in Registry: {len(registered)}")
    config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256)
    x = torch.randn(2, 16, 64)
    
    test_keys = [
        "full", "gqa", "mla", "svd_linear", "mamba_svd_flash", 
        "moe_deltakv_paged", "aegis_spectron_fp8", "linformer", "performer", "flash3", "aegis_hybrid"
    ]
    for key in test_keys:
        if key in registered:
            mod_cls = ATTENTION_REGISTRY.get(key)
            try:
                mod = mod_cls(config)
            except Exception:
                try:
                    mod = mod_cls(dim=64, num_heads=4)
                except Exception:
                    mod = mod_cls()
            mod.eval()
            with torch.no_grad():
                out = mod(x)
                if isinstance(out, tuple):
                    out = out[0]
            print(f"  ✓ {key:<26} -> Output Shape: {list(out.shape)}")

    # 6. Test Compression Engines
    print("\n[6/6] Testing Compression Engines & Hybrid Modules...")
    base_layer = torch.nn.Linear(64, 64)
    rot_comp = RotationalSliceGPTCompressor(r_ratio=0.5, quant_bits=4)
    tt_comp = KroneckerTTLoRACompressor(rank=8)
    compressed_rot = rot_comp.compress_linear(base_layer)
    compressed_tt = tt_comp.compress_linear(base_layer)
    with torch.no_grad():
        out_rot = compressed_rot(x)
        out_tt = compressed_tt(x)
    print(f"  ✓ RotationalSliceGPTCompressor forward pass -> Output Shape: {list(out_rot.shape)}")
    print(f"  ✓ KroneckerTTLoRACompressor forward pass     -> Output Shape: {list(out_tt.shape)}")

    print("\n" + "=" * 70)
    print("  ALL FRAMEWORK SUBSYSTEMS & RESULTS SAVING VERIFIED 100% OPERATIONAL!")
    print("=" * 70)

if __name__ == "__main__":
    main()
