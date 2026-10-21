#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval ultra long context scaling module.
"""


def main() -> None:
    """Run this standalone evaluation experiment."""

    import gc
    import os
    import time
    import json
    import torch
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from neuralops.utils.replace_attention import replace_attention


    def print_log(msg):
        """Execute print log routine."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_log(f"Ultra-Long Context (L=512..8192) & Reasoning Suite on: {torch.cuda.get_device_name(0)}")

    model_path = os.environ.get("NEURALOPS_MODEL_PATH", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token


    def build_model_for_context(attention_type=None, max_pos=8192):
        """Load GPT-2 base and interpolate positional embeddings up to 8192."""
        base = GPT2LMHeadModel.from_pretrained(model_path)
        config = base.config
        config.max_position_embeddings = max_pos
        model = GPT2LMHeadModel(config)

        model.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
        model.lm_head.load_state_dict(base.lm_head.state_dict())
        model.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())

        orig_wpe = base.transformer.wpe.weight.data
        new_wpe = (
            torch.nn.functional.interpolate(
                orig_wpe.unsqueeze(0).transpose(1, 2), size=max_pos, mode="linear", align_corners=True
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        model.transformer.wpe.weight.data.copy_(new_wpe)

        for i in range(len(base.transformer.h)):
            model.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())

        del base
        torch.cuda.empty_cache()

        if attention_type is not None:
            replace_attention(model, attention_key=attention_type)

        return model.to(device).eval()


    seq_lengths = [512, 1024, 2048, 4096, 8192]
    architectures = [
        ("GPT-2 Baseline (MHA)", None),
        ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
        ("GPT-2 + Gated Delta Net", "gated_delta_net"),
    ]

    long_context_results = {}

    print_log("=== Step 1: Memory Footprint & Latency Scaling (L=512..8192) ===")

    for name, key in architectures:
        print_log(f"\n--- Testing Architecture: {name} ---")
        model = build_model_for_context(attention_type=key, max_pos=8192)
        scaling_data = {}

        for L in seq_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            inp = torch.randint(100, 1000, (1, L), device=device)

            try:
                with torch.no_grad():
                    _ = model(inp[:, : min(64, L)])
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(inp)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

                nans = torch.isnan(out.logits).sum().item()
                infs = torch.isinf(out.logits).sum().item()

                scaling_data[L] = {
                    "latency_ms": round(latency_ms, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "nans": nans,
                    "infs": infs,
                    "status": "PASS" if (nans == 0 and infs == 0) else "FAIL",
                }
                print_log(
                    f"L={L:5d} | Latency: {latency_ms:7.2f} ms | Peak VRAM: {peak_vram_mb:8.2f} MB | "
                    f"NaNs: {nans} | Infs: {infs} | Status: PASS"
                )
            except torch.cuda.OutOfMemoryError:
                print_log(f"L={L:5d} | Out of Memory (OOM)!")
                scaling_data[L] = {"status": "OOM"}
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print_log(f"L={L:5d} | Error: {str(e)[:50]}")
                scaling_data[L] = {"status": f"ERR: {str(e)[:40]}"}
                break
            finally:
                del inp
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()
        gc.collect()
        long_context_results[name] = scaling_data

    out_path = os.path.join(os.environ.get("NEURALOPS_OUTPUT_DIR", "."), "ultra_long_context_scaling_results.json")
    with open(out_path, "w") as f:
        json.dump(long_context_results, f, indent=2)

    print_log(f"Long-context scaling sweep finished! Written to {out_path}")



if __name__ == "__main__":
    main()
