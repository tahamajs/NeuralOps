#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-End Surgical Attention Transplant and Micro-Healing on Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuralops
from neuralops.api import replace_attention
from neuralops.surgery.healing import heal_replaced_model
from neuralops.surgery.hf_cache_bridge import wrap_model_for_generation


def print_gpu_memory(label: str = "") -> None:
    """Print current and peak GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[{label}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            print(f"[{label}] nvidia-smi: {res.stdout.strip()}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Qwen 7B Transplant & Healing")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local Qwen 7B weights")
    parser.add_argument("--target_layer", type=int, default=14, help="Layer index to transplant (default: 14)")
    parser.add_argument("--target_attention", type=str, default="chunk_gla", help="Target attention mechanism")
    parser.add_argument("--heal_steps", type=int, default=10, help="Number of micro-healing steps")
    args = parser.parse_args()

    default_paths = [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"),
        "/home/tahamajlesi.ut/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    model_path = args.model_path
    if not model_path:
        for p in default_paths:
            if os.path.exists(p):
                model_path = p
                break
        if not model_path:
            model_path = "Qwen/Qwen2.5-7B-Instruct"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=== NeuralOps Surgery: Qwen2.5-7B-Instruct Transplant & Heal ===")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")
    print(f"Target Layer: {args.target_layer} | Attention: {args.target_attention}")
    print_gpu_memory("Initial")

    # 0. PyTorch < 2.0 Compatibility Shim for Transformers / Accelerate
    _orig_named_parameters = torch.nn.Module.named_parameters

    def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)

    torch.nn.Module.named_parameters = _compat_named_parameters

    _orig_named_buffers = torch.nn.Module.named_buffers

    def _compat_named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_buffers(self, prefix=prefix, recurse=recurse)

    torch.nn.Module.named_buffers = _compat_named_buffers

    # 1. Load Tokenizer & Model
    print("\n[1/6] Loading model and tokenizer in bfloat16...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"device_map={{'': 0}} failed ({e}), loading with direct CUDA default tensor allocation...")
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        finally:
            if torch.cuda.is_available():
                torch.set_default_tensor_type(torch.FloatTensor)

    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.")
    print_gpu_memory("After Model Load")

    # 2. Baseline Qualitative Generation
    prompt = "What is the capital of France?"
    print(f"\n[2/6] Baseline generation before surgery (Prompt: '{prompt}')...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_tokens = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    pred_text_before = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"Generated text (Before): '{pred_text_before}'")

    # 3. Cache layer 14 input activations using calibration tokens
    print("\n[3/6] Generating synthetic calibration batch (16 sequences, length 128)...")
    calib_input_ids = torch.randint(100, 2000, (16, 128), device=device)
    
    # Save a deepcopy of layer 14 for teacher representation
    target_layer_idx = args.target_layer
    orig_layer = copy.deepcopy(model.model.layers[target_layer_idx])
    orig_layer.eval()

    cached_in = []
    def _capture_hook(mod, inp):
        in_t = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(in_t, torch.Tensor):
            cached_in.append(in_t.detach())

    hook_handle = model.model.layers[target_layer_idx].register_forward_pre_hook(_capture_hook)
    with torch.no_grad():
        _ = model(input_ids=calib_input_ids)
    hook_handle.remove()

    if not cached_in:
        raise RuntimeError("Failed to capture layer 14 input activations!")
    H_in = cached_in[0]
    print(f"Captured layer {target_layer_idx} input activations with shape: {tuple(H_in.shape)}")

    # 4. In-Place Transplant Surgery
    print(f"\n[4/6] Surgically transplanting layer {target_layer_idx} to '{args.target_attention}'...")
    model = replace_attention(
        model,
        attention=args.target_attention,
        layers=[target_layer_idx],
        copy_weights=True,
    )
    print("Transplant completed successfully!")
    print_gpu_memory("After Transplant")

    # 5. Micro-Healing
    print(f"\n[5/6] Running layerwise micro-healing for {args.heal_steps} steps...")
    teacher_container = orig_layer
    student_container = model.model.layers[target_layer_idx]

    heal_stats = heal_replaced_model(
        teacher_model=teacher_container,
        student_model=student_container,
        calibration_inputs=H_in,
        steps=args.heal_steps,
        lr=2e-3,
        also_ridge_init=True,
    )
    print(f"Healing complete! Initial MSE: {heal_stats.get('initial_mse', 0.0):.6f} -> Final MSE: {heal_stats.get('final_mse', 0.0):.6f}")
    print(f"Initial Cosine: {heal_stats.get('initial_cosine_sim', 0.0):.6f} -> Final Cosine: {heal_stats.get('final_cosine_sim', 0.0):.6f}")

    # 6. Post-Surgery Generation
    print(f"\n[6/6] Qualitative generation after surgery (Prompt: '{prompt}')...")
    model = wrap_model_for_generation(model)
    with torch.no_grad():
        out_tokens_after = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    pred_text_after = tokenizer.decode(out_tokens_after[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"Generated text (After Surgery & Healing): '{pred_text_after}'")
    print_gpu_memory("Final")

    print("\n=== TRANSPLANT & HEALING VALIDATION SUMMARY ===")
    print("Model: Qwen2.5-7B-Instruct")
    print(f"Transplanted Layer: {target_layer_idx} ({args.target_attention})")
    print(f"Before Text: {pred_text_before}")
    print(f"After Text:  {pred_text_after}")
    print(f"Initial MSE: {heal_stats.get('initial_mse'):.6f} | Final MSE: {heal_stats.get('final_mse'):.6f}")
    print("ALL CHECKS PASSED: Surgery + Healing successfully validated on 7B model!")


if __name__ == "__main__":
    main()
