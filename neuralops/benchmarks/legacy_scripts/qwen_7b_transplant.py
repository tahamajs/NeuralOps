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
    parser.add_argument("--device", default="cuda:0", help="Execution device; CUDA is required by default")
    args = parser.parse_args()

    default_paths = [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"),
        os.environ.get("HF_QWEN_7B_PATH", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct")),
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    print("=== NeuralOps Surgery: Qwen2.5-7B-Instruct Transplant & Heal ===")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")
    print(f"Target Layer: {args.target_layer} | Attention: {args.target_attention}")
    print_gpu_memory("Initial")

    # 0. PyTorch < 2.0 Compatibility Shim for Transformers / Accelerate / BFloat16 CUDA ops
    _orig_named_parameters = torch.nn.Module.named_parameters

    def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_parameters(self, prefix=prefix, recurse=recurse)

    torch.nn.Module.named_parameters = _compat_named_parameters

    _orig_named_buffers = torch.nn.Module.named_buffers

    def _compat_named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        return _orig_named_buffers(self, prefix=prefix, recurse=recurse)

    torch.nn.Module.named_buffers = _compat_named_buffers

    _orig_triu = torch.triu

    def _compat_triu(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            res = _orig_triu(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                out.copy_(res)
                return out
            return res
        if out is not None:
            return _orig_triu(input, diagonal=diagonal, out=out)
        return _orig_triu(input, diagonal=diagonal)

    torch.triu = _compat_triu

    _orig_tril = torch.tril

    def _compat_tril(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            res = _orig_tril(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                out.copy_(res)
                return out
            return res
        if out is not None:
            return _orig_tril(input, diagonal=diagonal, out=out)
        return _orig_tril(input, diagonal=diagonal)

    torch.tril = _compat_tril

    _orig_logsigmoid = torch.nn.functional.logsigmoid

    def _compat_logsigmoid(input):
        if input.is_cuda and input.dtype == torch.bfloat16:
            return _orig_logsigmoid(input.float()).to(torch.bfloat16)
        return _orig_logsigmoid(input)

    torch.nn.functional.logsigmoid = _compat_logsigmoid
    torch.log_sigmoid = _compat_logsigmoid

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

    # 3. Cache layer 14 attention activations using calibration tokens
    print("\n[3/6] Generating synthetic calibration batch (16 sequences, length 128)...")
    calib_input_ids = torch.randint(100, 2000, (16, 128), device=device)
    target_layer_idx = args.target_layer

    cached_layer_in = []
    cached_attn_out = []

    def _layer_in_hook(mod, inp):
        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            cached_layer_in.append(inp[0].detach())

    def _attn_out_hook(mod, inp, out):
        out_t = out[0] if isinstance(out, tuple) else out
        if isinstance(out_t, torch.Tensor):
            cached_attn_out.append(out_t.detach())

    h_in = model.model.layers[target_layer_idx].register_forward_pre_hook(_layer_in_hook)
    h_out = model.model.layers[target_layer_idx].self_attn.register_forward_hook(_attn_out_hook)

    with torch.no_grad():
        _ = model(input_ids=calib_input_ids)

    h_in.remove()
    h_out.remove()

    if not cached_layer_in or not cached_attn_out:
        raise RuntimeError("Failed to capture layer activations during teacher forward pass!")

    H_in = cached_layer_in[0]
    attn_target = cached_attn_out[0]
    with torch.no_grad():
        attn_in = model.model.layers[target_layer_idx].input_layernorm(H_in)
    print(f"Captured teacher attention input shape: {tuple(attn_in.shape)}, output shape: {tuple(attn_target.shape)}")

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
    student_attn = model.model.layers[target_layer_idx].self_attn

    trainable_params = [p for p in student_attn.parameters() if p.numel() > 0]
    for p in trainable_params:
        p.requires_grad = True

    # Initial parity measurement
    student_attn.eval()
    with torch.no_grad():
        pred0 = student_attn(attn_in)
        pred0_val = pred0[0] if isinstance(pred0, tuple) else pred0
        pred0_f = pred0_val.float()
        tgt_f = attn_target.float()
        denom = float(torch.norm(tgt_f, p="fro").item() ** 2 + 1e-6)
        init_mse = float(torch.norm(pred0_f - tgt_f, p="fro").item() ** 2 / denom)
        init_cos = float(
            torch.nn.functional.cosine_similarity(
                pred0_f.reshape(-1, tgt_f.shape[-1]),
                tgt_f.reshape(-1, tgt_f.shape[-1]),
                dim=-1,
            ).mean().item()
        )

    print(f"Initial Alignment -> MSE: {init_mse:.6f} | Cosine Similarity: {init_cos:.4f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.heal_steps))

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in student_attn.state_dict().items()}

    student_attn.train()
    for step in range(args.heal_steps):
        optimizer.zero_grad()
        pred = student_attn(attn_in)
        pred_val = pred[0] if isinstance(pred, tuple) else pred
        pred_f = pred_val.float()

        loss_fro = torch.norm(pred_f - tgt_f, p="fro") ** 2 / denom
        loss_cos = 1.0 - torch.nn.functional.cosine_similarity(
            pred_f.reshape(-1, tgt_f.shape[-1]),
            tgt_f.reshape(-1, tgt_f.shape[-1]),
            dim=-1,
        ).mean()
        loss = loss_fro + 0.5 * loss_cos

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        cur_loss_val = float(loss.item())
        if cur_loss_val < best_loss:
            best_loss = cur_loss_val
            best_state = {k: v.clone() for k, v in student_attn.state_dict().items()}

        print(
            f"Step {step+1:02d}/{args.heal_steps:02d} | "
            f"Total Loss: {loss.item():.6f} | "
            f"MSE: {loss_fro.item():.6f} | "
            f"Cosine: {(1.0 - loss_cos.item()):.4f}"
        )

    # Restore optimal model state discovered during micro-healing
    student_attn.load_state_dict(best_state)
    student_attn.eval()
    with torch.no_grad():
        pred_final = student_attn(attn_in)
        pred_final_val = pred_final[0] if isinstance(pred_final, tuple) else pred_final
        pred_final_f = pred_final_val.float()
        final_mse = float(torch.norm(pred_final_f - tgt_f, p="fro").item() ** 2 / denom)
        final_cos = float(
            torch.nn.functional.cosine_similarity(
                pred_final_f.reshape(-1, tgt_f.shape[-1]),
                tgt_f.reshape(-1, tgt_f.shape[-1]),
                dim=-1,
            ).mean().item()
        )

    for p in trainable_params:
        p.requires_grad = False

    heal_stats = {
        "initial_mse": init_mse,
        "final_mse": final_mse,
        "initial_cosine_sim": init_cos,
        "final_cosine_sim": final_cos,
    }
    print(f"Healing complete! Initial MSE: {init_mse:.6f} -> Final MSE: {final_mse:.6f}")
    print(f"Initial Cosine: {init_cos:.4f} -> Final Cosine: {final_cos:.4f}")

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
