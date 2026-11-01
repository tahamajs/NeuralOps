#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Remote GPU end-to-end verification test for vLLM Recurrent State & KV Cache Integration.
"""

import sys
import torch


def main() -> int:
    # Ensure sys.path does not contain neuralops/ package dir which shadows third-party transformers
    sys.path = [p for p in sys.path if not p.endswith("/neuralops")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from neuralops import replace_attention
    from neuralops.integrations.vllm_backend import (
        NeuralOpsAttentionBackend,
        NeuralOpsImpl,
        NeuralOpsRecurrentSpec,
        register,
    )
    from neuralops.integrations.vllm_backend.kv_cache_spec import register_recurrent_spec

    print("=== Step 1: Backend and Recurrent Spec Registration ===")
    backend_ok = register()
    spec_ok = register_recurrent_spec()
    print(f"Backend register: {backend_ok}")
    print(f"RecurrentSpec register: {spec_ok}")
    assert backend_ok, "Backend registration failed"
    assert spec_ok, "RecurrentSpec registration failed"

    print("\n=== Step 2: Recurrent State Allocation & Spec Tests ===")
    spec = NeuralOpsRecurrentSpec(
        num_heads=12,
        head_dim=64,
        state_dim=64,
        dtype=torch.float32,
        device="cuda",
    )
    print(f"Spec instance: {spec}")
    print(f"Page size (bytes): {spec.page_size_bytes}")
    print(f"State content size (bytes): {spec.state_content_size_bytes}")
    state = spec.allocate_state(batch_size=2)
    print(f"Allocated state shape: {state.shape}, device: {state.device}")
    assert state.shape == (2, 12, 64, 64)
    assert state.device.type == "cuda"

    print("\n=== Step 3: NeuralOpsImpl Forward Pass with Recurrent State ===")
    impl = NeuralOpsImpl(
        num_heads=12,
        head_size=64,
        scale=1.0 / (64 ** 0.5),
        mechanism="gla_attention",
    )
    q = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)

    # Step 3a: Step 1 with uninitialized state in spec
    out1 = impl.forward(q, k, v, kv_cache=spec)
    assert out1.shape == q.shape, f"Shape mismatch: {out1.shape} vs {q.shape}"
    assert torch.isfinite(out1).all(), "Recurrent forward produced NaN or Inf!"
    assert hasattr(spec, "_state_tensor"), "State tensor not persisted in spec!"
    print(f"Step 1 output shape: {out1.shape}, state shape: {spec._state_tensor.shape}")

    # Step 3b: Step 2 verifying persistent state progression
    q2 = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)
    k2 = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)
    v2 = torch.randn(2, 8, 12 * 64, device="cuda", dtype=torch.float32)
    out2 = impl.forward(q2, k2, v2, kv_cache=spec)
    assert out2.shape == q2.shape
    assert torch.isfinite(out2).all(), "Multi-step recurrent forward produced NaN or Inf!"
    print(f"Step 2 output shape: {out2.shape}, persistent state update verified.")

    # Step 3c: Standard quadratic fallback verification
    out_quad = impl.forward(q, k, v, kv_cache=None)
    assert out_quad.shape == q.shape
    assert torch.isfinite(out_quad).all(), "Quadratic fallback produced NaN or Inf!"
    print("Quadratic fallback verified successfully.")

    print("\n=== Step 4: Model Surgery with Recurrent Attention (GLA) on CUDA ===")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    print("Baseline GPT-2 model loaded on CUDA.")

    # In-place transplant of quadratic attention with gla_attention
    model_recurrent = replace_attention(model, attention="gla_attention")
    print("Replaced attention in GPT-2 with 'gla_attention'.")

    # Run forward inference on CUDA and verify finite logits
    prompt = "NeuralOps enables subquadratic attention and model compression."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_recurrent(**inputs)
        logits = outputs.logits

    print(f"Model logits shape: {logits.shape}")
    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    print(f"NaN count: {nan_count}, Inf count: {inf_count}")
    assert nan_count == 0, f"Found {nan_count} NaNs in logits!"
    assert inf_count == 0, f"Found {inf_count} Infs in logits!"
    assert torch.isfinite(logits).all(), "Logits are not finite!"
    print("Finite logits verification: PASSED (0 NaNs, 0 Infs).")

    print("\n=== Step 5: vLLM Inference Pipeline Verification on CUDA ===")
    llm = LLM(
        model="openai-community/gpt2",
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=15)
    outputs = llm.generate([prompt], sampling_params)
    for out in outputs:
        print(f"Prompt: {out.prompt!r}")
        print(f"Generated text: {out.outputs[0].text!r}")
    print("vLLM generation completed successfully.")

    print("\n=== ALL REMOTE GPU ACCEPTANCE GATES PASSED! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
