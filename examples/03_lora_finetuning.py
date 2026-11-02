#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 03: Parameter-Efficient Fine-Tuning (PEFT) with LoRA Adaptation.

Subsystem Demonstrated:
    `neuralops.core.peft.lora.LoRALinear` and `neuralops.replace_attention(inject_lora=True)`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Injecting low-rank adapters (LoRA) into multi-head query and value projections.
    2. Enforcing selective parameter freezing (backbone requires_grad=False, adapter requires_grad=True).
    3. Performing forward and backward training steps with loss convergence.
    4. Merging adapters into base weights (`.merge()`) for zero-latency inference deployment.
    5. Saving and reloading adapter checkpoints with numerical parity validation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from examples._shared import (
    TinyClassifier,
    assert_parity,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    tiny_inputs,
)
from neuralops.core.peft.lora import LoRALinear


def inject_multi_target_lora(model: torch.nn.Module, rank: int = 4) -> list[str]:
    """Wrap q_lin and v_lin projections in each transformer block with LoRALinear."""
    wrapped_names: list[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "attention"):
            attn = module.attention
            if hasattr(attn, "q_lin") and isinstance(attn.q_lin, torch.nn.Linear):
                attn.q_lin = LoRALinear.from_linear(attn.q_lin, lora_rank=rank, freeze_base=True)
                wrapped_names.append(f"{name}.attention.q_lin")
            if hasattr(attn, "v_lin") and isinstance(attn.v_lin, torch.nn.Linear):
                attn.v_lin = LoRALinear.from_linear(attn.v_lin, lora_rank=rank, freeze_base=True)
                wrapped_names.append(f"{name}.attention.v_lin")
    return wrapped_names


def run_lora_workflow(device: str) -> bool:
    """Execute complete LoRA lifecycle: injection, freezing, training, merge, and save/load."""
    section("Part 1: Multi-Target LoRA Injection & Selective Freezing")

    model = TinyClassifier().to(device)
    total_initial_params = model.count_parameters()

    wrapped_layers = inject_multi_target_lora(model, rank=4)
    kv("Wrapped Layer Projections", f"{len(wrapped_layers)} targets ({', '.join(wrapped_layers[:2])}...)")

    # Freeze non-adapter backbone parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    trainable_params = model.count_trainable_parameters()
    total_params = model.count_parameters()
    kv("Total Model Parameters", f"{total_params:,}")
    kv("Trainable Adapter Parameters", f"{trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    correct_freezing = trainable_params < total_initial_params and trainable_params > 0
    pass_fail("Selective Parameter Freezing Check", correct_freezing)

    # Part 2: Forward and Backward Step
    section("Part 2: Training Step & Gradient Verification")
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}
    labels = torch.randint(0, 2, (inputs["input_ids"].shape[0],), device=device)

    model.train()
    out = model(**inputs, labels=labels)
    loss = out.loss
    loss.backward()

    finite_loss = bool(torch.isfinite(loss).item())
    pass_fail("Forward Loss Finiteness", finite_loss, f"loss={loss.item():.4f}")

    # Verify gradients only exist on LoRA parameters
    grad_only_on_lora = True
    for name, param in model.named_parameters():
        if "lora_" in name:
            if param.grad is None or not torch.isfinite(param.grad).all():
                grad_only_on_lora = False
        else:
            if param.grad is not None:
                grad_only_on_lora = False

    pass_fail("Gradient Isolation (Only LoRA adapters receive gradients)", grad_only_on_lora)

    # Part 3: Merge and Unload Round-Trip
    section("Part 3: Adapter Merging Parity Check")
    model.eval()
    with torch.no_grad():
        out_unmerged = model(**inputs).logits

    # Merge all LoRA adapters into the base weights
    lora_modules: list[LoRALinear] = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for mod in lora_modules:
        mod.merge()

    with torch.no_grad():
        out_merged = model(**inputs).logits

    merge_parity = assert_parity(out_merged, out_unmerged, atol=1e-4, label="LoRA Merge Parity Check")

    # Part 4: Save & Reload Round-Trip
    section("Part 4: Adapter Checkpoint Save & Reload")
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_path = Path(tmp_dir) / "lora_adapter.pt"
        adapter_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        torch.save(adapter_state, adapter_path)
        kv("Saved Adapter Checkpoint", f"{adapter_path.name} ({len(adapter_state)} tensors)")

        # Instantiate fresh model and inject fresh adapters
        fresh_model = TinyClassifier().to(device)
        inject_multi_target_lora(fresh_model, rank=4)

        # Unmerge first model to compare against fresh reloaded model
        for mod in lora_modules:
            mod.unmerge()
        with torch.no_grad():
            ref_logits = model(**inputs).logits

        # Load weights into fresh model
        fresh_model.load_state_dict(torch.load(adapter_path, weights_only=True), strict=False)
        fresh_model.eval()
        with torch.no_grad():
            reloaded_logits = fresh_model(**inputs).logits

        reload_parity = assert_parity(reloaded_logits, ref_logits, atol=1e-5, label="Adapter Reload Parity Check")

    return bool(correct_freezing and finite_loss and grad_only_on_lora and merge_parity and reload_parity)


def main() -> int:
    section("NeuralOps Example 03: LoRA PEFT & Adapter Merging")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_lora_workflow(backend)
    print()
    if success:
        pass_fail("Overall Example 03 Status", True, "All LoRA lifecycle and merge checks passed")
        return 0
    else:
        pass_fail("Overall Example 03 Status", False, "One or more LoRA checks failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
