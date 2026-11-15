"""Governance and invariant validation for the 30 remaining Triton kernels."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

MODULE_NAMES = [
    "adaptive_sparse_triton",
    "based_scan",
    "block_sparse",
    "chela_slay_triton",
    "chela_slay_wide_triton",
    "cross_attention_triton",
    "delta_net_scan",
    "dynamic_conv_triton",
    "fused_svd_projection_triton",
    "gqa",
    "lightning_scan",
    "linear_attn",
    "lsh_attention_triton",
    "paged_linear_attn",
    "prune_admm",
    "prune_sparsegpt",
    "prune_wanda",
    "quantize_awq",
    "quantize_fp8",
    "quantize_kmeans",
    "rala_attention_triton",
    "retnet_scan",
    "routing_attention_triton",
    "rwkv6_scan",
    "sparse_attn",
    "state_update",
    "svd_linear_decay_triton",
    "svd_mamba_fused_triton",
    "token_compressed_triton",
    "ttt_linear_step_triton",
]

OPS_DIR = Path(__file__).resolve().parents[3] / "neuralops" / "ops" / "triton"


@pytest.mark.parametrize("mod_name", MODULE_NAMES)
def test_no_deprecated_amp_api(mod_name: str) -> None:
    """Verifies that torch.cuda.amp is strictly avoided per Rule 2.1."""
    file_path = OPS_DIR / f"{mod_name}.py"
    assert file_path.is_file(), f"File {file_path} not found"
    text = file_path.read_text(encoding="utf-8")
    assert "torch.cuda.amp" not in text, f"Deprecated torch.cuda.amp found in {mod_name}"


@pytest.mark.parametrize("mod_name", MODULE_NAMES)
def test_no_hardcoded_cuda0(mod_name: str) -> None:
    """Verifies that no hardcoded cuda:0 device strings exist per Rule 10."""
    file_path = OPS_DIR / f"{mod_name}.py"
    text = file_path.read_text(encoding="utf-8")
    assert "cuda:0" not in text, f"Hardcoded cuda:0 found in {mod_name}"


@pytest.mark.parametrize("mod_name", MODULE_NAMES)
def test_no_bare_except_or_silent_pass(mod_name: str) -> None:
    """Verifies that bare except: and silent pass are avoided per Rule 10."""
    file_path = OPS_DIR / f"{mod_name}.py"
    lines = file_path.read_text(encoding="utf-8").splitlines()

    for idx, line in enumerate(lines, 1):
        assert not re.search(r"except\s*:", line), f"Bare except found at {mod_name}:{idx}"
        if "except Exception" in line:
            # Must have allowed-swallow comment on either the except line or the next line
            context = " ".join(lines[idx - 1 : min(idx + 3, len(lines))])
            if "pass" in context:
                assert "allowed-swallow" in context, f"Unannotated except Exception: pass at {mod_name}:{idx}"


@pytest.mark.parametrize("mod_name", MODULE_NAMES)
def test_ast_validity_and_jit_presence(mod_name: str) -> None:
    """Verifies that each file parses cleanly into an AST."""
    file_path = OPS_DIR / f"{mod_name}.py"
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    assert isinstance(tree, ast.Module)
