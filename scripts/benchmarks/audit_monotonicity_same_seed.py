"""Audit monotonicity same seed module.

Author: Taha Majlesi
"""

import torch
import torch.nn as nn
import copy
from neuralops.core.registry import COMPRESSOR_REGISTRY


class TinyTransformer(nn.Module):
    """Tinytransformer definition."""
    def __init__(self, d=64):
        """Execute init routine."""
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)

    def forward(self, x):
        """Execute forward routine."""
        h = self.out_proj(self.v_proj(x)) + x
        return self.fc2(torch.relu(self.fc1(h)))


torch.manual_seed(42)
base_model = TinyTransformer(64)
x = torch.randn(2, 64)
with torch.no_grad():
    y_true = base_model(x)


def run_compressor_rank(name, rank):
    """Execute run compressor rank routine."""
    cls = COMPRESSOR_REGISTRY[name]
    m = copy.deepcopy(base_model)
    c = None
    for init_call in [
        lambda: cls(m, rank=rank, target_rank=rank, fallback_to_original=False),
        lambda: cls(m, rank=rank, fallback_to_original=False),
        lambda: cls(m, rank=rank),
        lambda: cls(m),
    ]:
        try:
            c = init_call()
            break
        except Exception:
            continue

    if c is None:
        raise RuntimeError(f"Could not instantiate compressor {name}")

    if hasattr(c, "compress"):
        res = c.compress()
        if isinstance(res, tuple):
            m = res[0]
        elif isinstance(res, nn.Module):
            m = res
    elif isinstance(c, nn.Module):
        m = c

    with torch.no_grad():
        y = m(x)
    return (torch.norm(y - y_true) / torch.norm(y_true)).item()


print("=== Monotonicity Check (Rank 8 vs 16 vs 32 vs 48) on Identical Weights ===")
candidates = [
    "direct_svd",
    "aa_svd_compressor",
    "adasvd_compressor",
    "adaptive_ranks",
    "fwsvd_compressor",
    "dobi_svd",
    "erc_svd",
    "tt_compressor",
    "compot_compressor",
    "kronecker_compressor",
    "rpca_compressor",
    "entropy_budget_allocation",
    "cur_compressor",
    "cr_compressor",
    "slicegpt",
    "sparsegpt",
    "flap",
    "admm_pruner",
    "rotational_slicegpt_int4",
]

for name in candidates:
    try:
        e8 = run_compressor_rank(name, 8)
        e16 = run_compressor_rank(name, 16)
        e32 = run_compressor_rank(name, 32)
        e48 = run_compressor_rank(name, 48)
        mono = (e48 <= e32 + 1e-4) and (e32 <= e16 + 1e-4) and (e16 <= e8 + 1e-4)
        diff_flag = "RANK-RESPONSIVE" if (abs(e8 - e48) > 1e-4) else "RANK-INVARIANT"
        mono_flag = "STRICT MONOTONIC" if mono else "NON-MONOTONIC"
        print(
            f"{name:25s} | r8={e8:.4f} | r16={e16:.4f} | r32={e32:.4f} | r48={e48:.4f} | {diff_flag:15s} | {mono_flag}"
        )
    except Exception as e:
        print(f"{name:25s} | ERROR: {e}")
