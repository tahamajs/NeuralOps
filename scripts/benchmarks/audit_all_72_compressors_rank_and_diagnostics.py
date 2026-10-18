"""Audit all 72 compressors rank and diagnostics module.

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


def get_weights(model):
    """Execute get weights routine."""
    params = [p.detach().flatten() for p in model.parameters() if p.numel() > 0]
    return torch.cat(params) if params else torch.tensor([])


results = {}
print(f"Auditing {len(COMPRESSOR_REGISTRY)} registered compressors...")

for name in sorted(list(COMPRESSOR_REGISTRY.keys())):
    cls = COMPRESSOR_REGISTRY[name]
    entry = {
        "name": name,
        "status": "UNKNOWN",
        "rank_sensitive": False,
        "err_r8": None,
        "err_r32": None,
        "w_diff": None,
        "error_msg": None,
    }

    try:
        # Test rank 8
        m8 = TinyTransformer(64)
        m8_orig = copy.deepcopy(m8)
        x = torch.randn(2, 64)
        with torch.no_grad():
            y_orig = m8_orig(x)

        c8 = None
        for init_call in [
            lambda: cls(m8, rank=8, target_rank=8, fallback_to_original=False),
            lambda: cls(m8, rank=8, fallback_to_original=False),
            lambda: cls(m8, rank=8),
            lambda: cls(m8),
        ]:
            try:
                c8 = init_call()
                break
            except Exception:
                continue

        if c8 is None:
            raise RuntimeError("Could not instantiate compressor")

        if hasattr(c8, "compress"):
            res8 = c8.compress()
            if isinstance(res8, tuple):
                m8 = res8[0]
            elif isinstance(res8, nn.Module):
                m8 = res8
        elif isinstance(c8, nn.Module):
            m8 = c8

        with torch.no_grad():
            y8 = m8(x)
        err8 = (torch.norm(y8 - y_orig) / torch.norm(y_orig)).item()
        w8 = get_weights(m8)

        # Test rank 32
        m32 = TinyTransformer(64)
        c32 = None
        for init_call in [
            lambda: cls(m32, rank=32, target_rank=32, fallback_to_original=False),
            lambda: cls(m32, rank=32, fallback_to_original=False),
            lambda: cls(m32, rank=32),
            lambda: cls(m32),
        ]:
            try:
                c32 = init_call()
                break
            except Exception:
                continue

        if hasattr(c32, "compress"):
            res32 = c32.compress()
            if isinstance(res32, tuple):
                m32 = res32[0]
            elif isinstance(res32, nn.Module):
                m32 = res32
        elif isinstance(c32, nn.Module):
            m32 = c32

        with torch.no_grad():
            y32 = m32(x)
        err32 = (torch.norm(y32 - y_orig) / torch.norm(y_orig)).item()
        w32 = get_weights(m32)

        # Check divergence
        if w8.shape == w32.shape:
            w_diff = torch.norm(w8 - w32).item()
        else:
            w_diff = 999.0  # structural parameter count change

        entry["err_r8"] = round(err8, 4)
        entry["err_r32"] = round(err32, 4)
        entry["w_diff"] = round(w_diff, 4)

        if w_diff > 1e-4 or abs(err8 - err32) > 1e-4:
            entry["rank_sensitive"] = True
            entry["status"] = "VALID"
        elif err8 == 0.0 and err32 == 0.0:
            entry["status"] = "NO_OP"
        else:
            entry["status"] = "CONSTANT"

    except Exception as e:
        entry["status"] = "FAILED"
        entry["error_msg"] = str(e)[:120]

    results[name] = entry
    flag = "✓" if entry["rank_sensitive"] else ("✗ " + entry["status"])
    print(
        f"{name:32s} | status: {entry['status']:12s} | r8: {str(entry['err_r8']):7s} | r32: {str(entry['err_r32']):7s} | w_diff: {str(entry['w_diff']):7s} | {flag}"
    )

valid_count = sum(1 for v in results.values() if v["rank_sensitive"])
print(f"Total Valid Rank-Sensitive: {valid_count} / {len(results)}")
