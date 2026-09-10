import re
from pathlib import Path

attentions_dir = Path("Codes/framework/attentions")
py_files = list(attentions_dir.glob("*.py"))

bug_a_count = 0
bug_b_count = 0

for file_path in py_files:
    text = file_path.read_text(encoding="utf-8")
    modified = False

    # Bug A: Fix layer.attention = new_attn / new_block / test_module
    # Pattern 1:
    # if hasattr(layer, "attention"):
    #     layer.attention = <var>
    pattern_a = re.compile(
        r'(\n([ \t]+)if hasattr\(layer, ["\']attention["\']\):\s*\n[ \t]+layer\.attention = (\w+))',
        re.MULTILINE
    )

    def replace_a(match):
        full_match = match.group(1)
        indent = match.group(2)
        var_name = match.group(3)
        replacement = (
            f"\n{indent}if hasattr(layer, \"attention\"):\n"
            f"{indent}    if hasattr(layer.attention, \"self\"):\n"
            f"{indent}        layer.attention.self = {var_name}\n"
            f"{indent}    else:\n"
            f"{indent}        layer.attention = {var_name}"
        )
        return replacement

    if "layer.attention =" in text and "if hasattr(layer.attention, \"self\")" not in text:
        new_text = pattern_a.sub(replace_a, text)
        if new_text != text:
            text = new_text
            modified = True
            bug_a_count += 1

    # Bug B: Causal Masking in kernel_linear_attention
    if "def kernel_linear_attention(" in text and "is_causal" not in text:
        # Update function signature
        text = text.replace(
            "def kernel_linear_attention(\n    Q: torch.Tensor,\n    K: torch.Tensor,\n    V: torch.Tensor,\n    rff_proj: torch.Tensor,\n    use_zero_sum: bool = True,\n    max_clamp: float = 11.0,\n) -> torch.Tensor:",
            "def kernel_linear_attention(\n    Q: torch.Tensor,\n    K: torch.Tensor,\n    V: torch.Tensor,\n    rff_proj: torch.Tensor,\n    use_zero_sum: bool = True,\n    max_clamp: float = 11.0,\n    is_causal: bool = False,\n) -> torch.Tensor:"
        )
        # Update forward pass inside kernel_linear_attention
        old_einsum_block = """    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
    K_sum = K_feat.sum(dim=2)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)

    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)
    ctx = ctx / (denom + 1e-5)"""

        new_einsum_block = """    if is_causal:
        kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_feat, V_f)
        KV_cum = torch.cumsum(kv_outer, dim=2)
        z_cum = torch.cumsum(K_feat, dim=2)
        num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_cum)
        denom = torch.einsum("bhnf,bhnf->bhn", Q_feat, z_cum).unsqueeze(-1).clamp(min=1e-5)
        ctx = num / denom
    else:
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / (denom + 1e-5)"""

        if old_einsum_block in text:
            text = text.replace(old_einsum_block, new_einsum_block)
            # Pass is_causal=self.is_decoder to kernel_linear_attention in forward()
            text = text.replace(
                "ctx_global = kernel_linear_attention(\n            Q_global, K_global, V_global, self.rff_proj, self.cfg.use_zero_sum_keys\n        )",
                "ctx_global = kernel_linear_attention(\n            Q_global, K_global, V_global, self.rff_proj, self.cfg.use_zero_sum_keys, is_causal=getattr(self, 'is_decoder', False)\n        )"
            )
            modified = True
            bug_b_count += 1

    if modified:
        file_path.write_text(text, encoding="utf-8")

print(f"Fixed Bug A in {bug_a_count} files.")
print(f"Fixed Bug B in {bug_b_count} files.")
