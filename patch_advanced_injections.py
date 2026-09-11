#!/usr/bin/env python3
"""
patch_advanced_injections.py
Fixes remaining RoBERTa LayerNorm preservation bugs across framework attention/compression modules.
Ensures that whenever a module replaces attention on a layer:
- If layer.attention has a .self attribute (RoBERTa/BERT pattern), it replaces layer.attention.self
- Never destroys layer.attention.output (dense + LayerNorm)
"""

import os, glob, re
from pathlib import Path

ROBERTA_SAFE_BLOCK = """        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, {VAR})
                return {VAR}
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = {VAR}
            else:
                layer.attention = {VAR}
            return {VAR}
        layer.attention = {VAR}
        return {VAR}"""

files_to_check = glob.glob("Codes/framework/attentions/*.py") + glob.glob("Codes/framework/compression_engine/*.py")
patched = 0

for file_path in files_to_check:
    content = Path(file_path).read_text(encoding="utf-8")
    original = content

    # Check for:
    # for attr in ("attention", "self_attn", "attn", "self"):
    #     if hasattr(layer, attr):
    #         setattr(layer, attr, <var>)
    #         return <var>
    # layer.attention = <var>
    pattern1 = re.compile(
        r'for attr in \((["\']attention["\'],\s*["\']self_attn["\'],\s*["\']attn["\'],\s*["\']self["\']|["\']self_attn["\'],\s*["\']attn["\'],\s*["\']self["\'])\):\s*\n'
        r'\s*if hasattr\(layer, attr\):\s*\n'
        r'\s*setattr\(layer, attr, (\w+)\)\s*\n'
        r'\s*return (\w+)\s*\n\s*layer\.attention = \w+\s*\n\s*return \w+',
        re.MULTILINE
    )

    def sub1(match):
        var_name = match.group(2)
        return ROBERTA_SAFE_BLOCK.format(VAR=var_name).strip()

    new_content = pattern1.sub(sub1, content)

    # Check for isolated:
    # layer.attention = new_attn
    # return new_attn
    # when not preceded by hasattr(layer.attention, "self")
    if file_path.endswith("lora_attention.py"):
        old_block = "        layer.attention = new_attn\n        return new_attn"
        new_block = (
            "        if hasattr(layer, 'attention'):\n"
            "            if hasattr(layer.attention, 'self'):\n"
            "                layer.attention.self = new_attn\n"
            "            else:\n"
            "                layer.attention = new_attn\n"
            "        elif hasattr(layer, 'self'):\n"
            "            layer.self = new_attn\n"
            "        elif hasattr(layer, 'attn'):\n"
            "            layer.attn = new_attn\n"
            "        else:\n"
            "            layer.attention = new_attn\n"
            "        return new_attn"
        )
        new_content = new_content.replace(old_block, new_block)

    # Check for spectraformer_attention.py pattern:
    if file_path.endswith("spectraformer_attention.py"):
        old_sf = (
            "        if layer is not None:\n"
            "            if hasattr(layer, \"attention\"):\n"
            "                layer.attention = new_attn\n"
            "            elif hasattr(layer, \"self\"):\n"
            "                layer.self = new_attn\n"
            "            elif hasattr(layer, \"attn\"):\n"
            "                layer.attn = new_attn\n"
            "            else:\n"
            "                layer.attention = new_attn"
        )
        new_sf = (
            "        if layer is not None:\n"
            "            if hasattr(layer, \"attention\"):\n"
            "                if hasattr(layer.attention, \"self\"):\n"
            "                    layer.attention.self = new_attn\n"
            "                else:\n"
            "                    layer.attention = new_attn\n"
            "            elif hasattr(layer, \"self\"):\n"
            "                layer.self = new_attn\n"
            "            elif hasattr(layer, \"attn\"):\n"
            "                layer.attn = new_attn\n"
            "            else:\n"
            "                layer.attention = new_attn"
        )
        new_content = new_content.replace(old_sf, new_sf)

    # Check for orion / spectral_mamba / adasplash:
    pattern2 = re.compile(
        r'(\s+)elif hasattr\(layer, ["\']self["\']\):\s*\n\s+layer\.self = (\w+)\s*\n\s+elif hasattr\(layer, ["\']attn["\']\):\s*\n\s+layer\.attn = \w+\s*\n\s+else:\s*\n\s+layer\.attention = \w+',
        re.MULTILINE
    )
    def sub2(match):
        indent = match.group(1)
        var = match.group(2)
        return (
            f"{indent}elif hasattr(layer, 'attention'):\n"
            f"{indent}    if hasattr(layer.attention, 'self'):\n"
            f"{indent}        layer.attention.self = {var}\n"
            f"{indent}    else:\n"
            f"{indent}        layer.attention = {var}\n"
            f"{indent}elif hasattr(layer, 'self'):\n"
            f"{indent}    layer.self = {var}\n"
            f"{indent}elif hasattr(layer, 'attn'):\n"
            f"{indent}    layer.attn = {var}\n"
            f"{indent}else:\n"
            f"{indent}    layer.attention = {var}"
        )
    new_content = pattern2.sub(sub2, new_content)

    if new_content != original:
        Path(file_path).write_text(new_content, encoding="utf-8")
        patched += 1
        print(f"Patched: {file_path}")

print(f"Total files patched: {patched}")
