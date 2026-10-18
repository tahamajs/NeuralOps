import re
import neuralops.attentions
from neuralops.core.registry import ATTENTION_REGISTRY

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

table_lines = [l for l in readme.splitlines() if re.match(r"^\|\s*\d+\s*\|\s*`[a-zA-Z0-9_]+`", l)]
table_names = [re.search(r"`([a-zA-Z0-9_]+)`", l).group(1) for l in table_lines]
print(f"Found {len(table_names)} names in README table.")

reg_names = set(ATTENTION_REGISTRY.keys())
print(f"Registry has {len(reg_names)} names.")

missing_from_reg = [n for n in table_names if n not in reg_names]
print("In README table but missing from ATTENTION_REGISTRY:", missing_from_reg)

extra_in_reg = [n for n in reg_names if n not in table_names]
print("In ATTENTION_REGISTRY but missing from README table:", extra_in_reg)
