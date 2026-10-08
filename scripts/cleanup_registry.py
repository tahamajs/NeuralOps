"""Identify multiple registrations per class in neuralops/attentions."""
import ast
from collections import defaultdict
from pathlib import Path

def find_register_decorators(file_path: Path):
    content = file_path.read_text()
    try:
        tree = ast.parse(content)
    except Exception:
        return []
    
    decorators = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    # Check ATTENTION_REGISTRY.register
                    if isinstance(dec.func, ast.Attribute) and dec.func.attr == "register":
                        if dec.args and isinstance(dec.args[0], ast.Constant):
                            decorators.append((str(file_path), node.name, dec.args[0].value, dec.lineno))
    return decorators

all_decorators = []
for f in sorted(Path("neuralops/attentions/").rglob("*.py")):
    if "_legacy" in str(f) or "_archive" in str(f):
        continue
    all_decorators.extend(find_register_decorators(f))

class_registrations = defaultdict(list)
for f, cls, name, line in all_decorators:
    class_registrations[(f, cls)].append((name, line))

for (f, cls), regs in sorted(class_registrations.items()):
    if len(regs) > 1:
        print(f"\n{cls} ({f}):")
        for name, line in regs:
            print(f"  Line {line:4d} -> {name}")
