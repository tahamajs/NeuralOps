import json
import os
import ast

def process_all():
    with open('/tmp/attn_dedup_map.json', 'r') as f:
        dedup_map = json.load(f)

    with open('neuralops/core/deprecated_aliases.json', 'r') as f:
        aliases = json.load(f)

    for filepath, entries in dedup_map.items():
        filename = os.path.basename(filepath).replace('.py', '')
        keys = [e[0] for e in entries]
        canonical = keys[0]
        for k in keys:
            if k == filename:
                canonical = k
                break
        
        with open(filepath, 'r') as f:
            lines = f.readlines()

        source = "".join(lines)
        tree = ast.parse(source)
        
        decorators_to_remove = set()
        canonical_found = False
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        if dec.func.value.id == 'ATTENTION_REGISTRY' and dec.func.attr == 'register':
                            if dec.args and isinstance(dec.args[0], ast.Constant):
                                key = dec.args[0].value
                                if key == canonical and not canonical_found:
                                    canonical_found = True
                                else:
                                    start = dec.lineno
                                    end = getattr(dec, "end_lineno", dec.lineno)
                                    for i in range(start, end + 1):
                                        decorators_to_remove.add(i)
                                    if key != canonical:
                                        aliases[key] = canonical

        new_lines = []
        for i, line in enumerate(lines):
            # AST lineno is 1-indexed, enumerate is 0-indexed
            if (i + 1) not in decorators_to_remove:
                new_lines.append(line)

        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
    with open('neuralops/core/deprecated_aliases.json', 'w') as f:
        json.dump(aliases, f, indent=4, sort_keys=True)
        f.write("\n")

process_all()
print("Done processing dedup")
