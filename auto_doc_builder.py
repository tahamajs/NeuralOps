import os
import ast
import glob

def parse_module(py_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    registry_names = []
    classes = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == 'register':
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            registry_names.append(decorator.args[0].value)
            
            init_args = []
            docstring = ast.get_docstring(node)
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    for arg in child.args.args:
                        if arg.arg != 'self':
                            init_args.append(arg.arg)
            classes.append({
                'name': node.name,
                'docstring': docstring,
                'args': init_args
            })
            
    return registry_names, classes

def build_doc(py_path, out_dir, category):
    base_name = os.path.basename(py_path).replace('.py', '')
    md_path = os.path.join(out_dir, f"{base_name}.md")
    
    names, classes = parse_module(py_path)
    if not names:
        names = [base_name]
        
    title_name = names[0]
    
    # Check if md file already exists and is not a stub (e.g. > 50 lines)
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > 50:
            return False # Skip already comprehensive files
            
    # Generate new comprehensive content
    content = f"# {title_name.replace('_', ' ').title()}\n\n"
    content += f"## Overview\n"
    
    for c in classes:
        if c['docstring']:
            content += f"{c['docstring']}\n\n"
        
    content += "## Configuration & Hyperparameters\n\n"
    content += "| Parameter | Description |\n"
    content += "|-----------|-------------|\n"
    for c in classes:
        for arg in c['args']:
            content += f"| `{arg}` | TODO: Description for {arg} |\n"
            
    content += "\n## Usage Examples\n\n```python\n"
    content += f"from Codes.framework.{category} import {classes[0]['name'] if classes else 'Module'}\n"
    content += "```\n\n"
    content += "## Benchmark Results\n\n"
    content += "| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |\n"
    content += "|--------|----------|----------|-----------|-------------------|\n"
    content += "| Pending| Pending  | Pending  | Pending   | Pending           |\n\n"
    
    content += "## Limitations & Future Work\n\nTODO: Identify specific limitations and edge cases for this module.\n"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

generated = 0
for py in glob.glob("Codes/framework/attentions/*.py"):
    if build_doc(py, "Docs/content/attentions", "attentions"):
        generated += 1

for py in glob.glob("Codes/framework/compression_engine/*.py"):
    if build_doc(py, "Docs/content/compressions", "compression_engine"):
        generated += 1

print(f"Generated or expanded {generated} documentation files.")
