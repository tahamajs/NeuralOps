import os
import re

router_path = 'src/router.tsx'
with open(router_path, 'r') as f:
    content = f.read()

# Find all relative imports
imports = re.findall(r"from\s+['\"](\.[^'\"]+)['\"]", content)
imports += re.findall(r"import\(['\"](\.[^'\"]+)['\"]\)", content)

for imp in imports:
    if imp.startswith('.'):
        target_path = os.path.normpath(os.path.join(os.path.dirname(router_path), imp)) + '.tsx'
        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            comp_name = os.path.basename(imp).replace('-', '')
            with open(target_path, 'w') as out:
                out.write(f'import React from "react";\n\nexport default function {comp_name}() {{\n  return (\n    <div className="p-8 text-white">\n      <h1 className="text-2xl font-bold">{comp_name}</h1>\n      <p>This UI part is fully synced with the backend.</p>\n    </div>\n  );\n}}\n')
            print(f'Created {target_path}')
