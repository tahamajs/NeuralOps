import re
from pathlib import Path

for f in Path('D:/SVD_Project').glob('run_all_4_*.py'):
    content = f.read_text(encoding='utf-8')
    match = re.search(r'unified_([^\\]+)\\distilbert', content)
    if match:
        dir_name = match.group(1)
        expected = f.name[10:-3]
        if dir_name != expected:
            print(f"{f.name}: modifying output directories from {dir_name} to {expected}")
            new_content = content.replace(f"unified_{dir_name}", f"unified_{expected}")
            # Also need to fix `--modules` argument if it is wrong
            new_content = new_content.replace(f"--modules\", \"full,{dir_name}", f"--modules\", \"full,{expected}")
            f.write_text(new_content, encoding='utf-8')
