import glob
import os
import re
from pathlib import Path

runners = glob.glob('Codes/runners/run_all_4_*.py')
print(f'Processing {len(runners)} runner files for cross-platform portability...')

fixed_count = 0
for r in runners:
    with open(r, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    modified = False

    # 1. Ensure Path import
    if 'from pathlib import Path' not in content:
        content = 'from pathlib import Path\n' + content
        modified = True

    # 2. Fix os.chdir(r"PROJECT_ROOT")
    pattern_chdir = r'os\.chdir\(r?["\']PROJECT_ROOT["\']\)'
    if re.search(pattern_chdir, content):
        replacement = (
            'HERE = Path(__file__).resolve().parent\n'
            'PROJECT_ROOT = HERE.parent.parent\n'
            'if os.path.exists(r"PROJECT_ROOT"):\n'
            '    os.chdir(r"PROJECT_ROOT")\n'
            'else:\n'
            '    os.chdir(PROJECT_ROOT)\n'
            'RESULTS_DIR = Path("results").resolve()'
        )
        content = re.sub(pattern_chdir, replacement, content)
        modified = True

    # 3. Fix sys.executable literal
    if '"sys.executable"' in content:
        content = content.replace('r\'"sys.executable"', 'f\'"{sys.executable}"')
        content = content.replace('r"""sys.executable"""', 'f\'"{sys.executable}"')
        content = content.replace('r\'"sys.executable" -u', 'f\'"{sys.executable}" -u')
        content = content.replace('r\'"sys.executable"', 'f\'"{sys.executable}"')
        content = content.replace('r\'\"sys.executable\"', 'f\'"{sys.executable}"')
        modified = True

    # 4. Fix summary and output paths
    target_d = r"D:\SVD_Project\results"
    target_d_esc = r"D:\\SVD_Project\\results"
    if target_d in content or target_d_esc in content:
        content = content.replace(target_d, str(Path("results").resolve()))
        content = content.replace(target_d_esc, str(Path("results").resolve()))
        modified = True

    if modified:
        with open(r, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed_count += 1

print(f'Successfully made {fixed_count} runner scripts cross-platform and portable!')
