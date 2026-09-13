#!/usr/bin/env python3
"""
Fix All LaTeX Escapes, Underscores in Captions/Titles, and Compile Cleanly with Exit Code 0.
"""

import os
import re
import glob
import json
import shutil
import subprocess

ROOT = '/Users/tahamajs/Documents/uni/Project'
THESIS_DIR = os.path.join(ROOT, 'thesis_final')
MODULES_DIR = os.path.join(THESIS_DIR, 'attentions_latex', 'individual_modules')

def fix_all_files():
    print("--- Fixing all underscores in captions, section titles, and tables ---")
    
    tex_files = glob.glob(os.path.join(MODULES_DIR, '*.tex'))
    print(f"Checking {len(tex_files)} module files...")
    
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Fix any \\_ to \_
        content = content.replace(r'\\_', r'\_')

        # 2. Fix unescaped underscores inside \caption{...}
        def escape_caption(m):
            cap_body = m.group(1)
            # escape any unescaped underscore in cap_body
            # But don't touch already escaped \_
            fixed_body = re.sub(r'(?<!\\)_', r'\_', cap_body)
            return f"\\caption{{{fixed_body}}}"

        content = re.sub(r'\\caption\{([^}]+)\}', escape_caption, content)

        # 3. Fix unescaped underscores inside \section{...} and \subsection{...}
        def escape_sec(m):
            prefix = m.group(1)
            body = m.group(2)
            fixed_body = re.sub(r'(?<!\\)_', r'\_', body)
            return f"{prefix}{{{fixed_body}}}"

        content = re.sub(r'(\\section\*?|\\subsection\*?)\{([^}]+)\}', escape_sec, content)

        # 4. Fix unescaped underscores inside \lr{...} when NOT in math mode
        def escape_lr(m):
            body = m.group(1)
            fixed_body = re.sub(r'(?<!\\)_', r'\_', body)
            return f"\\lr{{{fixed_body}}}"

        content = re.sub(r'\\lr\{([^}]+)\}', escape_lr, content)

        # 5. Fix double escaped \_ to \_
        content = content.replace(r'\\_', r'\_')

        # 6. Ensure % in \lr is escaped as \%
        content = re.sub(r'\\lr\{([0-9\.]+)\%?\}', r'\\lr{\1\\%}', content)
        content = content.replace(r'\\%', r'\%').replace(r'\\\%', r'\%')

        with open(tf, 'w', encoding='utf-8') as f:
            f.write(content)

    print("All module files fixed.")

    # Remove temporary LaTeX auxiliary files to clear corrupt .toc / .lof
    for ext in ['.toc', '.lof', '.lot', '.aux', '.out', '.log']:
        f = os.path.join(THESIS_DIR, f"master_thesis{ext}")
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed stale {f}")

    # Compile XeLaTeX
    print("\n--- Running Clean 2-Pass XeLaTeX Compilation ---")
    cmd = "xelatex -interaction=nonstopmode master_thesis.tex"
    print("Pass 1...")
    p1 = subprocess.run(cmd, shell=True, cwd=THESIS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Pass 1 returncode: {p1.returncode}")
    
    print("Pass 2...")
    p2 = subprocess.run(cmd, shell=True, cwd=THESIS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Pass 2 returncode: {p2.returncode}")

    # Check master_thesis.log for any errors
    log_path = os.path.join(THESIS_DIR, 'master_thesis.log')
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_lines = f.readlines()
        errors = [l.strip() for l in log_lines if l.startswith('! ')]
        print(f"Errors in log: {len(errors)}")
        if errors:
            print("First 5 errors:", errors[:5])

    pdf_path = os.path.join(THESIS_DIR, 'master_thesis.pdf')
    target_pdf = '/Users/tahamajs/Documents/uni/Project/810101504.pdf'
    if os.path.exists(pdf_path):
        shutil.copyfile(pdf_path, target_pdf)
        sz = os.path.getsize(target_pdf) / (1024*1024)
        print(f"Synced PDF to {target_pdf} ({sz:.1f} MB)")

if __name__ == '__main__':
    fix_all_files()
