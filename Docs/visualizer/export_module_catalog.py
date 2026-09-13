#!/usr/bin/env python3
"""
export_module_catalog.py
========================
Indexes all 240+ attention modules and compression engines in the repository,
extracting docstrings, categories, mathematical formulations, and metadata
into Docs/visualizer/modules-catalog.json.
"""

import os
import glob
import re
import json

def extract_module_info(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    is_attn = 'attentions' in file_path

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    doc_match = re.search(r'\"\"\"(.*?)\"\"\"', content, re.DOTALL)
    raw_doc = doc_match.group(1).strip() if doc_match else ''
    first_line = raw_doc.split('\n')[0].strip() if raw_doc else base_name.replace('_', ' ').title()
    first_line = re.sub(r'^[#\s\-*]+', '', first_line)

    md_dir = 'Docs/content/attentions' if is_attn else 'Docs/content/compressions'
    md_path = os.path.join(md_dir, f'{base_name}.md')
    math_formula = ''
    doc_markdown = ''

    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as mf:
            md_content = mf.read()
            doc_markdown = md_content[:600]
            math_match = re.search(r'\$\$(.*?)\$\$', md_content, re.DOTALL)
            if math_match:
                math_formula = math_match.group(1).strip()
            header_match = re.search(r'#\s*(.*?)\n', md_content)
            if header_match:
                title = header_match.group(1).strip()
                if len(title) > len(first_line) and not title.lower().startswith('module'):
                    first_line = title

    low = (base_name + ' ' + raw_doc).lower()
    if any(k in low for k in ['svd', 'low_rank', 'factor', 'rank', 'rpca', 'nmf', 'cur_linear', 'saes']):
        cat = 'Low-Rank SVD & Factorization'
        paradigm = 'svd'
        icon = '⚡'
    elif any(k in low for k in ['moe', 'expert', 'router', 'switch', 'routing', 'bam_umoe']):
        cat = 'Mixture of Experts & Routing'
        paradigm = 'moe'
        icon = '🔀'
    elif any(k in low for k in ['mamba', 'ssm', 'linear_attention', 'deltanet', 'hgrn', 'griffin', 'cosformer', 'decay', 'simple_ssm']):
        cat = 'Linear Attention & State-Space (SSM)'
        paradigm = 'ssm'
        icon = '🐍'
    elif any(k in low for k in ['mla', 'kv', 'latent', 'cache', 'kvtc', 'cachenotes', 'hybridkv']):
        cat = 'Latent & KV-Cache Compression'
        paradigm = 'mla'
        icon = '🧠'
    elif any(k in low for k in ['slice', 'prun', 'admm', 'prox', 'bonsai', 'flap', 'slim', 'sparsegpt']):
        cat = 'Structured Pruning & Channel Slicing'
        paradigm = 'pruning'
        icon = '✂️'
    elif any(k in low for k in ['quant', 'int4', 'fp8', 'int8', 'astro', 'coopq']):
        cat = 'Quantization & Mixed Precision'
        paradigm = 'quant'
        icon = '💎'
    else:
        cat = 'Dynamic & Efficient Attention'
        paradigm = 'sparse'
        icon = '🌟'

    # Clean description
    clean_desc = raw_doc.replace('\n', ' ')[:350]
    if not clean_desc or len(clean_desc) < 20:
        clean_desc = f'Production research module for {cat.lower()} across transformer architectures.'

    return {
        'id': base_name,
        'name': first_line[:65],
        'file': file_path,
        'type': 'Attention Module' if is_attn else 'Compression Engine',
        'category': cat,
        'paradigm': paradigm,
        'icon': icon,
        'description': clean_desc,
        'math': math_formula[:250],
        'markdown_preview': doc_markdown
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    attn_files = sorted(glob.glob('Codes/framework/attentions/*.py'))
    comp_files = sorted(glob.glob('Codes/framework/compression_engine/*.py'))

    all_modules = []
    seen = set()

    for p in attn_files:
        if not p.endswith('__init__.py'):
            info = extract_module_info(p)
            all_modules.append(info)
            seen.add(info['id'])

    for p in comp_files:
        if not p.endswith('__init__.py'):
            info = extract_module_info(p)
            if info['id'] not in seen:
                all_modules.append(info)
                seen.add(info['id'])

    out_path = 'Docs/visualizer/modules-catalog.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_count': len(all_modules),
            'generated_at': '2026-09-12',
            'modules': all_modules
        }, f, indent=2)

    print(f'[OK] Generated {out_path} with {len(all_modules)} modules across 7 research paradigms.')


if __name__ == '__main__':
    main()
