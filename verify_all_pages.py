#!/usr/bin/env python3
"""
Inspect and verify the compiled 810101504.pdf with extreme rigor.
Renders and extracts text from exact target pages.
"""

import fitz
import os
import re

PDF_PATH = '/Users/tahamajs/Documents/uni/Project/810101504.pdf'
OUT_DIR = '/Users/tahamajs/.gemini/antigravity-cli/brain/59971975-8575-4b8d-a3bc-7d962ee6381e'

def inspect_pdf():
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"Total Pages in 810101504.pdf: {total_pages}")
    
    # 1. Check Title Page (Page 0)
    p0_text = doc[0].get_text()
    print("\n--- Page 1 (Cover Page) ---")
    lines0 = [l.strip() for l in p0_text.splitlines() if l.strip()]
    for l in lines0:
        if 'SVD' in l or 'طراحی' in l:
            print("Title line:", l)
            
    doc[0].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, 'page_01_cover.png'))

    # 2. Check Arithmetic Intensity Page (Search for "Arithmetic Intensity" or "شدت محاسباتی")
    ai_page_idx = None
    for i in range(15, 45):
        txt = doc[i].get_text()
        if 'شدت محاسباتی' in txt or 'Arithmetic Intensity' in txt:
            ai_page_idx = i
            print(f"\n--- Found Arithmetic Intensity on PDF Page {i+1} ---")
            for l in txt.splitlines():
                if any(k in l for k in ['FLOPs', 'FLOP', 'AI', 'Traffic', 'شدت محاسباتی']):
                    print("Formula line:", l.strip())
            doc[i].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, f'page_{i+1}_arithmetic_intensity.png'))
            break

    # 3. Check Table of Contents (Pages 7 to 12)
    toc_pages = []
    for i in range(5, 15):
        txt = doc[i].get_text()
        if 'فهرست مطالب' in txt or 'فصل چهارم' in txt:
            toc_pages.append(i)
            print(f"\n--- TOC Page {i+1} ---")
            for l in txt.splitlines():
                if any(k in l for k in ['moba', 'infini', 'expert_choice', 'GLUE', 'GQA', 'NSA', 'Roofline']):
                    print("TOC line:", l.strip())
            doc[i].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, f'page_{i+1}_toc.png'))

    # 4. Check Module moba and subsequent modules in Chapter 4
    moba_page_idx = None
    for i in range(60, 200):
        txt = doc[i].get_text()
        if 'کالبدشکافی عمیق و تحلیل ساختاری ماژول' in txt and 'moba' in txt:
            moba_page_idx = i
            print(f"\n--- Found moba start on PDF Page {i+1} ---")
            doc[i].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, f'page_{i+1}_moba.png'))
            break

    # 5. Check References Section (Near end of document)
    ref_page_idx = None
    for i in range(total_pages - 30, total_pages):
        txt = doc[i].get_text()
        if 'مراجع' in txt or 'Vaswani' in txt or 'Attention is all you need' in txt:
            ref_page_idx = i
            print(f"\n--- Found References on PDF Page {i+1} ---")
            for l in txt.splitlines()[:20]:
                print("Ref line:", l.strip())
            doc[i].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, f'page_{i+1}_references.png'))
            break

    # 6. Check a Results Table (e.g. infini_attention or asvd_attention)
    for i in range(moba_page_idx or 100, min(total_pages, (moba_page_idx or 100) + 50)):
        txt = doc[i].get_text()
        if 'کارنامه تجربی و ارزیابی' in txt and ('Accuracy' in txt or 'F1-Score' in txt):
            print(f"\n--- Found Results Table on PDF Page {i+1} ---")
            for l in txt.splitlines()[:25]:
                if any(k in l for k in ['Accuracy', 'F1-Score', 'DistilBERT', 'RoBERTa', 'GPT-2', 'PASS']):
                    print("Table line:", l.strip())
            doc[i].get_pixmap(dpi=150).save(os.path.join(OUT_DIR, f'page_{i+1}_table.png'))
            break

    # 7. Check if NeuralOps exists in any text
    neuralops_pages = []
    for i in range(total_pages):
        txt = doc[i].get_text()
        if 'NeuralOps' in txt:
            neuralops_pages.append(i+1)
    print(f"\nNeuralOps occurrences in PDF: {len(neuralops_pages)} (Pages: {neuralops_pages[:10]})")

    # 8. Check if legacy_bert exists in any text
    legacy_pages = []
    for i in range(total_pages):
        txt = doc[i].get_text()
        if 'legacy_bert' in txt:
            legacy_pages.append(i+1)
    print(f"legacy_bert occurrences in PDF: {len(legacy_pages)} (Pages: {legacy_pages[:10]})")

    # 9. Check if A100 exists in any text
    a100_pages = []
    for i in range(total_pages):
        txt = doc[i].get_text()
        if 'A100' in txt:
            a100_pages.append(i+1)
    print(f"A100 occurrences in PDF: {len(a100_pages)} (Pages: {a100_pages[:10]})")

if __name__ == '__main__':
    inspect_pdf()
