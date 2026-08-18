import re

unicode_ascii_map = {
    '─': ' ', '│': ' ', '┌': ' ', '┐': ' ', '└': ' ', '┘': ' ', 
    '├': ' ', '┤': ' ', '┬': ' ', '┴': ' ', '┼': ' ',
    '►': ' ', '▼': ' ', '◄': ' ', '✅': '[X]', '❌': '[ ]',
    '🚀': '*', '🔥': '*', '🔄': '*', '\u0007': '',
}
unicode_math_map = {
    'Δ': r'$\Delta$ ', 'Σ': r'$\Sigma$ ', 'α': r'$\alpha$ ', 'ε': r'$\epsilon$ ', 'σ': r'$\sigma$ ', 'τ': r'$\tau$ ',
    '⁸': r'$^8$', '₁': r'$_1$', '₂': r'$_2$', '₃': r'$_3$', 'ℝ': r'$\mathbb{R}$ ', '⇒': r'$\Rightarrow$ ',
    '∈': r'$\in$ ', '√': r'$\sqrt{}$', '≈': r'$\approx$ ', '≤': r'$\le$ ', '≥': r'$\ge$ ', '≪': r'$\ll$ ',
    '⊗': r'$\otimes$ ', '⊙': r'$\odot$ ', '⌊': r'$\lfloor$ ', '⌋': r'$\rfloor$ ', '²': r'$^2$', '³': r'$^3$',
    '⌈': r'$\lceil$ ', '⌉': r'$\rceil$ ', '·': r'$\cdot$ ',
}

def clean_label_for_tikz(text):
    text = text.replace('\f', ' ').replace('\t', ' ').replace('\r', ' ').replace('\v', ' ').replace('\b', ' ')
    for k, v in unicode_ascii_map.items(): text = text.replace(k, ' ')
    for k, v in unicode_math_map.items(): text = text.replace(k, v)
    text = text.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
    text = text.replace('^', r'\textasciicircum{}').replace('~', r'\textasciitilde{}')
    for ch in ['┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '─', '│', '►', '▼', '◄', '▲', '+---', '|---', '+', '──►', '───►', '-->', '==>', '->']:
        text = text.replace(ch, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    math_placeholders = []
    def save_math(m):
        math_placeholders.append(m.group(0))
        return f"MPHTIKZ{len(math_placeholders)-1}END"
        
    text = re.sub(r'\$(.*?)\$', save_math, text)
    text = re.sub(r'(\\widetilde\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\hat\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\mathbf\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\mathcal\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\Sigma|\\Delta|\\alpha|\\epsilon|\\sigma|\\tau|\\mu|\\theta|\\gamma|\\phi|\\psi|\\lambda|\\omega|\\odot|\\otimes)', r'$\1$', text)
    text = re.sub(r'\$(.*?)\$', save_math, text)
    
    text = text.replace('_', r'\_')
    
    for i, m in enumerate(math_placeholders):
        text = text.replace(f"MPHTIKZ{i}END", m)
        
    text = re.sub(r'\[([a-zA-Z\d\s,\*\+\\_]+)\]', r'$[\1]$', text)
    return text

s = "Input X [B, N, D] - Compute Saliency Scores in SRAM Bitonic Sort & Gather \\widetilde{X} and \\Sigma \\odot \\mu with bias_val"
print(clean_label_for_tikz(s))
