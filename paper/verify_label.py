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
    '⌈': r'$\lceil$ ', '⌉': r'$\rceil$ ',
}

def clean_label_for_tikz(text):
    text = text.replace('\f', ' ').replace('\t', ' ').replace('\r', ' ').replace('\v', ' ').replace('\b', ' ')
    for k, v in unicode_ascii_map.items(): text = text.replace(k, ' ')
    text = text.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
    text = text.replace('·', r' $\cdot$ ')
    text = text.replace('^', r'\textasciicircum{}').replace('~', r'\textasciitilde{}')
    for ch in ['┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '─', '│', '►', '▼', '◄', '▲', '+---', '|---', '+', '──►', '───►', '-->', '==>', '->']:
        text = text.replace(ch, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text.replace('_', r'\_')
    for k, v in unicode_math_map.items(): text = text.replace(k, v)
    text = re.sub(r'\[([BNDHLdkvr\d\s,\*\+\\_]+)\]', r'$[\1]$', text)
    return text

s = "v Full SVD Factorization W = U Σ V^T; r = max(1, ⌈α · K⌉) [M, N]"
print(clean_label_for_tikz(s))
