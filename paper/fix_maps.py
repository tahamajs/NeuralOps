import re

with open('clean_latex_injection.py', 'r') as f:
    text = f.read()

def fix_map(m):
    return m.group(0).replace(r"'$", r"'$").replace(r"$ '", r"$'")

# This is getting complicated. I'll just replace them literally.
# unicode_math_map:
text = text.replace(r"r'$\Delta$ '", r"r'$\Delta$'")
text = text.replace(r"r'$\Sigma$ '", r"r'$\Sigma$'")
text = text.replace(r"r'$\alpha$ '", r"r'$\alpha$'")
text = text.replace(r"r'$\epsilon$ '", r"r'$\epsilon$'")
text = text.replace(r"r'$\sigma$ '", r"r'$\sigma$'")
text = text.replace(r"r'$\tau$ '", r"r'$\tau$'")
text = text.replace(r"r'$\mathbb{R}$ '", r"r'$\mathbb{R}$'")
text = text.replace(r"r'$\Rightarrow$ '", r"r'$\Rightarrow$'")
text = text.replace(r"r'$\in$ '", r"r'$\in$'")
text = text.replace(r"r'$\approx$ '", r"r'$\approx$'")
text = text.replace(r"r'$\le$ '", r"r'$\le$'")
text = text.replace(r"r'$\ge$ '", r"r'$\ge$'")
text = text.replace(r"r'$\ll$ '", r"r'$\ll$'")
text = text.replace(r"r'$\otimes$ '", r"r'$\otimes$'")
text = text.replace(r"r'$\odot$ '", r"r'$\odot$'")
text = text.replace(r"r'$\lfloor$ '", r"r'$\lfloor$'")
text = text.replace(r"r'$\rfloor$ '", r"r'$\rfloor$'")
text = text.replace(r"r'$\lceil$ '", r"r'$\lceil$'")
text = text.replace(r"r'$\rceil$ '", r"r'$\rceil$'")
text = text.replace(r"r'$\cdot$ '", r"r'$\cdot$'")

text = text.replace(r"r'$\theta$ '", r"r'$\theta$'")
text = text.replace(r"r'$\phi$ '", r"r'$\phi$'")
text = text.replace(r"r'$\psi$ '", r"r'$\psi$'")
text = text.replace(r"r'$\lambda$ '", r"r'$\lambda$'")
text = text.replace(r"r'$\beta$ '", r"r'$\beta$'")
text = text.replace(r"r'$\gamma$ '", r"r'$\gamma$'")
text = text.replace(r"r'$\pi$ '", r"r'$\pi$'")
text = text.replace(r"r'$\Omega$ '", r"r'$\Omega$'")
text = text.replace(r"r'$\eta$ '", r"r'$\eta$'")
text = text.replace(r"r'$\rightarrow$ '", r"r'$\rightarrow$'")
text = text.replace(r"r'$\leftarrow$ '", r"r'$\leftarrow$'")
text = text.replace(r"r'$\uparrow$ '", r"r'$\uparrow$'")
text = text.replace(r"r'$\downarrow$ '", r"r'$\downarrow$'")
text = text.replace(r"r'$\bullet$ '", r"r'$\bullet$'")
text = text.replace(r"r'$\dots$ '", r"r'$\dots$'")
text = text.replace(r"r'$\times$ '", r"r'$\times$'")
text = text.replace(r"r'$\div$ '", r"r'$\div$'")
text = text.replace(r"r'$\pm$ '", r"r'$\pm$'")
text = text.replace(r"r'$\neq$ '", r"r'$\neq$'")
text = text.replace(r"r'$\sum$ '", r"r'$\sum$'")
text = text.replace(r"r'$\prod$ '", r"r'$\prod$'")
text = text.replace(r"r'$\int$ '", r"r'$\int$'")
text = text.replace(r"r'$\partial$ '", r"r'$\partial$'")
text = text.replace(r"r'$\nabla$ '", r"r'$\nabla$'")
text = text.replace(r"r'$\rho$ '", r"r'$\rho$'")

with open('clean_latex_injection.py', 'w') as f:
    f.write(text)
