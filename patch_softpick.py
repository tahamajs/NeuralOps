import re, glob
files = glob.glob("Codes/framework/attentions/*.py")

softpick_code = """def softpick(x, dim: int = -1, temperature: float = 1.0):
    import torch.nn.functional as F
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-9)).to(orig_dtype)"""

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    if "def softpick(" not in content: continue
    
    # regex to find the softpick function exactly up to its return statement
    pattern = re.compile(r'def softpick\(.*?->\s*torch\.Tensor:.*?return[^\n]+\n', re.DOTALL)
    new_content = pattern.sub(softpick_code + "\n", content)
    
    # also replace -1e9 with -10000.0 for masked_fill
    new_content = new_content.replace("-1e9", "-10000.0")
    
    with open(f, 'w') as file:
        file.write(new_content)
    print(f"Patched {f}")
