import re, glob
files = ["Codes/framework/attentions/leka_v1.py", "Codes/framework/attentions/leka_v2.py", "Codes/framework/attentions/leka_v3.py"]

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    # We will replace the poorly indented softpick block
    # We know it looks exactly like this in the broken files:
    broken = """    def softpick(x, dim: int = -1, temperature: float = 1.0):
    import torch.nn.functional as F
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-9)).to(orig_dtype)"""
    
    fixed = """    def softpick(x, dim: int = -1, temperature: float = 1.0):
        import torch.nn.functional as F
        orig_dtype = x.dtype
        x = x.float()
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        m_scaled = torch.clamp(-m / temperature, max=80.0)
        denom = e - torch.exp(m_scaled)
        p = F.relu(denom)
        return (p / (p.sum(dim=dim, keepdim=True) + 1e-9)).to(orig_dtype)"""
        
    content = content.replace(broken, fixed)
    
    with open(f, 'w') as file:
        file.write(content)
