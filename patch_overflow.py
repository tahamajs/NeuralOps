import os, glob, re

attn_files = glob.glob("Codes/framework/attentions/*.py")

safe_softpick = """def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = torch.nn.functional.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-9)).to(orig_dtype)"""

for f in attn_files:
    with open(f, "r") as file:
        content = file.read()
    
    modified = False
    
    # Simple regex to replace existing softpick if it matches standard patterns
    if "def softpick" in content:
        # Regex to capture the whole function block until the next top-level definition or empty lines + class
        pattern = re.compile(r'def softpick\(.*?\).*?return.*?\n(?=\n*(def |class |@))', re.DOTALL)
        # If it doesn't match perfectly, we might need to manually handle it. Let's see if we can just patch it easily.
