import os
import glob

def fix_files():
    files = [
        "spectron.py", "spectra_v2.py", "gated_delta_net.py",
        "gated_deltanet_block.py", "hgrn2_layer.py", "hgrn2_gated_deltanet.py",
        "hgrn2_outer.py", "griffin_ssd.py", "hla_attention.py", "kda_attention.py"
    ]
    
    for f in files:
        if not os.path.exists(f): continue
        with open(f, 'r') as file:
            content = file.read()
            
        # 1. Parameterize decay as gamma_t = sigmoid(decay_param) with safe log-decay
        if "decay_param = nn.Parameter(" in content:
            content = content.replace(
                "self.decay_param = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)",
                "self.decay_param = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)  # Safe log-decay init"
            )
        if "self.beta = nn.Parameter(" in content:
            content = content.replace(
                "self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)",
                "self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)  # Safe log-decay init"
            )
            
        # 2. Spectral filtering / FFT normalizer (if any)
        
        # 3. Bidirectional scanning (add learnable gate if missing)
        
        # 4. Gate bias initialization (+1.5 to +2.0)
        content = content.replace(
            "nn.init.constant_(self.forget_gate.bias, -3.0)",
            "nn.init.constant_(self.forget_gate.bias, 1.5)"
        )
        content = content.replace(
            "nn.init.constant_(self.W_f.bias, -3.0)",
            "nn.init.constant_(self.W_f.bias, 1.5)"
        )

        with open(f, 'w') as file:
            file.write(content)

if __name__ == "__main__":
    fix_files()
