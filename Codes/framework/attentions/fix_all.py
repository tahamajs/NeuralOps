import os
import re

files_to_fix = [
    "Mamba.py", "mamba_inject.py", "mamba_svd_flash.py", "spectral_mamba.py", "upi_stuffed_mamba.py",
    "simple_ssm.py", "decay_linear_attention.py", "gated_linear_attention.py",
    "factorised_matrix_state.py", "SOAFusedAttention.py", "soa_attn.py",
    "leka_v1.py", "leka_v2.py", "leka_v3.py"
]

dir_path = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/"

def modify_content(content, filename):
    orig_content = content
    
    # 1. Decay Parameterization
    # Match: lambdas = torch.exp(-F.softplus(decay_logits))
    content = re.sub(
        r"(lambdas|decay_rates)\s*=\s*torch\.exp\(-F\.softplus\((.*?)\)\)",
        r"\1 = torch.sigmoid(\2)",
        content
    )
    # Also fix any initialization from -3.0 to something positive for sigmoid
    content = re.sub(
        r"nn\.init\.constant_\(self\.decay_gate\.bias,\s*-3\.0\)",
        r"nn.init.constant_(self.decay_gate.bias, 2.0)",
        content
    )

    # 2. Spectral filtering / FFT normalizer: clamp frequency eigenvalues with epsilon = 1e-6
    # SVD projection
    content = re.sub(
        r"U,\s*S,\s*Vt\s*=\s*torch\.linalg\.svd\((.*?)\)",
        r"U, S, Vt = torch.linalg.svd(\1)\n                S = S.clamp(min=1e-6)",
        content
    )
    # Mamba SVD flash Sigma clamping
    content = re.sub(
        r"(Sigma\s*=\s*torch\.sigmoid\(.*?\))\s*\*\s*([0-9.]+)",
        r"\1.clamp(min=1e-6) * \2",
        content
    )
    
    # If there is FFT (just in case)
    content = re.sub(
        r"(torch\.fft\.irfft\(.*?)(,\s*n=\w+)?(\))",
        r"\1.clamp(min=1e-6)\2\3",
        content
    )

    # 4. Gate bias initialization: initialize output gates with positive bias (+1.5 to +2.0).
    # Find any nn.init.constant_(self.*gate.bias, <negative or small>)
    def gate_bias_replacer(match):
        return f"nn.init.constant_({match.group(1)}, 1.5)"
    
    content = re.sub(
        r"nn\.init\.constant_\((self\.[a-zA-Z_]*gate[a-zA-Z_]*\.bias),\s*[-0-9.]+\)",
        gate_bias_replacer,
        content
    )
    
    # 5. Weight transfer: DistilBERT self-attention weights (W_q, W_k, W_v, W_o)
    # Add _copy_pretrained if missing and if class inherits from BaseAttention or has q_lin, etc.
    if "def _copy_pretrained" not in content and "def from_distilbert_pretrained" not in content:
        if "self.query" in content or "self.q_lin" in content or "self.q_proj" in content:
            # We'll just append a generic copy_pretrained method if it seems like a class
            pass

    return content

for filename in files_to_fix:
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            new_content = modify_content(content, filename)
            if new_content != content:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"Fixed: {filename}")
        except Exception as e:
            print(f"Error on {filename}: {e}")

