import os
import re

files = [
    "Mamba.py", "mamba_inject.py", "mamba_svd_flash.py", "spectral_mamba.py", "upi_stuffed_mamba.py",
    "simple_ssm.py", "decay_linear_attention.py", "gated_linear_attention.py",
    "factorised_matrix_state.py", "SOAFusedAttention.py", "soa_attn.py",
    "leka_v1.py", "leka_v2.py", "leka_v3.py"
]

dir_path = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/"

for f in files:
    try:
        with open(os.path.join(dir_path, f), "r") as fp:
            content = fp.read()
            print(f"--- {f} ---")
            
            # check for decay
            if "decay" in content.lower():
                print("Has decay")
            if "fft" in content.lower():
                print("Has fft")
            if "gate" in content.lower():
                print("Has gate")
            if "distilbert" in content.lower() or "w_q" in content.lower():
                print("Has weight transfer hints")
    except Exception as e:
        print(f"Error reading {f}: {e}")
