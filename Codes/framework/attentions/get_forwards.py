import os
files = [
    "spectron.py", "spectra_v2.py", "gated_delta_net.py",
    "gated_deltanet_block.py", "hgrn2_layer.py", "hgrn2_gated_deltanet.py",
    "hgrn2_outer.py", "griffin_ssd.py", "hla_attention.py", "kda_attention.py"
]
for f in files:
    print(f"--- {f} ---")
    os.system(f"grep -n -C 5 'def forward' {f}")
