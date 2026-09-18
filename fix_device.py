with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "r") as f:
    c = f.read()
c = c.replace('compressed_model, metrics = comp.compress()', 'compressed_model, metrics = comp.compress()\n            compressed_model = compressed_model.to(device)')
with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "w") as f:
    f.write(c)

with open("D:/svd_project/Codes/framework/run_phase2.py", "r") as f:
    c = f.read()
c = c.replace('model, _ = apply_lora(model, r=8, alpha=32)', 'model, _ = apply_lora(model, r=8, alpha=32)\n    model = model.to(device)')
with open("D:/svd_project/Codes/framework/run_phase2.py", "w") as f:
    f.write(c)
