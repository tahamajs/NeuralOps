with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "r") as f:
    c = f.read()

c = c.replace('load_model(model_path, device, dtype=torch.float32)', 'load_model(model_path, 2, device=device, dtype=torch.float32)')

with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "w") as f:
    f.write(c)
