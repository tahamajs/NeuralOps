with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "r") as f:
    c = f.read()

c = c.replace('from framework.eval_all_models_datasets import evaluate_comprehensive', 'from framework.run_quality_sweep import evaluate_comprehensive')

with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "w") as f:
    f.write(c)
