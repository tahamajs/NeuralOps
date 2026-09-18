with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "r") as f:
    c = f.read()

import re
c = re.sub(r'load_model\(.*?\)', 'AutoModelForSequenceClassification.from_pretrained(model_path).to(device)', c)

c = c.replace('from framework.models_conf.core_loaders import load_model', 'from transformers import AutoModelForSequenceClassification')

with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "w") as f:
    f.write(c)
