with open("D:/svd_project/Codes/framework/run_phase2.py", "r") as f:
    c = f.read()

import re
c = re.sub(r'load_model\(.*?\)', 'AutoModelForSequenceClassification.from_pretrained(model_path).to(device)', c)

with open("D:/svd_project/Codes/framework/run_phase2.py", "w") as f:
    f.write(c)
