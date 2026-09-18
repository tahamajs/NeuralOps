with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "r") as f:
    c = f.read()

import re
# Replace the configs = list(...) line
new_configs = """
    combo_booleans = [(True, True, True), (True, False, False), (False, True, True)]
    configs = []
    for t in taus:
        for r in ranks:
            for attn, fisher, recal in combo_booleans:
                configs.append((t, r, attn, fisher, recal))
"""
c = re.sub(r'configs = list\(itertools.product\(.*?\)\)', new_configs.strip(), c)

with open("D:/svd_project/Codes/framework/run_a3_sweep.py", "w") as f:
    f.write(c)
