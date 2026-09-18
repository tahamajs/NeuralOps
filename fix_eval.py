import sys

with open("D:/svd_project/Codes/framework/eval_int8_quantized.py", "r") as f:
    c = f.read()

c = c.replace('kwargs = {"text": batch[a]}', 'kwargs = {a: batch[a]}')
c = c.replace('if b is not None: kwargs["text_pair"] = batch[b]', 'if b is not None: kwargs[b] = batch[b]')

with open("D:/svd_project/Codes/framework/eval_int8_quantized.py", "w") as f:
    f.write(c)
