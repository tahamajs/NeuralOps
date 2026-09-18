with open("D:/svd_project/Codes/framework/eval_int8.py", "r") as f:
    c = f.read()

c = c.replace('kwargs = {a: batch[a]}', 'kwargs = {"text": batch[a]}')
c = c.replace('if b is not None: kwargs[b] = batch[b]', 'if b is not None: kwargs["text_pair"] = batch[b]')

with open("D:/svd_project/Codes/framework/eval_int8.py", "w") as f:
    f.write(c)
