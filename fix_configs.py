import os
paths = [
    'D:/svd_project/results/unified_a3/distilbert/checkpoints/a3/hf_export/config.json',
    'D:/svd_project/results/unified_a3/roberta/checkpoints/a3/hf_export/config.json',
    'D:/svd_project/results/unified_a3/gpt2/checkpoints/a3/hf_export/config.json'
]
for p in paths:
    if os.path.exists(p):
        with open(p, 'r') as f:
            c = f.read()
        if '"tokenizer":' in c and '}' not in c[c.find('"tokenizer":'):]:
            c = c[:c.find('"tokenizer":')] + '"tokenizer_class": "AutoTokenizer"\n}'
            with open(p, 'w') as f:
                f.write(c)
        print('Fixed ' + p)
