from pathlib import Path
p=Path(r"D:/SVD_PROJECT/Codes/framework/run_quality_sweep.py")
s=p.read_text()
needle='                "roberta": "roberta-base",'
if '"qwen":' not in s:
    s=s.replace(needle, needle+'\n                "qwen": "Qwen/Qwen2.5-0.5B-Instruct",')
    p.write_text(s)
print('qwen mapping present:', '"qwen":' in s)
