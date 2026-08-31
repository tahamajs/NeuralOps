from pathlib import Path
roots=[Path(r"C:/Users/Hadi Schumacher/.cache/huggingface/datasets"), Path(r"D:/SVD_PROJECT/Codes/framework/data/glue_data"), Path(r"D:/SVD_PROJECT/Codes/data_cache")]
for root in roots:
    print(root, root.exists())
    if root.exists():
        print([str(p.relative_to(root)) for p in root.iterdir()][:30])
