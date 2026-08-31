from pathlib import Path
root=Path(r"C:/Users/Hadi Schumacher/.cache/huggingface/hub")
for d in sorted(root.glob("models--*")):
    blobs=d/"blobs"
    total=sum(p.stat().st_size for p in blobs.glob("*") if p.is_file()) if blobs.exists() else 0
    largest=max((p.stat().st_size for p in blobs.glob("*") if p.is_file()), default=0)
    print(f"{d.name}	total={total}	largest={largest}")
