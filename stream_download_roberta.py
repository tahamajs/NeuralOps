import urllib.request
import os

files = [
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/config.json", "config.json"),
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/tokenizer_config.json", "tokenizer_config.json"),
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/vocab.json", "vocab.json"),
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/merges.txt", "merges.txt"),
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/special_tokens_map.json", "special_tokens_map.json"),
    ("https://huggingface.co/textattack/roberta-base-SST-2/resolve/main/model.safetensors", "model.safetensors"),
]

out_dir = r"D:\svd_project\models\roberta_sst2"
os.makedirs(out_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}
for url, fname in files:
    dest = os.path.join(out_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"Already exists: {fname} ({os.path.getsize(dest)} bytes)")
        continue
    print(f"Downloading {fname} from {url}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, 'wb') as out_f:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024 * 4)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({(downloaded/total)*100:.1f}%)", end="", flush=True)
        print()
    print(f"Saved {fname} ({os.path.getsize(dest)} bytes)")

print("All downloads complete!")
