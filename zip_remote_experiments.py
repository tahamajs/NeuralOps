import os, zipfile

src_dir = r"D:\SVD_Project\Codes\framework\experiments"
zip_path = r"D:\SVD_Project\experiments_results.zip"

allowed_exts = {'.json', '.csv', '.md', '.png', '.txt'}

print("Starting zip...")
count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_exts:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, src_dir)
                zipf.write(full_path, rel_path)
                count += 1

print(f"Zipped {count} files. Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
