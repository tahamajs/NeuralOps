import os
import json
import shutil

def sync_and_get_md_files(src_dir, dest_subpath):
    dest_dir = os.path.join("Docs", dest_subpath)
    os.makedirs(dest_dir, exist_ok=True)
    
    md_files = []
    if os.path.exists(src_dir):
        for f in sorted(os.listdir(src_dir)):
            if f.endswith('.md'):
                src_path = os.path.join(src_dir, f)
                dest_path = os.path.join(dest_dir, f)
                
                # Copy file into Docs/content if src and dest differ
                if os.path.abspath(src_path) != os.path.abspath(dest_path):
                    shutil.copy2(src_path, dest_path)
                
                # Extract title from the first heading line if possible
                title = f.replace('.md', '').replace('_', ' ').title()
                try:
                    with open(src_path, 'r', encoding='utf-8') as file:
                        for _ in range(15):
                            line = file.readline()
                            if not line:
                                break
                            line_str = line.strip()
                            if line_str.startswith('# '):
                                title = line_str[2:].split(':')[0].strip()
                                break
                except Exception:
                    pass
                
                # Path is relative to Docs/index.html
                md_files.append({
                    "id": f,
                    "title": title,
                    "path": f"{dest_subpath}/{f}"
                })
                
    return sorted(md_files, key=lambda x: (x['title'].lower(), x['id'].lower()))

def main():
    os.makedirs("Docs/content/framework", exist_ok=True)
    
    # Sync framework compendium and readme
    for f in ["NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md", "README.md"]:
        src = os.path.join("Codes/framework", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("Docs/content/framework", f))
            
    catalog = {
        "attentions": sync_and_get_md_files("Codes/framework/attentions", "content/attentions"),
        "compressions": sync_and_get_md_files("Codes/framework/compression_engine", "content/compressions"),
        "attentions_blog": sync_and_get_md_files("Docs/content/attentions_blog", "content/attentions_blog"),
        "compressions_blog": sync_and_get_md_files("Docs/content/compressions_blog", "content/compressions_blog"),
        "blog_series": sync_and_get_md_files("Docs/blog", "blog"),
        "framework": [
            {
                "id": "NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md",
                "title": "Next-Gen Master Compendium",
                "path": "content/framework/NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md"
            },
            {
                "id": "MASTER_EMPIRICAL_BENCHMARK_REPORT.md",
                "title": "Master Empirical Benchmark Report",
                "path": "content/framework/MASTER_EMPIRICAL_BENCHMARK_REPORT.md"
            },
            {
                "id": "README.md",
                "title": "Framework Documentation",
                "path": "content/framework/README.md"
            }
        ]
    }
    
    with open("Docs/catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4)
        
    print(f"Generated Docs/catalog.json: {len(catalog['attentions'])} attentions, {len(catalog['compressions'])} compressions, {len(catalog['attentions_blog'])} attention blogs, {len(catalog['compressions_blog'])} compression blogs, {len(catalog['blog_series'])} series guides, {len(catalog['framework'])} framework docs.")

if __name__ == "__main__":
    main()
