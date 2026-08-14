import os
import json

def get_md_files(directory):
    md_files = []
    if os.path.exists(directory):
        for f in sorted(os.listdir(directory)):
            if f.endswith('.md'):
                # Extract title from the first line if possible
                title = f.replace('.md', '').replace('_', ' ').title()
                filepath = os.path.join(directory, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        first_line = file.readline().strip()
                        if first_line.startswith('# '):
                            title = first_line[2:].split(':')[0].strip()
                except Exception:
                    pass
                
                md_files.append({
                    "id": f,
                    "title": title,
                    "path": f"../{directory}/{f}"
                })
    # Sort deterministically by lowercase title, then by file ID
    return sorted(md_files, key=lambda x: (x['title'].lower(), x['id'].lower()))

def main():
    attentions_dir = "Codes/framework/attentions"
    compression_dir = "Codes/framework/compression_engine"
    
    catalog = {
        "attentions": get_md_files(attentions_dir),
        "compressions": get_md_files(compression_dir),
        "framework": [
            {
                "id": "NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md",
                "title": "Next-Gen Master Compendium",
                "path": "../Codes/framework/NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md"
            },
            {
                "id": "README.md",
                "title": "Framework Documentation",
                "path": "../Codes/framework/README.md"
            }
        ]
    }
    
    # Save to Docs directory
    os.makedirs("Docs", exist_ok=True)
    with open("Docs/catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4)
        
    print(f"Generated Docs/catalog.json: {len(catalog['attentions'])} attentions, {len(catalog['compressions'])} compression engines, {len(catalog['framework'])} framework docs.")

if __name__ == "__main__":
    main()
