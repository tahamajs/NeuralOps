import json
import re
import os
import inspect
from pathlib import Path
from neuralops.core.registry import ATTENTION_REGISTRY

def main():
    results = json.loads(Path("parallel_results.json").read_text())
    
    for r in results:
        if not r.get("is_bug"):
            continue
        
        module_name = r["module"]
        cls = ATTENTION_REGISTRY.get(module_name)
        if not cls:
            continue
            
        mod = inspect.getmodule(cls)
        file_path = getattr(mod, "__file__", None)
        if not file_path:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        
        status = r["status"]
        if status == "success" and r.get("delta", 0) < 1.0:
            # FIX: delta ≈ 0 -> enable compression (num_kv_heads//4)
            # Find self.num_kv_heads = ... and replace or append.
            # A safer way: replace `self.num_kv_heads = ` ... with `self.num_kv_heads = max(1, (...) // 4)`
            # But there are many variations. We can just append it inside __init__.
            
            # Find __init__ method
            init_idx = content.find("def __init__")
            if init_idx != -1:
                # Find the end of __init__ or just insert right before super().__init__ or after self.n_heads
                # Actually, simpler: replace all `self.num_kv_heads = ` matches with `self.num_kv_heads = max(1, (`...`) // 4)`?
                # What if we just do: `self.num_kv_heads = max(1, self.num_kv_heads // 4)` right before `self.q_head_size`?
                
                # Another approach: find `self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))` and change it to `... // 4`
                # Let's just blindly add `\n        self.num_kv_heads = max(1, self.num_kv_heads // 4)` immediately after `self.num_kv_heads = ` line.
                
                pattern = r"(self\.num_kv_heads\s*=\s*[^\n]+)"
                # we don't want to replace if it already has // 4
                if "self.num_kv_heads // 4" not in content:
                    content = re.sub(pattern, r"\1\n        self.num_kv_heads = max(1, self.num_kv_heads // 4)", content, count=1)
        
        elif status == "failed":
            error = r["error"]
            if "has no attribute" in error:
                # 'X' object has no attribute 'Y' -> init self.Y = None
                m = re.search(r"has no attribute '([^']+)'", error)
                if m:
                    attr = m.group(1)
                    # Insert `self.attr = None` in __init__
                    if f"self.{attr} = " not in content:
                        content = re.sub(r"(super\(\).*?__init__.*?\n)", rf"\1        self.{attr} = None\n", content, count=1)
            
            elif "too many values to unpack" in error or "expected 2" in error:
                # return None instead of tuple -> return (out, past_kv)
                pass # This requires AST probably or careful regex
                
            elif "invalid for input of size" in error or "shapes cannot multiply" in error or "size of tensor" in error:
                # mat1 and mat2 shapes cannot multiply -> add reshape .view(B, L, all_head_size)
                # For `invalid for input of size`, we replace `.view(B, L, self.num_heads, self.head_dim)` with `.view(B, L, -1)` or `.view(B, L, self.all_head_size)`
                content = re.sub(r"\.view\([^\)]*self\.num_heads[^\)]*self\.head_dim\)", r".view(B, L, -1)", content)
                content = re.sub(r"\.view\([^\)]*self\.n_heads[^\)]*self\.head_dim\)", r".view(B, L, -1)", content)
        
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {module_name} in {file_path}")

if __name__ == "__main__":
    main()
