import os
import glob
import re

attn_dir = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions"

def patch_file(filename, patch_func):
    path = os.path.join(attn_dir, filename)
    if not os.path.exists(path): return
    with open(path, "r") as fp:
        content = fp.read()
    new_content = patch_func(content)
    if new_content != content:
        with open(path, "w") as fp:
            fp.write(new_content)
        print(f"Patched {filename}")

def patch_compressed_mha(content):
    # What's wrong in compressed_mha?
    return content

def patch_nasc(content):
    # nasc block routing already patched in fixer.py, let's verify
    if "top_idx[:, :, :, 0] = 0" not in content and "top_idx = blk_scores.topk" in content:
        content = content.replace("_, top_idx = blk_scores.topk(top_k, dim=-1)",
                                  "_, top_idx = blk_scores.topk(top_k, dim=-1)\n        top_idx[:, :, :, 0] = 0")
    return content

patch_file("nasc_attention.py", patch_nasc)

# Let's ensure repeat_interleave is correct everywhere.
def patch_gqa(content):
    # if it's using repeat_interleave, make sure it's valid.
    return content
